"""Utilities for N-1 workbench image upgrade survival tests."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.image_stream import ImageStream
from ocp_resources.pod import ExecOnPodError, Pod
from ocp_resources.resource import NamespacedResource
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from packaging.version import InvalidVersion, Version as PackagingVersion
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from utilities.constants import Labels, Timeout
from utilities.infra import get_product_version

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_BASELINE_CM_NAME = "upgrade-n-minus-one-baseline"
UPGRADE_MARKER_FILENAME = ".upgrade-marker"
UPGRADE_MARKER_CONTENT = "n-minus-one-survival"
NOTEBOOK_PORT = 8888
RSTUDIO_BUILDCONFIG_NAME = "rstudio-server-rhel9"
RSTUDIO_BUILD_SECRET_NAME = "rhel-subscription-secret"
RSTUDIO_IMAGE_BUILD_TIMEOUT = Timeout.TIMEOUT_30MIN

_BLOCKED_LOG_KEYWORDS = (
    "Error",
    "error",
    "Warning",
    "warning",
    "Failed",
    "failed",
    "[W ",
    "[E ",
    "[warn] ",
    "[error] ",
    "[crit] ",
    "[alert] ",
    "[emerg] ",
    "Traceback",
)

_ALLOWED_LOG_MESSAGES = (
    "JupyterEventsVersionWarning: The `version` property of an event schema must be a string.",
    "connect() failed (111: Connection refused) while connecting to upstream, client",
    "WARNING: The Jupyter server is listening on all IP addresses and not using encryption.",
    "WARNING: The Jupyter server is listening on all IP addresses and not using authentication.",
    "ServerApp.token config is deprecated in 2.0. Use IdentityProvider.token.",
    "Unable to retrieve mac address (unexpected format)",
)

_RHOAI_VERSIONING_START = PackagingVersion(version="3.4")


class BuildConfig(NamespacedResource):
    """BuildConfig resource (build.openshift.io/v1). Not shipped by ocp_resources."""

    api_group: str = "build.openshift.io"


def _imagestream_tag_sort_key(tag_name: str) -> tuple[int, PackagingVersion] | None:
    """Return a sort key for ImageStream tag names across mixed version schemes."""
    try:
        version = PackagingVersion(version=tag_name)
    except InvalidVersion:
        return None

    if version >= _RHOAI_VERSIONING_START and version.major < 2000:
        return (2, version)
    if version.major >= 2000:
        return (1, version)
    return (0, version)


def _latest_resolved_imagestream_tag(resolved_tag_names: list[str]) -> str | None:
    """Return the highest-versioned resolved ImageStream tag name."""
    versioned_tags: list[tuple[tuple[int, PackagingVersion], str]] = []
    for tag_name in resolved_tag_names:
        sort_key = _imagestream_tag_sort_key(tag_name=tag_name)
        if sort_key is not None:
            versioned_tags.append((sort_key, tag_name))

    if not versioned_tags:
        return None

    versioned_tags.sort(reverse=True)
    return versioned_tags[0][1]


def _get_resolved_imagestream_tags(imagestream_data: dict[str, Any]) -> list[str]:
    """Return ImageStream tag names that have at least one resolved item."""
    return [
        str(status_tag.get("tag"))
        for status_tag in imagestream_data.get("status", {}).get("tags", [])
        if status_tag.get("items")
    ]


def _get_spec_imagestream_tag_names(imagestream_data: dict[str, Any]) -> list[str]:
    """Return tag names declared on the ImageStream spec."""
    return [str(spec_tag.get("name")) for spec_tag in imagestream_data.get("spec", {}).get("tags", [])]


def _resolve_target_image_tag(
    admin_client: DynamicClient,
    resolved_tags: list[str],
    *,
    allow_latest_tag: bool = False,
    spec_tag_names: list[str] | None = None,
) -> str | None:
    """Return the ImageStream tag to use for N-1 image resolution, or None when unavailable."""
    image_tag = py_config.get("workbench_image_tag")
    if image_tag:
        if image_tag in resolved_tags:
            return image_tag
        if allow_latest_tag and image_tag == "latest" and spec_tag_names and "latest" in spec_tag_names:
            return "latest"
        return None

    product_version = get_product_version(admin_client=admin_client)
    semver_tag = f"{product_version.major}.{product_version.minor}"
    if semver_tag in resolved_tags:
        return semver_tag

    versioned_tag = _latest_resolved_imagestream_tag(resolved_tag_names=resolved_tags)
    if versioned_tag:
        return versioned_tag

    if allow_latest_tag:
        if "latest" in resolved_tags:
            return "latest"
        if spec_tag_names and "latest" in spec_tag_names:
            return "latest"

    return None


def _resolve_docker_image_reference_from_status(imagestream_data: dict[str, Any], image_tag: str) -> str | None:
    """Return a resolved dockerImageReference for an ImageStream tag, if imported."""
    for status_tag in imagestream_data.get("status", {}).get("tags", []):
        if status_tag.get("tag") != image_tag:
            continue

        tag_items = status_tag.get("items") or []
        if not tag_items:
            return None

        docker_image_reference = str(tag_items[0].get("dockerImageReference", ""))
        return docker_image_reference or None

    return None


def _resolve_image_from_spec_tag(imagestream_data: dict[str, Any], image_tag: str) -> str | None:
    """Return the external image reference declared on an ImageStream spec tag."""
    for spec_tag in imagestream_data.get("spec", {}).get("tags", []):
        if spec_tag.get("name") != image_tag:
            continue

        tag_from = spec_tag.get("from") or {}
        if tag_from.get("kind") == "DockerImage":
            docker_image = str(tag_from.get("name", ""))
            return docker_image or None

    return None


def _rstudio_build_prerequisite_skip_reason(admin_client: DynamicClient, namespace: str) -> str | None:
    """Return a skip reason when RStudio cannot be built on the cluster."""
    build_config = BuildConfig(
        client=admin_client,
        name=RSTUDIO_BUILDCONFIG_NAME,
        namespace=namespace,
        ensure_exists=False,
    )
    if not build_config.exists:
        return (
            f"RStudio BuildConfig '{RSTUDIO_BUILDCONFIG_NAME}' not found in namespace '{namespace}'"
        )

    build_secret = Secret(
        client=admin_client,
        name=RSTUDIO_BUILD_SECRET_NAME,
        namespace=namespace,
        ensure_exists=False,
    )
    if not build_secret.exists:
        return (
            f"RStudio image is not built yet: secret '{RSTUDIO_BUILD_SECRET_NAME}' not found in "
            f"namespace '{namespace}'. Create the RHEL subscription secret and build with "
            f"'oc start-build {RSTUDIO_BUILDCONFIG_NAME} -n {namespace} --follow' before running tests."
        )

    return None


def _start_imagestream_build(namespace: str, buildconfig_name: str) -> None:
    """Trigger an OpenShift BuildConfig to populate an ImageStream tag."""
    result = subprocess.run(
        ["oc", "start-build", buildconfig_name, "-n", namespace],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Failed to start BuildConfig '{buildconfig_name}' in namespace '{namespace}': "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    LOGGER.info(
        f"Triggered BuildConfig '{buildconfig_name}' in namespace '{namespace}': "
        f"{result.stdout.strip()}"
    )


def _refresh_imagestream_data(admin_client: DynamicClient, imagestream_name: str, namespace: str) -> dict[str, Any]:
    """Return the latest ImageStream resource state."""
    imagestream = ImageStream(client=admin_client, name=imagestream_name, namespace=namespace)
    return imagestream.instance.to_dict()


def _ensure_imagestream_tag_imported(
    admin_client: DynamicClient,
    imagestream_name: str,
    namespace: str,
    image_tag: str,
    *,
    buildconfig_name: str | None = None,
) -> dict[str, Any]:
    """Wait for an ImageStream tag to import, triggering a build when configured."""
    imagestream_data = _refresh_imagestream_data(
        admin_client=admin_client,
        imagestream_name=imagestream_name,
        namespace=namespace,
    )
    if _resolve_docker_image_reference_from_status(imagestream_data=imagestream_data, image_tag=image_tag):
        return imagestream_data

    if buildconfig_name:
        _start_imagestream_build(namespace=namespace, buildconfig_name=buildconfig_name)

    sampler = TimeoutSampler(
        wait_timeout=RSTUDIO_IMAGE_BUILD_TIMEOUT,
        sleep=30,
        func=lambda: True,
    )
    try:
        for _ in sampler:
            imagestream_data = _refresh_imagestream_data(
                admin_client=admin_client,
                imagestream_name=imagestream_name,
                namespace=namespace,
            )
            if _resolve_docker_image_reference_from_status(
                imagestream_data=imagestream_data,
                image_tag=image_tag,
            ):
                return imagestream_data
    except TimeoutExpiredError as error:
        raise TimeoutExpiredError(
            f"ImageStream '{imagestream_name}' tag '{image_tag}' was not imported within "
            f"{RSTUDIO_IMAGE_BUILD_TIMEOUT} seconds"
        ) from error

    raise AssertionError(
        f"ImageStream '{imagestream_name}' tag '{image_tag}' is not imported or resolved."
    )


@dataclass(frozen=True)
class WorkbenchImageSpec:
    """Configuration for a representative workbench IDE under N-1 upgrade testing."""

    ide: str
    imagestream_name: str
    notebook_name: str
    skip_on_upstream: bool = False
    allow_latest_tag: bool = False
    probe_http: bool = True


def get_workbench_image_specs() -> list[WorkbenchImageSpec]:
    """Return the IDE matrix for N-1 survival tests."""
    is_upstream = py_config.get("distribution") == "upstream"
    jupyter_imagestream = "jupyter-minimal-notebook" if is_upstream else "s2i-minimal-notebook"

    return [
        WorkbenchImageSpec(
            ide="jupyterlab",
            imagestream_name=jupyter_imagestream,
            notebook_name="upgrade-n1-jupyterlab",
        ),
        WorkbenchImageSpec(
            ide="code-server",
            imagestream_name="code-server-notebook",
            notebook_name="upgrade-n1-codeserver",
            skip_on_upstream=True,
        ),
        WorkbenchImageSpec(
            ide="rstudio",
            imagestream_name="rstudio-rhel9",
            notebook_name="upgrade-n1-rstudio",
            skip_on_upstream=True,
            allow_latest_tag=True,
            probe_http=False,
        ),
    ]


def should_skip_workbench_spec(
    admin_client: DynamicClient,
    spec: WorkbenchImageSpec,
    *,
    post_upgrade: bool = False,
) -> str | None:
    """Return a skip reason when the IDE cannot be tested on the current cluster."""
    if spec.skip_on_upstream and py_config.get("distribution") == "upstream":
        return f"{spec.ide} ImageStream tests are downstream-only"

    imagestream = ImageStream(
        client=admin_client,
        name=spec.imagestream_name,
        namespace=py_config["applications_namespace"],
    )
    if not imagestream.exists:
        return f"ImageStream '{spec.imagestream_name}' not found in applications namespace"

    if not post_upgrade:
        imagestream_data: dict[str, Any] = imagestream.instance.to_dict()
        resolved_tags = _get_resolved_imagestream_tags(imagestream_data=imagestream_data)
        spec_tag_names = _get_spec_imagestream_tag_names(imagestream_data=imagestream_data)
        target_tag = _resolve_target_image_tag(
            admin_client=admin_client,
            resolved_tags=resolved_tags,
            allow_latest_tag=spec.allow_latest_tag,
            spec_tag_names=spec_tag_names,
        )
        if target_tag is None:
            return (
                f"{spec.ide} ImageStream '{spec.imagestream_name}' has no resolvable image tag "
                f"(resolved tags: {sorted(resolved_tags) or 'none'})"
            )

        if (
            spec.allow_latest_tag
            and target_tag == "latest"
            and _resolve_docker_image_reference_from_status(
                imagestream_data=imagestream_data,
                image_tag=target_tag,
            )
            is None
        ):
            skip_reason = _rstudio_build_prerequisite_skip_reason(
                admin_client=admin_client,
                namespace=py_config["applications_namespace"],
            )
            if skip_reason:
                return skip_reason

    return None


def resolve_n_minus_one_image(admin_client: DynamicClient, spec: WorkbenchImageSpec) -> str:
    """Resolve the N-1 workbench image reference from an ImageStream tag on the source cluster.

    Uses ``workbench_image_tag`` when set, otherwise the current product version tag
    (which equals N-1 on a pre-upgrade cluster). IDEs with ``allow_latest_tag`` fall back
    to the ImageStream ``latest`` tag when no versioned tag is available. RStudio images are
    built in-cluster via BuildConfig and must be imported before use.

    Args:
        admin_client: Cluster client for ImageStream and product version lookups.
        spec: Workbench IDE configuration including ImageStream name.

    Returns:
        Full image reference, preferring a digest-pinned ``dockerImageReference``.

    Raises:
        AssertionError: If the ImageStream or requested tag is missing or unresolved.
    """
    image_tag = py_config.get("workbench_image_tag")
    applications_namespace = py_config["applications_namespace"]
    imagestream = ImageStream(
        client=admin_client,
        name=spec.imagestream_name,
        namespace=applications_namespace,
    )
    assert imagestream.exists, (
        f"ImageStream '{spec.imagestream_name}' not found in namespace '{applications_namespace}'"
    )

    imagestream_data: dict[str, Any] = imagestream.instance.to_dict()
    resolved_tags = _get_resolved_imagestream_tags(imagestream_data=imagestream_data)
    spec_tag_names = _get_spec_imagestream_tag_names(imagestream_data=imagestream_data)

    if not image_tag:
        resolved_tag = _resolve_target_image_tag(
            admin_client=admin_client,
            resolved_tags=resolved_tags,
            allow_latest_tag=spec.allow_latest_tag,
            spec_tag_names=spec_tag_names,
        )
        assert resolved_tag, (
            f"ImageStream '{spec.imagestream_name}' has no resolvable image tag "
            f"(resolved tags: {sorted(resolved_tags) or 'none'})"
        )
        product_version = get_product_version(admin_client=admin_client)
        semver_tag = f"{product_version.major}.{product_version.minor}"
        if resolved_tag == "latest":
            LOGGER.warning(
                f"ImageStream '{spec.imagestream_name}' has no versioned tag for {spec.ide}; "
                "using 'latest' to verify workbench survival across upgrade"
            )
        elif resolved_tag != semver_tag:
            LOGGER.warning(
                f"ImageStream tag '{semver_tag}' not found for {spec.ide}; "
                f"using latest resolved tag '{resolved_tag}'"
            )
        image_tag = resolved_tag

    docker_image_reference = _resolve_docker_image_reference_from_status(
        imagestream_data=imagestream_data,
        image_tag=image_tag,
    )
    if docker_image_reference:
        LOGGER.info(
            f"Resolved N-1 image for {spec.ide}: {spec.imagestream_name}:{image_tag} -> {docker_image_reference}"
        )
        return docker_image_reference

    if spec.allow_latest_tag and image_tag == "latest":
        imagestream_data = _ensure_imagestream_tag_imported(
            admin_client=admin_client,
            imagestream_name=spec.imagestream_name,
            namespace=applications_namespace,
            image_tag=image_tag,
            buildconfig_name=RSTUDIO_BUILDCONFIG_NAME,
        )
        docker_image_reference = _resolve_docker_image_reference_from_status(
            imagestream_data=imagestream_data,
            image_tag=image_tag,
        )
        if docker_image_reference:
            LOGGER.info(
                f"Resolved N-1 image for {spec.ide}: {spec.imagestream_name}:{image_tag} -> "
                f"{docker_image_reference}"
            )
            return docker_image_reference

    spec_reference = _resolve_image_from_spec_tag(imagestream_data=imagestream_data, image_tag=image_tag)
    if spec_reference:
        LOGGER.warning(
            f"ImageStream tag '{image_tag}' is not resolved in status for {spec.ide}; "
            f"using spec reference {spec_reference}"
        )
        return spec_reference

    raise AssertionError(
        f"ImageStream '{spec.imagestream_name}' tag '{image_tag}' is not imported or resolved. "
        f"Cannot launch N-1 workbench for {spec.ide}."
    )


@contextmanager
def notebook_service_account(
    client: DynamicClient,
    name: str,
    namespace: str,
    *,
    teardown: bool = True,
) -> Generator[ServiceAccount, Any, Any]:
    """Ensure the per-notebook ServiceAccount exists before deploying a Notebook CR.

    The Kubeflow notebook controller creates the StatefulSet immediately, but on some
    RHOAI versions the ODH controller creates auth resources asynchronously. Pre-creating
    the ServiceAccount avoids pod scheduling failures when the SA is not found.

    Args:
        client: Kubernetes client for the target namespace.
        name: ServiceAccount name (matches the notebook name).
        namespace: Target namespace.
        teardown: Whether to delete the ServiceAccount on context exit.

    Yields:
        The existing or newly created ServiceAccount.
    """
    existing_sa = ServiceAccount(client=client, name=name, namespace=namespace, ensure_exists=False)
    if existing_sa.exists:
        yield existing_sa
        return

    with ServiceAccount(client=client, name=name, namespace=namespace, teardown=teardown) as service_account:
        yield service_account


def build_notebook_dict(
    namespace: str,
    name: str,
    image_path: str,
    extra_annotations: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a Notebook CR dict for the kubeflow.org/v1 API.

    Args:
        namespace: Target namespace for the Notebook.
        name: Notebook resource name (also used for PVC claim, service account, container).
        image_path: Full container image reference.
        extra_annotations: Optional annotations merged into metadata.

    Returns:
        A dict suitable for passing to ``Notebook(kind_dict=...)``.
    """
    probe_config = {
        "failureThreshold": 3,
        "httpGet": {
            "path": f"/notebook/{namespace}/{name}/api",
            "port": "notebook-port",
            "scheme": "HTTP",
        },
        "initialDelaySeconds": 10,
        "periodSeconds": 5,
        "successThreshold": 1,
        "timeoutSeconds": 1,
    }

    annotations: dict[str, str] = {
        Labels.Notebook.INJECT_AUTH: "true",
        "opendatahub.io/accelerator-name": "",
        "notebooks.opendatahub.io/last-image-selection": image_path,
    }
    if extra_annotations:
        annotations.update(extra_annotations)

    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": "Notebook",
        "metadata": {
            "annotations": annotations,
            "labels": {
                Labels.Openshift.APP: name,
                Labels.OpenDataHub.DASHBOARD: "true",
                "opendatahub.io/odh-managed": "true",
            },
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "template": {
                "spec": {
                    "affinity": {},
                    "containers": [
                        {
                            "env": [
                                {
                                    "name": "NOTEBOOK_ARGS",
                                    "value": "--ServerApp.port=8888\n"
                                    "                  "
                                    "--ServerApp.token=''\n"
                                    "                  "
                                    "--ServerApp.password=''\n"
                                    "                  "
                                    f"--ServerApp.base_url=/notebook/{namespace}/{name}\n"
                                    "                  "
                                    "--ServerApp.quit_button=False\n",
                                },
                                {"name": "JUPYTER_IMAGE", "value": image_path},
                            ],
                            "image": image_path,
                            "imagePullPolicy": "Always",
                            "livenessProbe": probe_config,
                            "name": name,
                            "ports": [{"containerPort": 8888, "name": "notebook-port", "protocol": "TCP"}],
                            "readinessProbe": probe_config,
                            "resources": {
                                "limits": {"cpu": "2", "memory": "4Gi"},
                                "requests": {"cpu": "1", "memory": "1Gi"},
                            },
                            "volumeMounts": [
                                {"mountPath": "/opt/app-root/src", "name": name},
                                {"mountPath": "/dev/shm", "name": "shm"},
                            ],
                            "workingDir": "/opt/app-root/src",
                        },
                    ],
                    "enableServiceLinks": False,
                    "serviceAccountName": name,
                    "volumes": [
                        {"name": name, "persistentVolumeClaim": {"claimName": name}},
                        {"emptyDir": {"medium": "Memory"}, "name": "shm"},
                    ],
                }
            }
        },
    }


def grab_and_check_workbench_logs(
    pod: Pod,
    container_name: str,
    *,
    extra_allowed: list[str] | None = None,
) -> None:
    """Scan workbench container logs for unexpected errors or warnings.

    Ported from ``workbench_image_test.grab_and_check_logs`` in opendatahub-io/notebooks.

    Args:
        pod: Running notebook pod.
        container_name: Primary workbench container name.
        extra_allowed: Additional log substrings to waive.

    Raises:
        AssertionError: If blocked keywords appear in logs outside the allowlist.
    """
    allowed_messages = list(_ALLOWED_LOG_MESSAGES)
    if extra_allowed:
        allowed_messages.extend(extra_allowed)

    time.sleep(3)
    log_output = pod.log(container=container_name)
    failed_lines: list[str] = []

    for line in log_output.splitlines():
        if not any(keyword in line for keyword in _BLOCKED_LOG_KEYWORDS):
            continue
        if any(allowed in line for allowed in allowed_messages):
            LOGGER.debug(f"Waived log message: {line}")
            continue
        LOGGER.error(f"Unexpected log keyword in: {line}")
        failed_lines.append(line)

    if failed_lines:
        joined_lines = "\n".join(failed_lines)
        raise AssertionError(
            f"Unexpected log messages ({len(failed_lines)}) from pod '{pod.name}' "
            f"container '{container_name}':\n{joined_lines}"
        )


def wait_for_http_inside_container(
    pod: Pod,
    container_name: str,
    *,
    port: int = NOTEBOOK_PORT,
    path: str = "/",
    timeout: float = 120,
) -> None:
    """Poll HTTP readiness from inside the workbench container.

    Ported from ``workbench_image_test._wait_for_http_inside_container``.

    Args:
        pod: Running notebook pod.
        container_name: Primary workbench container name.
        port: HTTP port exposed by the IDE.
        path: URL path to request (defaults to root).
        timeout: Maximum wait time in seconds.

    Raises:
        AssertionError: If the container exits before becoming ready.
        TimeoutError: If HTTP does not respond within the timeout.
    """
    check_script = f"import urllib.request; urllib.request.urlopen('http://localhost:{port}{path}', timeout=2)"
    deadline = time.monotonic() + timeout

    while time.monotonic() <= deadline:
        pod_phase = pod.instance.status.phase
        if pod_phase in {pod.Status.FAILED, pod.Status.SUCCEEDED}:
            raise AssertionError(f"Pod '{pod.name}' is not running before HTTP check (phase={pod_phase})")

        try:
            pod.execute(container=container_name, command=["python", "-c", check_script], timeout=30)
            return
        except ExecOnPodError:
            time.sleep(2)

    raise TimeoutError(
        f"HTTP server on port {port} path '{path}' did not become ready within {timeout}s "
        f"in pod '{pod.name}' container '{container_name}'"
    )


def verify_workbench_survival(
    pod: Pod,
    container_name: str,
    *,
    http_path: str | None = "/",
) -> None:
    """Run workbench survival checks after upgrade: logs, optional HTTP, logs.

    Args:
        pod: Running notebook pod after upgrade.
        container_name: Primary workbench container name.
        http_path: In-container HTTP path to probe. When ``None``, skip the HTTP step
            (for example RStudio serves via nginx and does not use Jupyter-style paths).
    """
    grab_and_check_workbench_logs(pod=pod, container_name=container_name)
    if http_path is not None:
        wait_for_http_inside_container(pod=pod, container_name=container_name, path=http_path)
    grab_and_check_workbench_logs(pod=pod, container_name=container_name)


def write_pvc_upgrade_marker(pod: Pod, container_name: str) -> None:
    """Write a marker file to the workbench PVC before upgrade."""
    command = [
        "sh",
        "-c",
        f"echo {UPGRADE_MARKER_CONTENT} > /opt/app-root/src/{UPGRADE_MARKER_FILENAME}",
    ]
    pod.execute(container=container_name, command=command, timeout=60)


def read_pvc_upgrade_marker(pod: Pod, container_name: str) -> str:
    """Read the pre-upgrade marker file from the workbench PVC."""
    output = pod.execute(
        container=container_name,
        command=["cat", f"/opt/app-root/src/{UPGRADE_MARKER_FILENAME}"],
        timeout=60,
    )
    return output.strip()


def merge_baseline_entry(
    admin_client: DynamicClient,
    namespace: str,
    notebook_name: str,
    baseline_entry: dict[str, Any],
) -> None:
    """Merge one notebook baseline entry into the shared upgrade ConfigMap."""
    from ocp_resources.config_map import ConfigMap

    cm = ConfigMap(client=admin_client, name=UPGRADE_BASELINE_CM_NAME, namespace=namespace)
    existing_data: dict[str, Any] = {}

    if cm.exists:
        raw = (cm.instance.data or {}).get("baseline", "{}")
        existing_data = json.loads(raw)

    existing_data[notebook_name] = baseline_entry

    if cm.exists:
        resource_dict = cm.instance.to_dict()
        resource_dict.setdefault("data", {})
        resource_dict["data"]["baseline"] = json.dumps(existing_data)
        cm.update(resource_dict=resource_dict)
    else:
        ConfigMap(
            client=admin_client,
            name=UPGRADE_BASELINE_CM_NAME,
            namespace=namespace,
            data={"baseline": json.dumps(existing_data)},
        ).deploy()


def load_baseline_entry(
    admin_client: DynamicClient,
    namespace: str,
    notebook_name: str,
) -> dict[str, Any]:
    """Load a single notebook baseline entry from the upgrade ConfigMap."""
    from ocp_resources.config_map import ConfigMap

    cm = ConfigMap(client=admin_client, name=UPGRADE_BASELINE_CM_NAME, namespace=namespace)
    assert cm.exists, (
        f"Baseline ConfigMap '{UPGRADE_BASELINE_CM_NAME}' not found in namespace '{namespace}'. "
        "Ensure pre-upgrade tests ran successfully."
    )

    raw = (cm.instance.data or {}).get("baseline")
    assert raw, f"Baseline ConfigMap '{UPGRADE_BASELINE_CM_NAME}' has no 'baseline' key."

    baselines: dict[str, Any] = json.loads(raw)
    baseline = baselines.get(notebook_name)
    assert baseline is not None, (
        f"Missing baseline for notebook '{notebook_name}'. Available: {sorted(baselines.keys())}"
    )
    return baseline
