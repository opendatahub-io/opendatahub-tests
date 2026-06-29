"""Utilities for N-1 workbench image upgrade survival tests."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.image_stream import ImageStream
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import ExecOnPodError, Pod
from ocp_resources.resource import NamespacedResource, ResourceEditor
from ocp_resources.secret import Secret
from pytest_testconfig import config as py_config
from semver import Version
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.workbenches.notebooks_server.controller.utils import StatefulSet
from utilities.constants import INTERNAL_IMAGE_REGISTRY_PATH, Labels, Timeout
from utilities.general import collect_pod_information
from utilities.infra import check_internal_image_registry_available, get_product_version
from utilities.resources.http_route import HTTPRoute
from utilities.resources.reference_grant import ReferenceGrant

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_NAMESPACE = "upgrade-notebook-images"
UPGRADE_BASELINE_CM_NAME = "upgrade-n-minus-one-baseline"
UPGRADE_MARKER_FILENAME = ".upgrade-marker"
UPGRADE_MARKER_CONTENT = "n-minus-one-survival"
NOTEBOOK_PORT = 8888
REFERENCE_GRANT_NAME = "notebook-httproute-access"
TRUSTED_CA_BUNDLE_NAME = "workbench-trusted-ca-bundle"
PIPELINE_RUNTIME_IMAGES_NAME = "pipeline-runtime-images"
RSTUDIO_BUILDCONFIG_NAME = "rstudio-server-rhel9"
RSTUDIO_BUILD_SECRET_NAME = "rhel-subscription-secret"  # pragma: allowlist secret
RSTUDIO_IMAGE_BUILD_TIMEOUT = Timeout.TIMEOUT_30MIN

SEMVER_TAG_PATTERN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)$")
LEGACY_TAG_PATTERN = re.compile(r"^(?P<year>\d{4})\.(?P<minor>\d+)$")

BLOCKED_LOG_KEYWORDS = (
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

ALLOWED_LOG_MESSAGES = (
    "connect() failed (111: Connection refused) while connecting to upstream, client",
    "Skipping trusted CA bundle mount because the ConfigMap is not available",
    "WARNING: skipping notebook trusted CA setup because no bundle was mounted",
    "JupyterEventsVersionWarning: The `version` property of an event schema must be a string.",
    "ServerApp.token config is deprecated in 2.0. Use IdentityProvider.token.",
    "WARNING: The Jupyter server is listening on all IP addresses and not using encryption.",
    "WARNING: The Jupyter server is listening on all IP addresses and not using authentication.",
    "Unable to retrieve mac address (unexpected format)",
)

_SENSITIVE_LOG_VALUE_RE = re.compile(
    r"(?i)\b(token|access[_-]?token|refresh[_-]?token|password|passwd|secret)=([^\&\s]+)"
)
_SENSITIVE_HEADER_RE = re.compile(r"(?i)\b(authorization|cookie):\s*[^\r\n]+")
_SECRET_PATTERN = re.compile(r"(?i)\b(authorization|bearer|token|password|secret|api[-_]?key)\b\s*[:=]\s*\S+")


class BuildConfig(NamespacedResource):
    """BuildConfig resource (build.openshift.io/v1). Not shipped by ocp_resources."""

    api_group: str = "build.openshift.io"


def _redact_log_line(line: str) -> str:
    """Redact common secret-bearing values before logs reach CI output."""
    line = _SENSITIVE_LOG_VALUE_RE.sub(repl=r"\1=<redacted>", string=line)
    line = _SENSITIVE_HEADER_RE.sub(repl=r"\1: <redacted>", string=line)
    return _SECRET_PATTERN.sub(repl=r"\1=[REDACTED]", string=line)


def _applications_namespace() -> str:
    """Return the configured applications namespace at runtime."""
    return str(py_config["applications_namespace"])


@dataclass(frozen=True)
class WorkbenchImageSpec:
    """Configuration for a representative workbench IDE under N-1 upgrade testing."""

    ide: str
    imagestream_name: str
    notebook_name: str
    baseline_prefix: str
    pvc_name: str
    skip_on_upstream: bool = False
    require_eus_track: bool = False
    resolve_imagestream_dynamically: bool = False
    allow_build_import: bool = False
    probe_http: bool = True


@dataclass(frozen=True)
class ResolvedWorkbenchImage:
    """Resolved image metadata for a workbench ImageStream tag."""

    imagestream_name: str
    tag_name: str
    image_url: str
    image_selection: str
    image_digest: str
    build_commit: str | None = None


@dataclass(frozen=True)
class WorkbenchImageBaseline:
    """Serialized pre-upgrade workbench baseline persisted in a ConfigMap."""

    creation_timestamp: str
    image_tag: str
    image_url: str
    image_digest: str
    pod_image_digest: str
    last_image_selection: str
    pod_name: str
    restart_counts: dict[str, int]
    notebook_generation: int
    upgrade_marker: str = UPGRADE_MARKER_CONTENT

    def to_configmap_data(self, prefix: str) -> dict[str, str]:
        """Convert the baseline into ConfigMap-friendly string data."""
        return {
            f"{prefix}_creation_timestamp": self.creation_timestamp,
            f"{prefix}_image_tag": self.image_tag,
            f"{prefix}_image_url": self.image_url,
            f"{prefix}_image_digest": self.image_digest,
            f"{prefix}_pod_image_digest": self.pod_image_digest,
            f"{prefix}_last_image_selection": self.last_image_selection,
            f"{prefix}_pod_name": self.pod_name,
            f"{prefix}_restart_counts": json.dumps(self.restart_counts, sort_keys=True),
            f"{prefix}_notebook_generation": str(self.notebook_generation),
            f"{prefix}_upgrade_marker": self.upgrade_marker,
        }

    @classmethod
    def from_configmap_data(cls, prefix: str, data: dict[str, str]) -> WorkbenchImageBaseline:
        """Build a baseline object from ConfigMap string data."""
        required_keys = (
            f"{prefix}_creation_timestamp",
            f"{prefix}_image_tag",
            f"{prefix}_image_url",
            f"{prefix}_image_digest",
            f"{prefix}_pod_image_digest",
            f"{prefix}_last_image_selection",
            f"{prefix}_pod_name",
            f"{prefix}_restart_counts",
            f"{prefix}_notebook_generation",
            f"{prefix}_upgrade_marker",
        )
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise AssertionError(f"Baseline data for '{prefix}' is incomplete: missing {missing_keys}")

        return cls(
            creation_timestamp=data[f"{prefix}_creation_timestamp"],
            image_tag=data[f"{prefix}_image_tag"],
            image_url=data[f"{prefix}_image_url"],
            image_digest=data[f"{prefix}_image_digest"],
            pod_image_digest=data[f"{prefix}_pod_image_digest"],
            last_image_selection=data[f"{prefix}_last_image_selection"],
            pod_name=data[f"{prefix}_pod_name"],
            restart_counts=json.loads(data[f"{prefix}_restart_counts"]),
            notebook_generation=int(data[f"{prefix}_notebook_generation"]),
            upgrade_marker=data[f"{prefix}_upgrade_marker"],
        )


def get_workbench_image_specs() -> list[WorkbenchImageSpec]:
    """Return the IDE matrix for N-1 survival tests."""
    is_upstream = py_config.get("distribution") == "upstream"
    jupyter_imagestream = "jupyter-minimal-notebook" if is_upstream else "s2i-minimal-notebook"

    return [
        WorkbenchImageSpec(
            ide="jupyterlab",
            imagestream_name=jupyter_imagestream,
            notebook_name="upgrade-n1-jupyterlab",
            baseline_prefix="jupyterlab",
            pvc_name="upgrade-n1-jupyterlab-storage",
        ),
        WorkbenchImageSpec(
            ide="code-server",
            imagestream_name="code-server-notebook",
            notebook_name="upgrade-n1-codeserver",
            baseline_prefix="codeserver",
            pvc_name="upgrade-n1-codeserver-storage",
            skip_on_upstream=True,
        ),
        WorkbenchImageSpec(
            ide="rstudio",
            imagestream_name="rstudio-rhel9",
            notebook_name="upgrade-n1-rstudio",
            baseline_prefix="rstudio",
            pvc_name="upgrade-n1-rstudio-storage",
            skip_on_upstream=True,
            require_eus_track=True,
            resolve_imagestream_dynamically=True,
            allow_build_import=True,
            probe_http=False,
        ),
    ]


def resolve_workbench_upgrade_track(admin_client: DynamicClient) -> str:
    """Return the configured workbench upgrade track."""
    if explicit_track := str(py_config.get("workbench_upgrade_track", "")).strip().lower():
        allowed_tracks = {"stable", "eus"}
        if explicit_track not in allowed_tracks:
            raise ValueError(
                f"Unsupported workbench_upgrade_track '{explicit_track}'. Expected one of: {sorted(allowed_tracks)}"
            )
        return explicit_track
    if is_legacy_track_tag(tag_name=str(py_config.get("workbench_image_tag", "")).strip()):
        return "eus"
    current_product_version = get_product_version(admin_client=admin_client)
    return "eus" if current_product_version.major < 3 else "stable"


def effective_imagestream_name(admin_client: DynamicClient, spec: WorkbenchImageSpec) -> str:
    """Return the ImageStream name for a workbench spec, resolving dynamic names when needed."""
    if spec.resolve_imagestream_dynamically:
        if imagestream_name := find_rstudio_imagestream_name(admin_client=admin_client):
            return imagestream_name
        raise AssertionError("Legacy RStudio ImageStream is not present on this cluster")
    return spec.imagestream_name


def should_skip_workbench_spec(
    admin_client: DynamicClient,
    spec: WorkbenchImageSpec,
    *,
    post_upgrade: bool = False,
    workbench_upgrade_track: str | None = None,
) -> str | None:
    """Return a skip reason when the IDE cannot be tested on the current cluster."""
    if spec.skip_on_upstream and py_config.get("distribution") == "upstream":
        return f"{spec.ide} ImageStream tests are downstream-only"

    track = workbench_upgrade_track or resolve_workbench_upgrade_track(admin_client=admin_client)
    if spec.require_eus_track and track != "eus":
        return f"{spec.ide} workbench survival coverage is only supported on the EUS upgrade track"

    try:
        imagestream_name = effective_imagestream_name(admin_client=admin_client, spec=spec)
    except AssertionError as error:
        return str(error)

    if post_upgrade:
        return None

    try:
        resolved_image = resolve_n_minus_one_image(
            admin_client=admin_client,
            imagestream_name=imagestream_name,
        )
    except AssertionError as error:
        if spec.allow_build_import:
            skip_reason = _rstudio_build_prerequisite_skip_reason(
                admin_client=admin_client,
                namespace=_applications_namespace(),
            )
            if skip_reason:
                return skip_reason
        return str(error)

    if spec.require_eus_track and not is_legacy_track_tag(tag_name=resolved_image.tag_name):
        return f"{spec.ide} workbench survival tests require a legacy EUS workbench image tag"

    return None


def _parse_semver_tag(tag_name: str) -> tuple[int, int] | None:
    """Parse a stable ``major.minor`` ImageStream tag."""
    if match := SEMVER_TAG_PATTERN.fullmatch(tag_name):
        return int(match.group("major")), int(match.group("minor"))
    return None


def _parse_legacy_tag(tag_name: str) -> tuple[int, int] | None:
    """Parse a legacy ``year.release`` ImageStream tag."""
    if match := LEGACY_TAG_PATTERN.fullmatch(tag_name):
        return int(match.group("year")), int(match.group("minor"))
    return None


def is_legacy_track_tag(tag_name: str) -> bool:
    """Return whether the tag belongs to the legacy EUS track."""
    return _parse_legacy_tag(tag_name=tag_name) is not None


def _get_imagestream_status_tag_data(imagestream_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index ImageStream status tags by tag name."""
    return {
        str(status_tag["tag"]): status_tag
        for status_tag in imagestream_data.get("status", {}).get("tags", [])
        if status_tag.get("tag")
    }


def _get_imagestream_spec_tag_data(imagestream_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index ImageStream spec tags by tag name."""
    return {
        str(spec_tag["name"]): spec_tag
        for spec_tag in imagestream_data.get("spec", {}).get("tags", [])
        if spec_tag.get("name")
    }


def _resolve_tag_digest(status_tag_data: dict[str, Any], imagestream_name: str, tag_name: str) -> str:
    """Extract a digest reference from ImageStream status data."""
    for item in status_tag_data.get("items") or []:
        docker_image_reference = str(item.get("dockerImageReference", ""))
        if "@sha256:" in docker_image_reference:
            return docker_image_reference.split("@", maxsplit=1)[1]

    raise AssertionError(
        f"ImageStream {imagestream_name}:{tag_name} does not have a resolved dockerImageReference in status.tags.items"
    )


def _resolve_image_repository(
    admin_client: DynamicClient, imagestream_data: dict[str, Any], imagestream_name: str
) -> str:
    """Resolve the repository used by the Dashboard for a workbench ImageStream."""
    status_data = imagestream_data.get("status", {})
    if image_repository := status_data.get("dockerImageRepository") or status_data.get("publicDockerImageRepository"):
        return str(image_repository)

    if check_internal_image_registry_available(admin_client=admin_client):
        return f"{INTERNAL_IMAGE_REGISTRY_PATH}/{_applications_namespace()}/{imagestream_name}"

    raise AssertionError(f"ImageStream {imagestream_name} does not expose a usable docker image repository")


def _resolve_requested_tag_name(
    imagestream_name: str,
    status_tag_data: dict[str, dict[str, Any]],
    current_product_version: Version,
) -> str:
    """Resolve the tag that should survive the upcoming upgrade."""
    if requested_tag_name := str(py_config.get("workbench_image_tag", "")).strip():
        if requested_tag_name not in status_tag_data:
            available_tags = sorted(status_tag_data)
            raise AssertionError(
                f"Requested workbench_image_tag '{requested_tag_name}' does not exist on ImageStream "
                f"{imagestream_name}. Available tags: {available_tags}"
            )
        return requested_tag_name

    requested_track = str(py_config.get("workbench_upgrade_track", "")).strip().lower()
    if not requested_track:
        requested_track = "eus" if current_product_version.major < 3 else "stable"
    if requested_track not in {"stable", "eus"}:
        raise ValueError("workbench_upgrade_track must be either 'stable' or 'eus'")

    if requested_track == "eus":
        legacy_tags = sorted(
            (tag_name for tag_name in status_tag_data if _parse_legacy_tag(tag_name=tag_name)),
            key=lambda tag_name: _parse_legacy_tag(tag_name=tag_name) or (0, 0),
        )
        if not legacy_tags:
            raise AssertionError(f"ImageStream {imagestream_name} does not have any legacy EUS tags")
        return legacy_tags[-1]

    stable_tags = sorted(
        (tag_name for tag_name in status_tag_data if _parse_semver_tag(tag_name=tag_name)),
        key=lambda tag_name: _parse_semver_tag(tag_name=tag_name) or (0, 0),
    )
    current_tag_name = f"{current_product_version.major}.{current_product_version.minor}"
    if current_tag_name in status_tag_data:
        return current_tag_name
    if stable_tags:
        return stable_tags[-1]

    raise AssertionError(f"ImageStream {imagestream_name} does not have any stable semver tags")


def _rstudio_build_prerequisite_skip_reason(admin_client: DynamicClient, namespace: str) -> str | None:
    """Return a skip reason when RStudio cannot be built on the cluster."""
    build_config = BuildConfig(
        client=admin_client,
        name=RSTUDIO_BUILDCONFIG_NAME,
        namespace=namespace,
        ensure_exists=False,
    )
    if not build_config.exists:
        return f"RStudio BuildConfig '{RSTUDIO_BUILDCONFIG_NAME}' not found in namespace '{namespace}'"

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
            f"namespace '{namespace}'. Instantiate BuildConfig '{RSTUDIO_BUILDCONFIG_NAME}' before running tests."
        )

    return None


def _start_imagestream_build(
    admin_client: DynamicClient,
    namespace: str,
    buildconfig_name: str,
) -> None:
    """Trigger an OpenShift BuildConfig to populate an ImageStream tag."""
    build_config = BuildConfig(
        client=admin_client,
        name=buildconfig_name,
        namespace=namespace,
    )
    build_request_body = json.dumps({"kind": "BuildRequest", "apiVersion": "build.openshift.io/v1"})
    try:
        build_config.api_request(
            method="POST",
            action="instantiate",
            url=f"/apis/build.openshift.io/v1/namespaces/{namespace}/buildconfigs/{buildconfig_name}",
            body=build_request_body,
            headers={"Content-Type": "application/json"},
        )
    except Exception as error:
        raise AssertionError(
            f"Failed to instantiate BuildConfig '{buildconfig_name}' in namespace '{namespace}'"
        ) from error

    LOGGER.info(f"Instantiated BuildConfig '{buildconfig_name}' in namespace '{namespace}'")


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
    status_tag_data = _get_imagestream_status_tag_data(imagestream_data=imagestream_data)
    if image_tag in status_tag_data:
        return imagestream_data

    if buildconfig_name:
        _start_imagestream_build(
            admin_client=admin_client,
            namespace=namespace,
            buildconfig_name=buildconfig_name,
        )

    try:
        for _ in TimeoutSampler(wait_timeout=RSTUDIO_IMAGE_BUILD_TIMEOUT, sleep=30, func=lambda: True):
            imagestream_data = _refresh_imagestream_data(
                admin_client=admin_client,
                imagestream_name=imagestream_name,
                namespace=namespace,
            )
            status_tag_data = _get_imagestream_status_tag_data(imagestream_data=imagestream_data)
            if image_tag in status_tag_data:
                _resolve_tag_digest(
                    status_tag_data=status_tag_data[image_tag],
                    imagestream_name=imagestream_name,
                    tag_name=image_tag,
                )
                return imagestream_data
    except TimeoutExpiredError as error:
        raise TimeoutExpiredError(
            f"ImageStream '{imagestream_name}' tag '{image_tag}' was not imported within "
            f"{RSTUDIO_IMAGE_BUILD_TIMEOUT} seconds"
        ) from error

    raise AssertionError(f"ImageStream '{imagestream_name}' tag '{image_tag}' is not imported or resolved.")


def resolve_n_minus_one_image(admin_client: DynamicClient, imagestream_name: str) -> ResolvedWorkbenchImage:
    """Resolve the workbench image tag that should survive the upgrade."""
    applications_namespace = _applications_namespace()
    imagestream = ImageStream(client=admin_client, name=imagestream_name, namespace=applications_namespace)
    if not imagestream.exists:
        raise AssertionError(f"ImageStream {imagestream_name} does not exist in namespace {applications_namespace}")

    imagestream_data = imagestream.instance.to_dict()
    status_tag_data = _get_imagestream_status_tag_data(imagestream_data=imagestream_data)
    spec_tag_data = _get_imagestream_spec_tag_data(imagestream_data=imagestream_data)
    current_product_version = get_product_version(admin_client=admin_client)
    tag_name = _resolve_requested_tag_name(
        imagestream_name=imagestream_name,
        status_tag_data=status_tag_data,
        current_product_version=current_product_version,
    )
    if tag_name not in status_tag_data:
        if tag_name == "latest":
            imagestream_data = _ensure_imagestream_tag_imported(
                admin_client=admin_client,
                imagestream_name=imagestream_name,
                namespace=applications_namespace,
                image_tag=tag_name,
                buildconfig_name=RSTUDIO_BUILDCONFIG_NAME,
            )
            status_tag_data = _get_imagestream_status_tag_data(imagestream_data=imagestream_data)
            spec_tag_data = _get_imagestream_spec_tag_data(imagestream_data=imagestream_data)
        if tag_name not in status_tag_data:
            raise AssertionError(
                f"ImageStream {imagestream_name}:{tag_name} is missing from status.tags "
                "and cannot be used for upgrade tests"
            )

    image_repository = _resolve_image_repository(
        admin_client=admin_client,
        imagestream_data=imagestream_data,
        imagestream_name=imagestream_name,
    )
    tag_digest = _resolve_tag_digest(
        status_tag_data=status_tag_data[tag_name],
        imagestream_name=imagestream_name,
        tag_name=tag_name,
    )
    build_commit = spec_tag_data.get(tag_name, {}).get("annotations", {}).get("opendatahub.io/notebook-build-commit")

    return ResolvedWorkbenchImage(
        imagestream_name=imagestream_name,
        tag_name=tag_name,
        image_url=f"{image_repository}:{tag_name}",
        image_selection=f"{imagestream_name}:{tag_name}",
        image_digest=tag_digest,
        build_commit=str(build_commit) if build_commit else None,
    )


def resolve_workbench_image(admin_client: DynamicClient, spec: WorkbenchImageSpec) -> ResolvedWorkbenchImage:
    """Resolve the N-1 image for a parametrized workbench IDE spec."""
    imagestream_name = effective_imagestream_name(admin_client=admin_client, spec=spec)
    return resolve_n_minus_one_image(admin_client=admin_client, imagestream_name=imagestream_name)


def find_rstudio_imagestream_name(admin_client: DynamicClient) -> str | None:
    """Return the legacy RStudio ImageStream name when it exists."""
    imagestreams = ImageStream.get(
        client=admin_client,
        namespace=_applications_namespace(),
        label_selector="opendatahub.io/notebook-image=true,platform.opendatahub.io/part-of=workbenches",
    )

    for imagestream in imagestreams:
        if "rstudio" in imagestream.name.lower():
            return imagestream.name

    return None


def build_n1_notebook_dict(
    namespace: str,
    notebook_name: str,
    pvc_name: str,
    image: ResolvedWorkbenchImage,
) -> dict[str, Any]:
    """Build a dashboard-faithful Notebook CR for workbench upgrade tests."""
    probe_path = f"/notebook/{namespace}/{notebook_name}/api"
    probe_config = {
        "failureThreshold": 3,
        "httpGet": {
            "path": probe_path,
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
        "notebooks.opendatahub.io/last-image-selection": image.image_selection,
        "openshift.io/display-name": notebook_name,
        "openshift.io/description": "",
        "opendatahub.io/workbench-image-namespace": "",
    }
    if image.build_commit:
        annotations["notebooks.opendatahub.io/last-image-version-git-commit-selection"] = image.build_commit

    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": "Notebook",
        "metadata": {
            "annotations": annotations,
            "finalizers": [
                "notebook.opendatahub.io/httproute-cleanup",
                "notebook.opendatahub.io/referencegrant-cleanup",
                "notebook.opendatahub.io/kube-rbac-proxy-cleanup",
            ],
            "labels": {
                Labels.Openshift.APP: notebook_name,
                Labels.OpenDataHub.DASHBOARD: "true",
                "opendatahub.io/odh-managed": "true",
            },
            "name": notebook_name,
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
                                    f"--ServerApp.base_url=/notebook/{namespace}/{notebook_name}\n"
                                    "                  "
                                    "--ServerApp.quit_button=False\n",
                                },
                                {"name": "JUPYTER_IMAGE", "value": image.image_url},
                                {"name": "PIP_CERT", "value": "/etc/pki/tls/custom-certs/ca-bundle.crt"},
                                {"name": "REQUESTS_CA_BUNDLE", "value": "/etc/pki/tls/custom-certs/ca-bundle.crt"},
                                {"name": "SSL_CERT_FILE", "value": "/etc/pki/tls/custom-certs/ca-bundle.crt"},
                                {"name": "PIPELINES_SSL_SA_CERTS", "value": "/etc/pki/tls/custom-certs/ca-bundle.crt"},
                                {
                                    "name": "KF_PIPELINES_SSL_SA_CERTS",
                                    "value": "/etc/pki/tls/custom-certs/ca-bundle.crt",
                                },
                                {"name": "GIT_SSL_CAINFO", "value": "/etc/pki/tls/custom-certs/ca-bundle.crt"},
                            ],
                            "image": image.image_url,
                            "imagePullPolicy": "Always",
                            "livenessProbe": probe_config,
                            "name": notebook_name,
                            "ports": [{"containerPort": 8888, "name": "notebook-port", "protocol": "TCP"}],
                            "readinessProbe": probe_config,
                            "resources": {
                                "limits": {"cpu": "2", "memory": "4Gi"},
                                "requests": {"cpu": "1", "memory": "1Gi"},
                            },
                            "volumeMounts": [
                                {"mountPath": "/opt/app-root/src", "name": pvc_name},
                                {"mountPath": "/dev/shm", "name": "shm"},
                                {
                                    "mountPath": "/etc/pki/tls/custom-certs",
                                    "name": "trusted-ca",
                                    "readOnly": True,
                                },
                                {
                                    "mountPath": "/opt/app-root/pipeline-runtimes/",
                                    "name": "runtime-images",
                                },
                            ],
                            "workingDir": "/opt/app-root/src",
                        },
                    ],
                    "enableServiceLinks": False,
                    "serviceAccountName": notebook_name,
                    "volumes": [
                        {"name": pvc_name, "persistentVolumeClaim": {"claimName": pvc_name}},
                        {"emptyDir": {"medium": "Memory"}, "name": "shm"},
                        {
                            "name": "trusted-ca",
                            "configMap": {
                                "name": TRUSTED_CA_BUNDLE_NAME,
                                "optional": True,
                            },
                        },
                        {
                            "name": "runtime-images",
                            "configMap": {
                                "name": PIPELINE_RUNTIME_IMAGES_NAME,
                                "optional": True,
                            },
                        },
                        {
                            "name": "kube-rbac-proxy-config",
                            "configMap": {
                                "defaultMode": 420,
                                "name": f"{notebook_name}-kube-rbac-proxy-config",
                            },
                        },
                        {
                            "name": "kube-rbac-proxy-tls-certificates",
                            "secret": {
                                "defaultMode": 420,
                                "secretName": f"{notebook_name}-kube-rbac-proxy-tls",
                            },
                        },
                    ],
                }
            }
        },
    }


def wait_for_controller_reconciliation(
    admin_client: DynamicClient,
    notebook_name: str,
    notebook_namespace: str,
    notebook_pod: Pod,
    timeout: int = Timeout.TIMEOUT_5MIN,
) -> None:
    """Wait for the notebook controller to finish reconciling auth and routing resources."""
    try:
        notebook_pod.wait()
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=timeout,
        )
    except (TimeoutError, TimeoutExpiredError) as exc:
        if notebook_pod.exists:
            collect_pod_information(pod=notebook_pod)
            raise AssertionError(
                f"Pod '{notebook_name}-0' failed to reach Ready state within {timeout} seconds.\nOriginal error: {exc}"
            ) from exc
        raise AssertionError(f"Pod '{notebook_name}-0' was not created. Check notebook controller logs.") from exc

    def _controller_reconciled() -> bool:
        container_names = {container.name for container in notebook_pod.instance.spec.containers}
        if "kube-rbac-proxy" not in container_names:
            return False

        reference_grant = ReferenceGrant(
            client=admin_client,
            name=REFERENCE_GRANT_NAME,
            namespace=notebook_namespace,
        )
        try:
            if not reference_grant.exists:
                return False
        except ResourceNotFoundError:
            return False

        http_route = HTTPRoute(
            client=admin_client,
            name=f"nb-{notebook_namespace}-{notebook_name}",
            namespace=_applications_namespace(),
        )
        try:
            if not http_route.exists:
                return False
            http_route_instance = http_route.instance
        except ResourceNotFoundError:
            return False

        http_route_status = http_route_instance.to_dict().get("status", {})
        conditions = {
            condition.get("type"): condition.get("status")
            for parent in http_route_status.get("parents", [])
            for condition in parent.get("conditions", [])
        }
        return conditions.get("Accepted") == "True" and conditions.get("ResolvedRefs") == "True"

    try:
        for sample in TimeoutSampler(wait_timeout=timeout, sleep=5, func=_controller_reconciled):
            if sample:
                return
    except TimeoutExpiredError as exc:
        collect_pod_information(pod=notebook_pod)
        raise AssertionError(
            f"Notebook controller did not finish reconciling auth/routing resources for "
            f"{notebook_namespace}/{notebook_name} within {timeout} seconds"
        ) from exc


def grab_and_check_pod_logs(
    pod: Pod,
    container_name: str,
    extra_allowed: tuple[str, ...] | None = None,
) -> str:
    """Fail when the workbench container logs contain unexpected errors or warnings."""
    time.sleep(3)
    full_logs = pod.log(container=container_name)
    allowed_messages = ALLOWED_LOG_MESSAGES + tuple(extra_allowed or ())
    failed_lines: list[str] = []

    for line in full_logs.splitlines():
        if any(keyword in line for keyword in BLOCKED_LOG_KEYWORDS):
            if any(allowed_message in line for allowed_message in allowed_messages):
                LOGGER.debug(f"Waived workbench log line: {_redact_log_line(line=line)}")
                continue
            failed_lines.append(_redact_log_line(line=line))

    if failed_lines:
        collect_pod_information(pod=pod)
        joined_lines = "\n".join(failed_lines)
        raise AssertionError(
            "Unexpected log message(s) were emitted by the workbench container during startup or probing:\n"
            + joined_lines
        )

    return full_logs


def wait_for_http_inside_pod(
    pod: Pod,
    container_name: str,
    namespace: str,
    notebook_name: str,
    timeout: int = Timeout.TIMEOUT_2MIN,
) -> None:
    """Wait until the in-pod workbench HTTP endpoint responds successfully."""
    probe_url = f"http://localhost:{NOTEBOOK_PORT}/notebook/{namespace}/{notebook_name}/api"
    check_script = "import sys; import urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)"
    probe_commands = (
        ["python", "-c", check_script, probe_url],
        ["python3", "-c", check_script, probe_url],
    )

    def _probe_http() -> bool:
        for command in probe_commands:
            try:
                pod.execute(container=container_name, command=command, timeout=10)
                return True
            except ExecOnPodError:
                continue
        return False

    try:
        for probe_succeeded in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=_probe_http,
        ):
            if probe_succeeded:
                return
    except TimeoutExpiredError as exc:
        collect_pod_information(pod=pod)
        raise AssertionError(
            f"Timed out waiting for in-pod HTTP access to '{probe_url}' from container '{container_name}'"
        ) from exc


def get_container_restart_counts(pod: Pod) -> dict[str, int]:
    """Return restart counts for all containers in the pod."""
    return {
        container_status.name: int(container_status.restartCount)
        for container_status in (pod.instance.status.containerStatuses or [])
    }


def get_container_image_digest(pod: Pod, container_name: str) -> str:
    """Return the resolved digest for the requested pod container."""
    for container_status in pod.instance.status.containerStatuses or []:
        if container_status.name != container_name:
            continue

        image_id = str(getattr(container_status, "imageID", "") or "")
        if "@sha256:" in image_id:
            return image_id.split("@", maxsplit=1)[1]
        if "sha256:" in image_id:
            return f"sha256:{image_id.split('sha256:', maxsplit=1)[1]}"

        image = str(getattr(container_status, "image", "") or "")
        if "@sha256:" in image:
            return image.split("@", maxsplit=1)[1]

        raise AssertionError(
            f"Container '{container_name}' in pod '{pod.name}' does not expose a digest-backed image reference"
        )

    available_containers = [container_status.name for container_status in (pod.instance.status.containerStatuses or [])]
    raise AssertionError(
        f"Container '{container_name}' was not found in pod '{pod.name}'. Available containers: {available_containers}"
    )


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
    marker_path = f"/opt/app-root/src/{UPGRADE_MARKER_FILENAME}"
    try:
        output = pod.execute(
            container=container_name,
            command=["cat", marker_path],
            timeout=60,
        )
    except ExecOnPodError as error:
        raise AssertionError(
            f"Failed to read upgrade marker file '{marker_path}' from pod '{pod.name}' "
            f"container '{container_name}'. "
            "The pre-upgrade write may have failed silently, or the path is wrong. "
            f"Underlying exec error: {error}"
        ) from error
    return output.strip()


def capture_workbench_baseline(
    notebook: Notebook,
    pod: Pod,
    resolved_image: ResolvedWorkbenchImage,
    *,
    upgrade_marker: str = UPGRADE_MARKER_CONTENT,
) -> WorkbenchImageBaseline:
    """Capture the pre-upgrade baseline that post-upgrade tests compare against."""
    notebook_annotations = notebook.instance.metadata.annotations or {}
    container_name = notebook.name

    return WorkbenchImageBaseline(
        creation_timestamp=pod.instance.metadata.creationTimestamp,
        image_tag=resolved_image.tag_name,
        image_url=resolved_image.image_url,
        image_digest=resolved_image.image_digest,
        pod_image_digest=get_container_image_digest(pod=pod, container_name=container_name),
        last_image_selection=notebook_annotations["notebooks.opendatahub.io/last-image-selection"],
        pod_name=pod.name,
        restart_counts=get_container_restart_counts(pod=pod),
        notebook_generation=int(notebook.instance.metadata.generation),
        upgrade_marker=upgrade_marker,
    )


def verify_notebook_pod_not_recreated(pod: Pod, baseline: WorkbenchImageBaseline) -> None:
    """Verify that the workbench pod is the original pre-upgrade pod."""
    actual_creation_timestamp = pod.instance.metadata.creationTimestamp
    assert actual_creation_timestamp == baseline.creation_timestamp, (
        f"Workbench pod {pod.name} was recreated during the upgrade. "
        f"Expected creationTimestamp {baseline.creation_timestamp}, got {actual_creation_timestamp}"
    )


def verify_notebook_image_selection_unchanged(notebook: Notebook, baseline: WorkbenchImageBaseline) -> None:
    """Verify that the Notebook CR still points to the pre-upgrade image selection."""
    annotations = notebook.instance.metadata.annotations or {}
    actual_image_selection = annotations.get("notebooks.opendatahub.io/last-image-selection")
    assert actual_image_selection == baseline.last_image_selection, (
        "Workbench last-image-selection annotation changed during the upgrade. "
        f"Expected {baseline.last_image_selection}, got {actual_image_selection}"
    )


def verify_notebook_image_digest_unchanged(
    pod: Pod,
    container_name: str,
    baseline: WorkbenchImageBaseline,
) -> None:
    """Verify that the running workbench container still uses the original image digest."""
    actual_digest = get_container_image_digest(pod=pod, container_name=container_name)
    assert actual_digest == baseline.pod_image_digest, (
        f"Workbench image digest changed during the upgrade. Expected {baseline.pod_image_digest}, got {actual_digest}"
    )


def verify_notebook_restart_counts_unchanged(pod: Pod, baseline: WorkbenchImageBaseline) -> None:
    """Verify that no workbench pod container restarted across the upgrade."""
    actual_restart_counts = get_container_restart_counts(pod=pod)
    assert actual_restart_counts == baseline.restart_counts, (
        "Workbench container restart counts changed during the upgrade. "
        f"Expected {baseline.restart_counts}, got {actual_restart_counts}"
    )


def verify_notebook_generation_unchanged(notebook: Notebook, baseline: WorkbenchImageBaseline) -> None:
    """Verify that the Notebook CR was not modified during the upgrade."""
    actual_generation = int(notebook.instance.metadata.generation)
    assert actual_generation == baseline.notebook_generation, (
        f"Notebook CR was modified during the upgrade. "
        f"Pre-upgrade generation: {baseline.notebook_generation}, "
        f"post-upgrade generation: {actual_generation}"
    )


def verify_statefulset_healthy(statefulset: StatefulSet) -> None:
    """Verify that the workbench StatefulSet is healthy after the upgrade."""
    assert statefulset.exists, f"StatefulSet '{statefulset.name}' no longer exists after upgrade"

    sts = statefulset.instance
    expected_replicas = sts.spec.replicas
    ready_replicas = sts.status.readyReplicas or 0
    assert ready_replicas == expected_replicas, (
        f"StatefulSet '{statefulset.name}' has {ready_replicas} ready replicas, expected {expected_replicas}"
    )

    current_revision = sts.status.currentRevision
    update_revision = sts.status.updateRevision
    assert current_revision == update_revision, (
        f"StatefulSet '{statefulset.name}' has a pending rollout: "
        f"currentRevision='{current_revision}', updateRevision='{update_revision}'"
    )


def store_workbench_baseline(
    config_map_data: dict[str, str],
    baseline_prefix: str,
    baseline: WorkbenchImageBaseline,
) -> dict[str, str]:
    """Merge one workbench baseline into ConfigMap string data."""
    updated_data = dict(config_map_data)
    updated_data.update(baseline.to_configmap_data(prefix=baseline_prefix))
    return updated_data


def load_workbench_baseline(config_map_data: dict[str, str], baseline_prefix: str) -> WorkbenchImageBaseline:
    """Load one workbench baseline from ConfigMap string data."""
    return WorkbenchImageBaseline.from_configmap_data(prefix=baseline_prefix, data=config_map_data)


def get_workbench_image_spec_by_ide(ide: str) -> WorkbenchImageSpec:
    """Return the workbench IDE configuration for the requested IDE name."""
    for spec in get_workbench_image_specs():
        if spec.ide == ide:
            return spec
    raise KeyError(f"Unknown workbench IDE '{ide}'")


def wait_for_notebook_deletion(
    unprivileged_client: DynamicClient,
    *,
    notebook_name: str,
    namespace: str,
) -> None:
    """Wait until a Notebook CR is fully removed from the cluster."""
    notebook_kwargs = {
        "client": unprivileged_client,
        "name": notebook_name,
        "namespace": namespace,
    }
    try:
        for notebook_deleted in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_5MIN,
            sleep=5,
            func=lambda: not Notebook(**notebook_kwargs).exists,
        ):
            if notebook_deleted:
                return
    except TimeoutExpiredError as error:
        raise AssertionError(
            f"Notebook '{notebook_name}' was not deleted within {Timeout.TIMEOUT_5MIN} seconds"
        ) from error


def manage_upgrade_persistent_volume_claim(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    namespace_name: str,
    spec: WorkbenchImageSpec,
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """Create or reuse the PVC backing an upgrade workbench."""
    pvc_kwargs = {
        "client": unprivileged_client,
        "name": spec.pvc_name,
        "namespace": namespace_name,
    }

    if pytestconfig.option.post_upgrade:
        yield PersistentVolumeClaim(**pvc_kwargs)
        return

    existing_pvc = PersistentVolumeClaim(**pvc_kwargs)
    if existing_pvc.exists:
        LOGGER.info(f"PVC '{spec.pvc_name}' already exists, reusing it")
        yield existing_pvc
        return

    with PersistentVolumeClaim(
        **pvc_kwargs,
        label={Labels.OpenDataHub.DASHBOARD: "true"},
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        size="1Gi",
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
        teardown=teardown_resources,
    ) as pvc:
        yield pvc


def manage_upgrade_notebook(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    namespace_name: str,
    spec: WorkbenchImageSpec,
    resolved_image: ResolvedWorkbenchImage,
    teardown_resources: bool,
) -> Generator[Notebook, Any, Any]:
    """Create or reuse the Notebook CR for an upgrade workbench."""
    from tests.workbenches.notebooks_server.controller.utils import notebook_service_account

    notebook_kwargs = {
        "client": unprivileged_client,
        "name": spec.notebook_name,
        "namespace": namespace_name,
    }

    if pytestconfig.option.post_upgrade:
        yield Notebook(**notebook_kwargs)
        return

    existing_notebook = Notebook(**notebook_kwargs)
    if existing_notebook.exists:
        annotations = existing_notebook.instance.metadata.annotations or {}
        selected_image = annotations.get("notebooks.opendatahub.io/last-image-selection")
        if selected_image == resolved_image.image_selection:
            LOGGER.info(f"Notebook '{spec.notebook_name}' already exists, reusing it")
            with notebook_service_account(
                client=unprivileged_client,
                name=spec.notebook_name,
                namespace=namespace_name,
                teardown=False,
            ):
                yield existing_notebook
            return

        LOGGER.warning(
            f"Notebook '{spec.notebook_name}' exists with image '{selected_image}' "
            f"but expected '{resolved_image.image_selection}'; recreating notebook"
        )
        existing_notebook.delete()
        wait_for_notebook_deletion(
            unprivileged_client=unprivileged_client,
            notebook_name=spec.notebook_name,
            namespace=namespace_name,
        )

    notebook_dict = build_n1_notebook_dict(
        namespace=namespace_name,
        notebook_name=spec.notebook_name,
        pvc_name=spec.pvc_name,
        image=resolved_image,
    )
    with (
        notebook_service_account(
            client=unprivileged_client,
            name=spec.notebook_name,
            namespace=namespace_name,
            teardown=teardown_resources,
        ),
        Notebook(client=unprivileged_client, kind_dict=notebook_dict, teardown=teardown_resources) as notebook,
    ):
        yield notebook


def get_ready_upgrade_notebook_pod(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    spec: WorkbenchImageSpec,
    notebook: Notebook,
) -> Pod:
    """Return the Ready notebook pod after controller reconciliation finishes."""
    notebook_pod = Pod(
        client=unprivileged_client,
        namespace=notebook.namespace,
        name=f"{notebook.name}-0",
    )
    wait_for_controller_reconciliation(
        admin_client=admin_client,
        notebook_name=spec.notebook_name,
        notebook_namespace=notebook.namespace,
        notebook_pod=notebook_pod,
        timeout=Timeout.TIMEOUT_10MIN,
    )
    return notebook_pod


def capture_or_load_workbench_baseline(
    pytestconfig: pytest.Config,
    config_map: ConfigMap,
    spec: WorkbenchImageSpec,
    notebook: Notebook,
    pod: Pod,
    resolved_image: ResolvedWorkbenchImage,
) -> WorkbenchImageBaseline:
    """Capture pre-upgrade baseline data or load the persisted post-upgrade baseline."""
    if pytestconfig.option.post_upgrade:
        return load_workbench_baseline(
            config_map_data=dict(config_map.instance.data or {}),
            baseline_prefix=spec.baseline_prefix,
        )

    write_pvc_upgrade_marker(pod=pod, container_name=spec.notebook_name)
    baseline = capture_workbench_baseline(
        notebook=notebook,
        pod=pod,
        resolved_image=resolved_image,
    )
    updated_data = store_workbench_baseline(
        config_map_data=dict(config_map.instance.data or {}),
        baseline_prefix=spec.baseline_prefix,
        baseline=baseline,
    )
    ResourceEditor(patches={config_map: {"data": updated_data}}).update()
    LOGGER.info(f"Saved N-1 baseline for {spec.ide}: tag={baseline.image_tag}")
    return baseline
