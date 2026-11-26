"""Helper functions for multi-model serving test fixtures.

Extracted from conftest.py so the component conftest contains only fixtures.
"""

from __future__ import annotations

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.serving_runtime import ServingRuntime

from utilities.general import download_model_data

LOGGER = structlog.get_logger(name=__name__)

# Static-profile environment defaults recorded per RHAI-49122
STATIC_PROFILE_ENV_VARS: list[str] = [
    "CHECK_REPOSITORY_POPULATED",
    "MLSERVER_STRICT_READINESS",
    "MLSERVER_EMPTY_REGISTRY_READINESS",
    "MLSERVER_LOAD_MODELS_AT_STARTUP",
    "MLSERVER_MODELS_DIR",
    "MLSERVER_PARALLEL_WORKERS",
]


def append_env_to_serving_runtime(
    runtime: ServingRuntime,
    env_vars: list[dict[str, str]],
) -> None:
    """Append environment variables to the first container of a live ServingRuntime.

    Reads the current env list, appends any vars not already present, then
    sends a merge-patch with the full container array.  This avoids
    overwriting template-provided env vars (which the ``containers`` kwarg
    of ``ServingRuntimeFromTemplate`` would do).

    Args:
        runtime: An already-created ServingRuntime resource.
        env_vars: List of ``{"name": ..., "value": ...}`` dicts to append.
    """
    runtime.reload()
    containers = runtime.instance.spec.get("containers", [])
    if not containers:
        LOGGER.warning(
            "ServingRuntime has no containers; env patch skipped",
            runtime=runtime.name,
            namespace=runtime.namespace,
        )
        return

    existing_env = list(containers[0].get("env", []))
    existing_names = {e.get("name") for e in existing_env}
    for var in env_vars:
        if var["name"] not in existing_names:
            existing_env.append(var)

    containers[0]["env"] = existing_env
    runtime.update(resource_dict={"spec": {"containers": [dict(c) for c in containers]}})


def record_pod_image(pod: Pod) -> dict[str, str]:
    """Extract container image reference (repository, tag, digest) from a pod.

    Satisfies RHAI-49122 image traceability requirement.

    Args:
        pod: Running Kubernetes pod.

    Returns:
        Dictionary with repository, tag, digest, and full image string.
    """
    containers = pod.instance.spec.get("containers", [])
    if not containers:
        return {"image": "", "repository": "", "tag": "", "digest": ""}

    image: str = containers[0].get("image", "")
    repository = image
    tag = ""
    digest = ""

    if "@" in image:
        repository, digest = image.rsplit("@", 1)
    if ":" in repository:
        repository, tag = repository.rsplit(":", 1)

    # Also capture imageID from status for the digest that was actually pulled
    container_statuses = pod.instance.status.get("containerStatuses", [])
    if container_statuses:
        image_id = container_statuses[0].get("imageID", "")
        if "@" in image_id and not digest:
            digest = image_id.rsplit("@", 1)[1]

    return {"image": image, "repository": repository, "tag": tag, "digest": digest}


def record_static_profile_env(pod: Pod) -> dict[str, str | None]:
    """Record static-profile environment variable defaults from a deployed pod.

    Args:
        pod: Running Kubernetes pod.

    Returns:
        Dictionary mapping env var names to their values (None if unset).
    """
    env_values: dict[str, str | None] = {}
    containers = pod.instance.spec.get("containers", [])
    if not containers:
        return env_values

    env_list = containers[0].get("env", [])
    env_map = {e.get("name"): e.get("value") for e in env_list}
    for var_name in STATIC_PROFILE_ENV_VARS:
        env_values[var_name] = env_map.get(var_name)

    return env_values


def stage_s3_to_pvc(
    admin_client: DynamicClient,
    model_namespace: str,
    pvc_name: str,
    model_path: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    bucket_name: str,
    endpoint: str,
    region: str,
) -> str:
    """Download an S3 prefix into a PVC subdirectory.

    Thin wrapper around ``download_model_data`` so every staging fixture
    uses the same call convention.

    Args:
        admin_client: Kubernetes dynamic client.
        model_namespace: Namespace name.
        pvc_name: Target PVC name.
        model_path: S3 prefix **and** PVC subdirectory (they are the same
            by convention — ``download_model_data`` uses ``model_path``
            for both the source key-prefix and the ``subPath`` mount).
        aws_access_key_id: AWS access key.
        aws_secret_access_key: AWS secret key.
        bucket_name: S3 bucket name.
        endpoint: S3 endpoint URL.
        region: S3 region.

    Returns:
        ``model_path`` unchanged — callers build ``pvc://{pvc}/{path}``.
    """
    return download_model_data(
        client=admin_client,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        model_namespace=model_namespace,
        model_pvc_name=pvc_name,
        bucket_name=bucket_name,
        aws_endpoint_url=endpoint,
        aws_default_region=region,
        model_path=model_path,
        use_sub_path=True,
        restricted_scc_init=True,
    )
