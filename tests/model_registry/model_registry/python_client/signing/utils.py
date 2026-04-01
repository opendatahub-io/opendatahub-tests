"""Utility functions for Model Registry Python Client Signing Tests."""

import hashlib
import os
from collections.abc import Generator
from typing import Any

import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from pyhelper_utils.shell import run_command
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_registry.model_registry.async_job.constants import (
    ASYNC_JOB_ANNOTATIONS,
    ASYNC_JOB_LABELS,
    MODEL_SYNC_CONFIG,
    VOLUME_MOUNTS,
)
from tests.model_registry.model_registry.python_client.signing.constants import (
    SECURESIGN_NAMESPACE,
    SECURESIGN_ORGANIZATION_EMAIL,
    SECURESIGN_ORGANIZATION_NAME,
)
from tests.model_registry.utils import get_endpoint_from_mr_service, get_mr_service_by_label
from utilities.constants import MinIo, OCIRegistry, Protocols
from utilities.general import collect_pod_information
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry

LOGGER = structlog.get_logger(name=__name__)


def get_organization_config() -> dict[str, str]:
    """Get organization configuration for certificates."""
    return {
        "organizationName": SECURESIGN_ORGANIZATION_NAME,
        "organizationEmail": SECURESIGN_ORGANIZATION_EMAIL,
    }


def get_tas_service_urls(securesign_instance: dict) -> dict[str, str]:
    """Extract TAS service URLs from Securesign instance status.

    Args:
        securesign_instance: Securesign instance dictionary from Kubernetes API

    Returns:
        dict: Service URLs with keys 'fulcio', 'rekor', 'tsa', 'tuf'

    Raises:
        KeyError: If expected status fields are missing from Securesign instance
    """
    status = securesign_instance["status"]

    return {
        "fulcio": status["fulcio"]["url"],
        "rekor": status["rekor"]["url"],
        "tsa": status["tsa"]["url"],
        "tuf": status["tuf"]["url"],
    }


def create_connection_type_field(
    name: str, description: str, env_var: str, default_value: str, required: bool = True
) -> dict:
    """Create a Connection Type field dictionary for ODH dashboard.

    Args:
        name: Display name of the field shown in UI
        description: Help text describing the field's purpose
        env_var: Environment variable name for programmatic access
        default_value: Default value to populate (typically a service URL)
        required: Whether the field must be filled

    Returns:
        dict: Field dictionary conforming to ODH Connection Type schema
    """
    return {
        "type": "short-text",
        "name": name,
        "description": description,
        "envVar": env_var,
        "properties": {"defaultValue": default_value},
        "required": required,
    }


def generate_token(temp_base_folder) -> str:
    """
    Create a service account token and save it to a temporary directory.
    """
    filepath = os.path.join(temp_base_folder, "token")

    LOGGER.info(f"Creating service account token for namespace {SECURESIGN_NAMESPACE}...")
    _, out, _ = run_command(
        command=["oc", "create", "token", "default", "-n", SECURESIGN_NAMESPACE, "--duration=1h"], check=True
    )

    token = out.strip()
    with open(filepath, "w") as fd:
        fd.write(token)
    return filepath


def get_root_checksum(sigstore_tuf_url: str) -> str:
    """
    Download root.json from TUF URL and calculate SHA256 checksum.
    """
    if not sigstore_tuf_url:
        raise ValueError("sigstore_tuf_url cannot be empty or None")

    try:
        LOGGER.info(f"Downloading root.json from: {sigstore_tuf_url}/root.json")
        response = requests.get(f"{sigstore_tuf_url}/root.json", timeout=30)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Calculate SHA256 checksum
        checksum = hashlib.sha256(response.content).hexdigest()
        LOGGER.info(f"Calculated root.json checksum: {checksum}")

    except requests.RequestException as e:
        LOGGER.error(f"Failed to download root.json from {sigstore_tuf_url}: {e}")
        raise
    except (ValueError, OSError) as e:
        LOGGER.error(f"Failed to calculate checksum: {e}")
        raise RuntimeError(f"Checksum calculation failed: {e}")
    else:
        return checksum


def check_model_signature_file(model_dir: str) -> bool:
    """
    Check for the presence of model.sig file in the model directory.

    Args:
        model_dir: Path to the model directory

    Returns:
        bool: True if model.sig file exists, False otherwise
    """
    sig_file_path = os.path.join(model_dir, "model.sig")
    LOGGER.info(f"Checking for signature file: {sig_file_path}")

    if os.path.exists(sig_file_path):
        LOGGER.info(f"Signature file found: {sig_file_path}")
        return True
    else:
        LOGGER.info(f"Signature file not found: {sig_file_path}")
        return False


def run_minio_uploader_pod(
    admin_client: DynamicClient,
    namespace: str,
    minio_service: Service,
    pod_name: str,
    mc_commands: str,
    volumes: list[dict[str, Any]] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
) -> None:
    """Run a MinIO mc uploader pod with the given shell commands.

    Creates a pod that sets up an mc alias to MinIO and runs the provided commands.

    Args:
        admin_client: Kubernetes dynamic client
        namespace: Namespace to create the pod in
        minio_service: MinIO service for endpoint resolution
        pod_name: Name for the uploader pod
        mc_commands: Shell commands to run after mc alias setup (e.g. mc cp ...)
        volumes: Additional volumes to mount
        volume_mounts: Additional volume mounts for the container
    """
    from tests.model_registry.model_registry.python_client.signing.constants import (
        MINIO_MC_IMAGE,
        MINIO_UPLOADER_SECURITY_CONTEXT,
    )

    mc_url = f"http://{minio_service.name}.{minio_service.namespace}.svc.cluster.local:{MinIo.Metadata.DEFAULT_PORT}"

    all_volumes = [{"name": "work", "emptyDir": {}}]
    if volumes:
        all_volumes.extend(volumes)

    all_volume_mounts = [{"name": "work", "mountPath": "/work"}]
    if volume_mounts:
        all_volume_mounts.extend(volume_mounts)

    mc_setup = (
        f"export MC_CONFIG_DIR=/work/.mc && "
        f"mc alias set testminio {mc_url} "
        f"{MinIo.Credentials.ACCESS_KEY_VALUE} {MinIo.Credentials.SECRET_KEY_VALUE} && "
        f"mc mb --ignore-existing testminio/{MinIo.Buckets.MODELMESH_EXAMPLE_MODELS}"
    )

    with Pod(
        client=admin_client,
        name=pod_name,
        namespace=namespace,
        restart_policy="Never",
        volumes=all_volumes,
        containers=[
            {
                "name": "minio-uploader",
                "image": MINIO_MC_IMAGE,
                "command": ["/bin/sh", "-c"],
                "args": [f"{mc_setup} && {mc_commands}"],
                "volumeMounts": all_volume_mounts,
                "securityContext": MINIO_UPLOADER_SECURITY_CONTEXT,
            }
        ],
        wait_for_resource=True,
    ) as upload_pod:
        LOGGER.info(f"Running minio uploader pod: {pod_name}")
        try:
            upload_pod.wait_for_status(status="Succeeded", timeout=300)
        except TimeoutExpiredError:
            collect_pod_information(pod=upload_pod)
            raise
        LOGGER.info(f"Minio uploader pod '{pod_name}' completed successfully")


def get_base_async_job_env_vars(
    mr_host: str,
    sa_token: str,
    oci_internal: str,
    oci_repo: str,
) -> list[dict[str, str]]:
    """Build the common environment variables for async upload jobs."""
    return [
        {"name": "MODEL_SYNC_SOURCE_TYPE", "value": MODEL_SYNC_CONFIG["SOURCE_TYPE"]},
        {"name": "MODEL_SYNC_SOURCE_AWS_KEY", "value": MODEL_SYNC_CONFIG["SOURCE_AWS_KEY"]},
        {"name": "MODEL_SYNC_SOURCE_S3_CREDENTIALS_PATH", "value": VOLUME_MOUNTS["SOURCE_CREDS_PATH"]},
        {"name": "MODEL_SYNC_MODEL_ID", "value": MODEL_SYNC_CONFIG["MODEL_ID"]},
        {"name": "MODEL_SYNC_MODEL_VERSION_ID", "value": MODEL_SYNC_CONFIG["MODEL_VERSION_ID"]},
        {"name": "MODEL_SYNC_MODEL_ARTIFACT_ID", "value": MODEL_SYNC_CONFIG["MODEL_ARTIFACT_ID"]},
        {"name": "MODEL_SYNC_REGISTRY_SERVER_ADDRESS", "value": f"https://{mr_host}"},
        {"name": "MODEL_SYNC_REGISTRY_USER_TOKEN", "value": sa_token},
        {"name": "MODEL_SYNC_REGISTRY_IS_SECURE", "value": "False"},
        {"name": "MODEL_SYNC_DESTINATION_OCI_REGISTRY", "value": oci_internal},
        {"name": "MODEL_SYNC_DESTINATION_OCI_URI", "value": f"{oci_internal}/{oci_repo}"},
        {"name": "MODEL_SYNC_DESTINATION_OCI_BASE_IMAGE", "value": MODEL_SYNC_CONFIG["DESTINATION_OCI_BASE_IMAGE"]},
        {
            "name": "MODEL_SYNC_DESTINATION_OCI_ENABLE_TLS_VERIFY",
            "value": MODEL_SYNC_CONFIG["DESTINATION_OCI_ENABLE_TLS_VERIFY"],
        },
    ]


def get_oci_internal_endpoint(oci_registry_service: Service) -> str:
    """Build the internal OCI registry endpoint from a Service."""
    oci_internal_host = f"{oci_registry_service.name}.{oci_registry_service.namespace}.svc.cluster.local"
    return f"{oci_internal_host}:{OCIRegistry.Metadata.DEFAULT_PORT}"


def get_model_registry_host(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_instance: list[ModelRegistry],
) -> str:
    """Resolve the Model Registry REST host from the first instance."""
    mr_instance = model_registry_instance[0]
    mr_service = get_mr_service_by_label(
        client=admin_client, namespace_name=model_registry_namespace, mr_instance=mr_instance
    )
    mr_endpoint = get_endpoint_from_mr_service(svc=mr_service, protocol=Protocols.REST)
    return mr_endpoint.split(":")[0]


def create_async_upload_job(
    admin_client: DynamicClient,
    job_name: str,
    namespace: str,
    async_upload_image: str,
    s3_secret: Secret,
    oci_secret: Secret,
    environment_variables: list[dict[str, str]],
    extra_volume_mounts: list[dict[str, Any]] | None = None,
    extra_volumes: list[dict[str, Any]] | None = None,
    teardown: bool = True,
) -> Generator[Job, Any, Any]:
    """Create and run an async upload Job with the given configuration."""
    volume_mounts = [
        {"name": "source-credentials", "readOnly": True, "mountPath": VOLUME_MOUNTS["SOURCE_CREDS_PATH"]},
        {"name": "destination-credentials", "readOnly": True, "mountPath": VOLUME_MOUNTS["DEST_CREDS_PATH"]},
    ]
    volumes = [
        {"name": "source-credentials", "secret": {"secretName": s3_secret.name}},
        {"name": "destination-credentials", "secret": {"secretName": oci_secret.name}},
    ]

    if extra_volume_mounts:
        volume_mounts.extend(extra_volume_mounts)
    if extra_volumes:
        volumes.extend(extra_volumes)

    with Job(
        client=admin_client,
        name=job_name,
        namespace=namespace,
        label=ASYNC_JOB_LABELS,
        annotations=ASYNC_JOB_ANNOTATIONS,
        restart_policy="Never",
        containers=[
            {
                "name": "async-upload",
                "image": async_upload_image,
                "volumeMounts": volume_mounts,
                "env": environment_variables,
            }
        ],
        volumes=volumes,
        teardown=teardown,
    ) as job:
        job.wait_for_condition(condition="Complete", status="True")
        LOGGER.info(f"Async upload job '{job_name}' completed successfully")
        yield job


def get_oci_image_with_digest(oci_host: str, repo: str, tag: str) -> Generator[str, Any, Any]:
    """Get OCI image reference with digest and set COSIGN_ALLOW_INSECURE_REGISTRY."""
    registry_url = f"https://{oci_host}"

    LOGGER.info(f"Waiting for OCI registry to be reachable at {registry_url}/v2/")
    for sample in TimeoutSampler(
        wait_timeout=120,
        sleep=5,
        func=requests.get,
        url=f"{registry_url}/v2/",
        timeout=5,
        verify=False,
    ):
        if sample.ok:
            break

    tags_url = f"{registry_url}/v2/{repo}/tags/list"
    response = requests.get(url=tags_url, verify=False, timeout=10)
    response.raise_for_status()
    tags_data = response.json()
    LOGGER.info(f"OCI registry tags: {tags_data}")
    assert tag in tags_data.get("tags", []), f"Expected tag '{tag}' not found in registry: {tags_data}"

    manifest_url = f"{registry_url}/v2/{repo}/manifests/{tag}"
    manifest_response = requests.get(
        url=manifest_url,
        headers={"Accept": "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json"},
        verify=False,
        timeout=10,
    )
    manifest_response.raise_for_status()
    digest = manifest_response.headers.get("Docker-Content-Digest")
    assert digest, "Could not get digest from manifest response"

    image_ref = f"{oci_host}/{repo}@{digest}"
    LOGGER.info(f"OCI image reference with digest: {image_ref}")

    os.environ["COSIGN_ALLOW_INSECURE_REGISTRY"] = "true"
    yield image_ref
    os.environ.pop("COSIGN_ALLOW_INSECURE_REGISTRY", None)
