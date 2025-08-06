import hashlib
import requests
import json
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.pod import Pod

from tests.model_registry.async_job.constants import (
    ASYNC_JOB_ANNOTATIONS,
    ASYNC_JOB_LABELS,
    MODEL_SYNC_CONFIG,
    VOLUME_MOUNTS,
)
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


def get_job_pod(admin_client: DynamicClient, job: Job) -> Pod:
    """Get the Pod created by a Job"""
    pods = list(
        Pod.get(
            dyn_client=admin_client,
            namespace=job.namespace,
            label_selector=f"job-name={job.name}",
        )
    )
    assert len(pods) == 1, f"Expected 1 pod for job {job.name}, found {len(pods)}"
    return pods[0]


def validate_job_labels_and_annotations(job: Job) -> None:
    """Validate Job has all required labels and annotations"""
    job_labels = job.instance.metadata.labels or {}
    job_annotations = job.instance.metadata.annotations or {}

    # Validate required labels
    for key, expected_value in ASYNC_JOB_LABELS.items():
        actual_value = job_labels.get(key)
        assert actual_value == expected_value, f"Label {key}: expected {expected_value}, got {actual_value}"

    # Validate required annotations
    for key, expected_value in ASYNC_JOB_ANNOTATIONS.items():
        actual_value = job_annotations.get(key)
        assert actual_value == expected_value, f"Annotation {key}: expected {expected_value}, got {actual_value}"


def validate_job_pod_template_labels(job: Job) -> None:
    """Validate Job pod template has required model sync labels"""
    pod_template_labels = job.instance.spec.template.metadata.labels or {}

    expected_model_labels = {
        "modelregistry.opendatahub.io/model-sync-model-id": MODEL_SYNC_CONFIG["MODEL_ID"],
        "modelregistry.opendatahub.io/model-sync-model-version-id": MODEL_SYNC_CONFIG["MODEL_VERSION_ID"],
        "modelregistry.opendatahub.io/model-sync-model-artifact-id": MODEL_SYNC_CONFIG["MODEL_ARTIFACT_ID"],
    }

    for key, expected_value in expected_model_labels.items():
        actual_value = pod_template_labels.get(key)
        assert actual_value == expected_value, (
            f"Pod template label {key}: expected {expected_value}, got {actual_value}"
        )


def validate_job_environment_variables(job: Job) -> None:
    """Validate Job container has all required environment variables"""
    container = job.instance.spec.template.spec.containers[0]
    env_dict = {env.name: env.value for env in container.env}

    required_env_vars = [
        "MODEL_SYNC_SOURCE_TYPE",
        "MODEL_SYNC_DESTINATION_TYPE",
        "MODEL_SYNC_MODEL_ID",
        "MODEL_SYNC_MODEL_VERSION_ID",
        "MODEL_SYNC_MODEL_ARTIFACT_ID",
    ]

    for var in required_env_vars:
        assert var in env_dict, f"Required environment variable {var} not found"
        assert env_dict[var], f"Environment variable {var} is empty"


def validate_job_volume_mounts(job: Job) -> None:
    """Validate Job container has correct volume mounts"""
    container = job.instance.spec.template.spec.containers[0]
    volume_mounts = {mount.name: mount.mountPath for mount in container.volumeMounts}

    expected_mounts = {
        "source-credentials": VOLUME_MOUNTS["SOURCE_CREDS_PATH"],
        "destination-credentials": VOLUME_MOUNTS["DEST_CREDS_PATH"],
    }

    for volume_name, expected_path in expected_mounts.items():
        actual_path = volume_mounts.get(volume_name)
        assert actual_path == expected_path, f"Volume {volume_name}: expected {expected_path}, got {actual_path}"


def push_blob_to_oci_registry(registry_url: str, data: bytes, repo: str = "test/simple-artifact") -> str:
    """
    Push a blob to an OCI registry.
    https://specs.opencontainers.org/distribution-spec/?v=v1.0.0#pushing-blobs
    POST to /v2/<repo>/blobs/uploads/ in order to initiate the upload
    The response will contain a Location header that contains the upload URL
    PUT to the Location URL with the data to be uploaded
    """

    blob_digest = f"sha256:{hashlib.sha256(data).hexdigest()}"

    LOGGER.info(f"Pushing blob with digest: {blob_digest}")

    upload_response = requests.post(f"{registry_url}/v2/{repo}/blobs/uploads/", timeout=10)
    LOGGER.info(f"Blob upload initiation: {upload_response.status_code}")
    assert upload_response.status_code == 202, f"Failed to initiate blob upload: {upload_response.status_code}"

    upload_location = upload_response.headers.get("Location")
    LOGGER.info(f"Upload location: {upload_location}")
    base_url = f"{registry_url}{upload_location}"
    upload_url = f"{base_url}?digest={blob_digest}"
    response = requests.put(url=upload_url, data=data, headers={"Content-Type": "application/octet-stream"}, timeout=10)
    assert response.status_code == 201, f"Failed to upload blob: {response.status_code}"
    return blob_digest


def create_manifest(blob_digest: str, config_json: str, config_digest: str, data: bytes) -> bytes:
    """Create a manifest for an OCI registry."""

    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "size": len(config_json),
            "digest": config_digest,
        },
        "layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar", "size": len(data), "digest": blob_digest}],
    }

    return json.dumps(manifest, separators=(",", ":")).encode("utf-8")


def push_manifest_to_oci_registry(registry_url: str, manifest: bytes, repo: str, tag: str) -> None:
    """Push a manifest to an OCI registry."""
    response = requests.put(
        f"{registry_url}/v2/{repo}/manifests/{tag}",
        data=manifest,
        headers={"Content-Type": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    assert response.status_code == 201, f"Failed to push manifest: {response.status_code}"


def pull_manifest_from_oci_registry(registry_url: str, repo: str, tag: str) -> dict:
    """Pull a manifest from an OCI registry."""
    response = requests.get(
        f"{registry_url}/v2/{repo}/manifests/{tag}",
        headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    LOGGER.info(f"Manifest pull: {response.status_code}")
    assert response.status_code == 200, f"Failed to pull manifest: {response.status_code}"
    return response.json()
