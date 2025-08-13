import hashlib
import requests
import json
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.pod import Pod

from utilities.general import b64_encoded_string
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


def check_job_completion(admin_client: DynamicClient, job: Job) -> bool:
    """
    Check if job has completed successfully.

    Args:
        admin_client: Kubernetes dynamic client
        job: Job object to check

    Returns:
        bool: True if job completed successfully, False otherwise
    """
    job_status = job.instance.status

    # Check if job succeeded
    if hasattr(job_status, "succeeded") and job_status.succeeded:
        LOGGER.info("Job completed successfully")
        return True

    # Log current pod status for debugging
    try:
        current_pod = get_latest_job_pod(admin_client=admin_client, job=job)
        pod_status = current_pod.instance.status
        if hasattr(pod_status, "phase"):
            if pod_status.phase == "Succeeded":
                LOGGER.info(f"Pod {current_pod.name} succeeded")
            elif pod_status.phase == "Failed":
                LOGGER.warning(f"Pod {current_pod.name} failed, job may retry with new pod")
            elif pod_status.phase == "Running":
                LOGGER.info(f"Pod {current_pod.name} is running...")
            else:
                LOGGER.info(f"Pod {current_pod.name} in phase: {pod_status.phase}")
    except Exception as e:
        LOGGER.warning(f"Could not get pod status: {e}")

    return False


def get_latest_job_pod(admin_client: DynamicClient, job: Job) -> Pod:
    """Get the latest (most recently created) Pod created by a Job"""
    pods = list(
        Pod.get(
            dyn_client=admin_client,
            namespace=job.namespace,
            label_selector=f"job-name={job.name}",
        )
    )

    if not pods:
        raise AssertionError(f"No pods found for job {job.name}")

    # Sort pods by creation time (latest first)
    sorted_pods = sorted(pods, key=lambda p: p.instance.metadata.creationTimestamp or "", reverse=True)

    latest_pod = sorted_pods[0]
    LOGGER.info(f"Found {len(pods)} pod(s) for job {job.name}, using latest: {latest_pod.name}")
    return latest_pod


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


def get_async_job_s3_secret_dict(
    access_key: str,
    secret_access_key: str,
    s3_bucket: str,
    s3_endpoint: str,
    s3_region: str,
) -> dict[str, str]:
    """
    Returns a dictionary of s3 secret values

    Args:
        access_key (str): access key
        secret_access_key (str): secret key
        s3_bucket (str): S3 bucket
        s3_endpoint (str): S3 endpoint
        s3_region (str): S3 region

    Returns:
        dict[str, str]: A dictionary of s3 secret encoded values

    """
    return {
        "AWS_ACCESS_KEY_ID": b64_encoded_string(string_to_encode=access_key),
        "AWS_SECRET_ACCESS_KEY": b64_encoded_string(string_to_encode=secret_access_key),
        "AWS_BUCKET": b64_encoded_string(string_to_encode=s3_bucket),
        "AWS_ENDPOINT_URL": b64_encoded_string(string_to_encode=s3_endpoint),
        "AWS_REGION": b64_encoded_string(string_to_encode=s3_region),
    }


def upload_test_model_to_minio(
    admin_client: DynamicClient,
    namespace: str,
    minio_service,
    object_key: str = "my-model/mnist-8.onnx",
) -> None:
    """
    Upload mnist-8.onnx test model to MinIO using a temporary pod

    Args:
        admin_client: Kubernetes client
        namespace: Namespace to create upload pod in
        minio_service: MinIO service resource
        object_key: S3 object key path (default: "my-model/mnist-8.onnx")
    """
    from ocp_resources.pod import Pod
    from utilities.constants import MinIo
    import base64
    import os

    # Read the mnist-8.onnx file from the repository
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(current_dir, "..", "..", "..")
    model_file_path = os.path.join(repo_root, "tests", "model_registry", "async_job", "artifacts", "mnist-8.onnx")

    try:
        with open(model_file_path, "rb") as f:
            model_file_content = f.read()
        LOGGER.info(f"Loaded mnist-8.onnx file ({len(model_file_content)} bytes)")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"mnist-8.onnx not found at {model_file_path}. Please ensure the file exists in the repository root."
        )

    # Encode the model file content as base64 for embedding in the pod
    encoded_content = base64.b64encode(model_file_content).decode("ascii")

    with Pod(
        client=admin_client,
        name="test-model-uploader",
        namespace=namespace,
        restart_policy="Never",
        volumes=[{"name": "upload-data", "emptyDir": {}}],
        init_containers=[
            {
                "name": "decode-model-file",
                "image": "registry.redhat.io/ubi8/ubi-minimal:latest",
                "command": ["/bin/sh", "-c"],
                "args": [f"echo '{encoded_content}' | base64 -d > /upload-data/model-file"],
                "volumeMounts": [{"name": "upload-data", "mountPath": "/upload-data"}],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
            }
        ],
        containers=[
            {
                "name": "minio-uploader",
                "image": "quay.io/minio/mc@sha256:470f5546b596e16c7816b9c3fa7a78ce4076bb73c2c73f7faeec0c8043923123",
                "command": ["/bin/sh", "-c"],
                "args": [
                    # Set up MinIO client and upload the model file
                    f"export MC_CONFIG_DIR=/upload-data/.mc && "
                    f"mc alias set testminio http://{minio_service.name}.{minio_service.namespace}.svc.cluster.local:{MinIo.Metadata.DEFAULT_PORT} "  # noqa: E501
                    f"{MinIo.Credentials.ACCESS_KEY_VALUE} {MinIo.Credentials.SECRET_KEY_VALUE} && "
                    f"mc mb --ignore-existing testminio/{MinIo.Buckets.MODELMESH_EXAMPLE_MODELS} && "
                    f"mc cp /upload-data/model-file testminio/{MinIo.Buckets.MODELMESH_EXAMPLE_MODELS}/{object_key} && "
                    f"mc ls testminio/{MinIo.Buckets.MODELMESH_EXAMPLE_MODELS}/my-model/ && "
                    f"echo 'Upload completed successfully'"
                ],
                "volumeMounts": [{"name": "upload-data", "mountPath": "/upload-data"}],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
            }
        ],
        wait_for_resource=True,
    ) as upload_pod:
        LOGGER.info(f"Uploading model file to MinIO: {object_key}")
        upload_pod.wait_for_status(status="Succeeded", timeout=300)

        # Get upload logs for verification
        try:
            upload_logs = upload_pod.log()
            LOGGER.info(f"Upload logs: {upload_logs}")
        except Exception as e:
            LOGGER.warning(f"Could not retrieve upload logs: {e}")

        LOGGER.info(f"✓ Model file uploaded successfully to s3://{MinIo.Buckets.MODELMESH_EXAMPLE_MODELS}/{object_key}")
