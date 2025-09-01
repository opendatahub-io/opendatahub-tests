import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.pod import Pod
from ocp_resources.service import Service
from utilities.constants import MinIo
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


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


def upload_test_model_to_minio_from_image(
    admin_client: DynamicClient,
    namespace: str,
    minio_service: Service,
    object_key: str = "my-model/model.onnx",
    model_image: str = MinIo.PodConfig.KSERVE_MINIO_IMAGE,
) -> None:
    """
    Extract and upload test model to MinIO from a container image

    Args:
        admin_client: Kubernetes client
        namespace: Namespace to create upload pod in
        minio_service: MinIO service resource
        object_key: S3 object key path
        model_image: Container image containing the model
    """

    with Pod(
        client=admin_client,
        name="test-model-uploader-from-image",
        namespace=namespace,
        restart_policy="Never",
        volumes=[{"name": "upload-data", "emptyDir": {}}],
        init_containers=[
            {
                "name": "extract-model-from-image",
                "image": model_image,
                "command": ["/bin/sh", "-c"],
                "args": [
                    # Create a test model file for upload testing
                    "echo 'Creating test model file for async upload pipeline testing...' && "
                    "echo 'Test model file for validating the async upload pipeline' > /upload-data/model.onnx && "
                    "echo 'Test model file created successfully'"
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
        containers=[
            {
                "name": "minio-uploader",
                "image": "quay.io/minio/mc@sha256:470f5546b596e16c7816b9c3fa7a78ce4076bb73c2c73f7faeec0c8043923123",
                "command": ["/bin/sh", "-c"],
                "args": [
                    # Upload the test model file to MinIO
                    f"echo 'Model file details:' && ls -la /upload-data/model.onnx && "
                    f"echo 'Model file content preview:' && head -c 100 /upload-data/model.onnx && echo && "
                    f"export MC_CONFIG_DIR=/upload-data/.mc && "
                    f"mc alias set testminio http://{minio_service.name}.{minio_service.namespace}.svc.cluster.local:{MinIo.Metadata.DEFAULT_PORT} "  # noqa: E501
                    f"{MinIo.Credentials.ACCESS_KEY_VALUE} {MinIo.Credentials.SECRET_KEY_VALUE} && "
                    f"mc mb --ignore-existing testminio/{MinIo.Buckets.MODELMESH_EXAMPLE_MODELS} && "
                    f"mc cp /upload-data/model.onnx testminio/{MinIo.Buckets.MODELMESH_EXAMPLE_MODELS}/{object_key} && "
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
        LOGGER.info(f"Extracting model from image {model_image} and uploading to MinIO: {object_key}")
        upload_pod.wait_for_status(status="Succeeded", timeout=300)

        # Get upload logs for verification
        try:
            upload_logs = upload_pod.log()
            LOGGER.info(f"Upload logs: {upload_logs}")
        except Exception as e:
            LOGGER.warning(f"Could not retrieve upload logs: {e}")

        LOGGER.info(
            f"Test model file uploaded successfully to s3://{MinIo.Buckets.MODELMESH_EXAMPLE_MODELS}/{object_key}"
        )
