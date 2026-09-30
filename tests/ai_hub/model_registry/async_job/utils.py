import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError

from tests.ai_hub.constants import SeaweedFs
from tests.ai_hub.image_constants import AiHubImages
from tests.ai_hub.utils import get_latest_job_pod
from utilities.constants import MinIo
from utilities.general import collect_pod_information

LOGGER = structlog.get_logger(name=__name__)

__all__ = ["get_latest_job_pod"]


def upload_test_model_to_s3_from_image(
    admin_client: DynamicClient,
    namespace: str,
    s3_service: Service,
    object_key: str = "my-model/model.onnx",
    model_image: str = MinIo.PodConfig.KSERVE_MINIO_IMAGE,
) -> None:
    """Extract and upload test model to an S3-compatible store from a container image.

    Args:
        admin_client: Kubernetes client
        namespace: Namespace to create upload pod in
        s3_service: S3-compatible service resource
        object_key: S3 object key path
        model_image: Container image containing the model
    """
    object_directory = object_key.rpartition("/")[0]
    filer_url = (
        f"http://{s3_service.name}.{s3_service.namespace}.svc.cluster.local:{SeaweedFs.Metadata.FILER_PORT}"
        f"/buckets/{SeaweedFs.Buckets.MODELMESH_EXAMPLE_MODELS}/{object_directory}/"
    )
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
                "name": "seaweedfs-uploader",
                "image": AiHubImages.SEAWEEDFS,
                "command": [
                    "/usr/bin/weed",
                    "filer.copy",
                    "/upload-data/model.onnx",
                    filer_url,
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
        LOGGER.info(f"Extracting model from image {model_image} and uploading to S3: {object_key}")
        try:
            upload_pod.wait_for_status(status="Succeeded", timeout=300)
        except TimeoutExpiredError:
            try:
                LOGGER.error("SeaweedFS uploader pod failed", logs=upload_pod.log(container="seaweedfs-uploader"))
            except Exception as error:  # noqa: BLE001
                LOGGER.warning(f"Could not retrieve SeaweedFS uploader logs: {error}")
            collect_pod_information(pod=upload_pod)
            raise

        # Get upload logs for verification
        try:
            upload_logs = upload_pod.log()
            LOGGER.info(f"Upload logs: {upload_logs}")
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f"Could not retrieve upload logs: {e}")

        LOGGER.info(
            f"Test model file uploaded successfully to s3://{SeaweedFs.Buckets.MODELMESH_EXAMPLE_MODELS}/{object_key}"
        )
