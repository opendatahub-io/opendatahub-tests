import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError

from tests.ai_hub.constants import SeaweedFs
from tests.ai_hub.image_constants import AiHubImages
from tests.ai_hub.model_registry.async_job.constants import (
    CA_BUNDLE_CONFIG,
    MODEL_SYNC_CONFIG,
    VOLUME_MOUNTS,
)
from tests.ai_hub.utils import get_latest_job_pod
from utilities.constants import MinIo, OCIRegistry
from utilities.general import collect_pod_information

LOGGER = structlog.get_logger(name=__name__)

__all__ = ["build_async_job_config", "get_latest_job_pod", "upload_test_model_to_s3_from_image"]


def build_async_job_config(
    mr_server_address: str,
    sa_token: str,
    oci_registry_host: str,
    repo_name: str,
    include_ca_bundle: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Build volume mounts and environment variables for async job.

    Args:
        mr_server_address: Model Registry server address.
        sa_token: Service account token.
        oci_registry_host: OCI registry hostname.
        repo_name: OCI repository name.
        include_ca_bundle: Whether to include CA bundle volume and env var.

    Returns:
        Tuple of (volume_mounts, environment_variables).
    """
    volume_mounts = [
        {
            "name": "source-credentials",
            "readOnly": True,
            "mountPath": VOLUME_MOUNTS["SOURCE_CREDS_PATH"],
        },
        {
            "name": "destination-credentials",
            "readOnly": True,
            "mountPath": VOLUME_MOUNTS["DEST_CREDS_PATH"],
        },
    ]

    if include_ca_bundle:
        volume_mounts.append({
            "name": "ca-bundle",
            "readOnly": True,
            "mountPath": CA_BUNDLE_CONFIG["MOUNT_PATH"],
        })

    environment_variables = [
        {"name": "MODEL_SYNC_SOURCE_TYPE", "value": MODEL_SYNC_CONFIG["SOURCE_TYPE"]},
        {"name": "MODEL_SYNC_SOURCE_AWS_KEY", "value": MODEL_SYNC_CONFIG["SOURCE_AWS_KEY"]},
        {"name": "MODEL_SYNC_SOURCE_S3_CREDENTIALS_PATH", "value": VOLUME_MOUNTS["SOURCE_CREDS_PATH"]},
        {"name": "MODEL_SYNC_MODEL_ID", "value": MODEL_SYNC_CONFIG["MODEL_ID"]},
        {"name": "MODEL_SYNC_MODEL_VERSION_ID", "value": MODEL_SYNC_CONFIG["MODEL_VERSION_ID"]},
        {"name": "MODEL_SYNC_MODEL_ARTIFACT_ID", "value": MODEL_SYNC_CONFIG["MODEL_ARTIFACT_ID"]},
        {"name": "MODEL_SYNC_REGISTRY_SERVER_ADDRESS", "value": mr_server_address},
        {"name": "MODEL_SYNC_REGISTRY_USER_TOKEN", "value": sa_token},
        {"name": "MODEL_SYNC_REGISTRY_IS_SECURE", "value": "False"},
        {
            "name": "MODEL_SYNC_DESTINATION_OCI_REGISTRY",
            "value": f"{oci_registry_host}:{OCIRegistry.Metadata.DEFAULT_PORT}",
        },
        {"name": "MODEL_SYNC_DESTINATION_OCI_URI", "value": f"{oci_registry_host}/{repo_name}"},
        {"name": "MODEL_SYNC_DESTINATION_OCI_BASE_IMAGE", "value": MODEL_SYNC_CONFIG["DESTINATION_OCI_BASE_IMAGE"]},
        {
            "name": "MODEL_SYNC_DESTINATION_OCI_ENABLE_TLS_VERIFY",
            "value": MODEL_SYNC_CONFIG["DESTINATION_OCI_ENABLE_TLS_VERIFY"],
        },
    ]

    if include_ca_bundle:
        environment_variables.append({
            "name": CA_BUNDLE_CONFIG["ENV_VAR"],
            "value": f"{CA_BUNDLE_CONFIG['MOUNT_PATH']}/{CA_BUNDLE_CONFIG['CA_FILE']}",
        })

    return volume_mounts, environment_variables


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
                    (
                        "echo 'Creating test model file for async upload pipeline testing...' && "
                        "echo 'Test model file for validating the async upload pipeline' > /upload-data/model.onnx && "
                        "echo 'Test model file created successfully'"
                    )
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
