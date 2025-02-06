from __future__ import annotations

import base64

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

import utilities.infra
from utilities.constants import Annotations, KServeDeploymentType, MODELMESH_SERVING

LOGGER = get_logger(name=__name__)


def get_s3_secret_dict(
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
) -> dict[str, str]:
    """
    Returns a dictionary of s3 secret values

    Args:
        aws_access_key (str): AWS access key
        aws_secret_access_key (str): AWS secret key
        aws_s3_bucket (str): AWS S3 bucket
        aws_s3_endpoint (str): AWS S3 endpoint
        aws_s3_region (str): AWS S3 region

    Returns:
        dict[str, str]: A dictionary of s3 secret encoded values

    """
    return {
        "AWS_ACCESS_KEY_ID": b64_encoded_string(string_to_encode=aws_access_key),
        "AWS_SECRET_ACCESS_KEY": b64_encoded_string(string_to_encode=aws_secret_access_key),
        "AWS_S3_BUCKET": b64_encoded_string(string_to_encode=aws_s3_bucket),
        "AWS_S3_ENDPOINT": b64_encoded_string(string_to_encode=aws_s3_endpoint),
        "AWS_DEFAULT_REGION": b64_encoded_string(string_to_encode=aws_s3_region),
    }


def b64_encoded_string(string_to_encode: str) -> str:
    """Returns openshift compliant base64 encoding of a string

    encodes the input string to bytes-like, encodes the bytes-like to base 64,
    decodes the b64 to a string and returns it. This is needed for openshift
    resources expecting b64 encoded values in the yaml.

    Args:
        string_to_encode: The string to encode in base64

    Returns:
        A base64 encoded string that is compliant with openshift's yaml format
    """
    return base64.b64encode(string_to_encode.encode()).decode()


def download_model_data(
    admin_client: DynamicClient,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    model_namespace: str,
    model_pvc_name: str,
    bucket_name: str,
    aws_endpoint_url: str,
    aws_default_region: str,
    model_path: str,
    use_sub_path: bool = False,
) -> str:
    """
    Downloads the model data from the bucket to the PVC

    Args:
        admin_client (DynamicClient): Admin client
        aws_access_key_id (str): AWS access key
        aws_secret_access_key (str): AWS secret key
        model_namespace (str): Namespace of the model
        model_pvc_name (str): Name of the PVC
        bucket_name (str): Name of the bucket
        aws_endpoint_url (str): AWS endpoint URL
        aws_default_region (str): AWS default region
        model_path (str): Path to the model
        use_sub_path (bool): Whether to use a sub path

    Returns:
        str: Path to the model path

    """
    volume_mount = {"mountPath": "/mnt/models/", "name": model_pvc_name}
    if use_sub_path:
        volume_mount["subPath"] = model_path

    pvc_model_path = f"/mnt/models/{model_path}"
    init_containers = [
        {
            "name": "init-container",
            "image": "quay.io/quay/busybox@sha256:92f3298bf80a1ba949140d77987f5de081f010337880cd771f7e7fc928f8c74d",
            "command": ["sh"],
            "args": ["-c", f"mkdir -p {pvc_model_path} && chmod -R 777 {pvc_model_path}"],
            "volumeMounts": [volume_mount],
        }
    ]
    containers = [
        {
            "name": "model-downloader",
            "image": utilities.infra.get_kserve_storage_initialize_image(client=admin_client),
            "args": [
                f"s3://{bucket_name}/{model_path}/",
                pvc_model_path,
            ],
            "env": [
                {"name": "AWS_ACCESS_KEY_ID", "value": aws_access_key_id},
                {"name": "AWS_SECRET_ACCESS_KEY", "value": aws_secret_access_key},
                {"name": "S3_USE_HTTPS", "value": "1"},
                {"name": "AWS_ENDPOINT_URL", "value": aws_endpoint_url},
                {"name": "AWS_DEFAULT_REGION", "value": aws_default_region},
                {"name": "S3_VERIFY_SSL", "value": "false"},
                {"name": "awsAnonymousCredential", "value": "false"},
            ],
            "volumeMounts": [volume_mount],
        }
    ]
    volumes = [{"name": model_pvc_name, "persistentVolumeClaim": {"claimName": model_pvc_name}}]

    with Pod(
        client=admin_client,
        namespace=model_namespace,
        name="download-model-data",
        init_containers=init_containers,
        containers=containers,
        volumes=volumes,
        restart_policy="Never",
    ) as pod:
        pod.wait_for_status(status=Pod.Status.RUNNING)
        LOGGER.info("Waiting for model download to complete")
        pod.wait_for_status(status=Pod.Status.SUCCEEDED, timeout=25 * 60)

    return model_path


def create_isvc_label_selector_str(isvc: InferenceService, resource_type: str, runtime_name: str | None = None) -> str:
    """
    Creates a label selector string for the given InferenceService.

    Args:
        isvc (InferenceService): InferenceService object
        resource_type (str): Type of the resource: service or other for model mesh
        runtime_name (str): ServingRuntime name

    Returns:
        str: Label selector string

    Raises:
        ValueError: If the deployment mode is not supported

    """
    deployment_mode = isvc.instance.metadata.annotations.get(Annotations.KserveIo.DEPLOYMENT_MODE)
    if deployment_mode in (
        KServeDeploymentType.SERVERLESS,
        KServeDeploymentType.RAW_DEPLOYMENT,
    ):
        return f"{isvc.ApiGroup.SERVING_KSERVE_IO}/inferenceservice={isvc.name}"

    elif deployment_mode == KServeDeploymentType.MODEL_MESH:
        if resource_type == "service":
            return f"modelmesh-service={MODELMESH_SERVING}"
        else:
            return f"name={MODELMESH_SERVING}-{runtime_name}"

    else:
        raise ValueError(f"Unknown deployment mode {deployment_mode}")
