import base64
import re
import pytest
from typing import List, Dict, Tuple

from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from ocp_resources.cluster_service_version import ClusterServiceVersion
from pytest_testconfig import config as py_config

import utilities.infra
from utilities.constants import Annotations, KServeDeploymentType, MODELMESH_SERVING

# Constants for image validation
REGISTRY_REDHAT_IO = "registry.redhat.io"
SHA256_DIGEST_PATTERN = r"@sha256:[a-f0-9]{64}$"

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
    client: DynamicClient,
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
        client (DynamicClient): Admin client
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
            "image": utilities.infra.get_kserve_storage_initialize_image(client=client),
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
        client=client,
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


def get_csv_related_images(admin_client: DynamicClient, csv_name: str | None = None) -> List[Dict[str, str]]:
    """Get relatedImages from the CSV.

    Args:
        admin_client: The kubernetes client
        csv_name: Optional CSV name. If not provided, will use {operator_name}.{version}
                 where operator_name is determined by the distribution (rhods-operator for OpenShift AI,
                 opendatahub-operator for Open Data Hub)

    Returns:
        List of related images from the CSV
    """
    from utilities.infra import (
        get_product_version,
        get_operator_distribution,
    )  # Import here to avoid circular dependency

    if csv_name is None:
        distribution = get_operator_distribution(client=admin_client)
        operator_name = "opendatahub-operator" if distribution == "Open Data Hub" else "rhods-operator"
        csv_name = f"{operator_name}.{get_product_version(admin_client=admin_client)}"

    csvs = list(
        ClusterServiceVersion.get(
            dyn_client=admin_client,
            name=csv_name,
            namespace=py_config["applications_namespace"],
        )
    )

    if not csvs:
        raise ValueError(f"CSV {csv_name} not found in namespace {py_config['applications_namespace']}")

    csv = csvs[0]  # Get the first (and should be only) CSV instance
    if len(csvs) > 1:
        LOGGER.warning(f"Multiple CSV instances found for {csv_name}, using the first one")
    if not csv.instance:
        raise ValueError(f"CSV {csv_name} has no instance data")

    return csv.instance.spec.relatedImages


def get_pod_images(pod: Pod) -> List[str]:
    """Get all container images from a pod.

    Args:
        pod: The pod to get images from

    Returns:
        List of container image strings
    """
    return [container.image for container in pod.instance.spec.containers]


def validate_image_format(image: str) -> Tuple[bool, str]:
    """Validate image format according to requirements.

    Args:
        image: The image string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not image.startswith(REGISTRY_REDHAT_IO):
        return False, f"Image {image} is not from {REGISTRY_REDHAT_IO}"

    if not re.search(SHA256_DIGEST_PATTERN, image):
        return False, f"Image {image} does not use sha256 digest"

    return True, ""


def get_pods_by_labels(
    admin_client: DynamicClient,
    namespace: str,
    label_selector: str,
) -> list[Pod]:
    """
    Get pods by label selector in a namespace.

    Args:
        admin_client: The admin client to use for pod retrieval
        namespace: The namespace to search in
        label_selector: The label selector to filter pods

    Returns:
        List of matching pods

    Raises:
        pytest.fail: If no pods are found
    """
    pods = list(
        Pod.get(
            dyn_client=admin_client,
            namespace=namespace,
            label_selector=label_selector,
        )
    )
    if not pods:
        pytest.fail(f"No pods found with label selector {label_selector} in namespace {namespace}")
    return pods


def validate_container_images(
    pod: Pod,
    valid_image_refs: set[str],
    skip_patterns: list[str] | None = None,
) -> list[str]:
    """
    Validate all container images in a pod against a set of valid image references.

    Args:
        pod: The pod whose images to validate
        valid_image_refs: Set of valid image references to check against
        skip_patterns: List of patterns to skip validation for (e.g. ["openshift-service-mesh"])

    Returns:
        List of validation error messages, empty if all validations pass
    """
    validation_errors = []
    skip_patterns = skip_patterns or []

    pod_images = get_pod_images(pod=pod)
    for image in pod_images:
        # Skip images matching any skip patterns
        if any(pattern in image for pattern in skip_patterns):
            LOGGER.warning(f"Skipping image {image} as it matches skip patterns")
            continue

        # Validate image format
        is_valid, error_msg = validate_image_format(image=image)
        if not is_valid:
            validation_errors.append(f"Pod {pod.name} image validation failed: {error_msg}")

        # Check if image is in valid references
        if image not in valid_image_refs:
            validation_errors.append(f"Pod {pod.name} image {image} is not in valid image references")

    return validation_errors
