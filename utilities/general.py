import base64
import re
from typing import List, Tuple
import uuid

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, NotFoundError
from ocp_resources.inference_graph import InferenceGraph
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

import utilities.infra
from utilities.constants import Annotations, KServeDeploymentType, MODELMESH_SERVING, Timeout
from utilities.exceptions import UnexpectedResourceCountError, ResourceValueMismatch
from ocp_resources.resource import Resource
from timeout_sampler import retry

# Constants for image validation
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
    if not image.startswith(Resource.ApiGroup.IMAGE_REGISTRY):
        return False, f"Image {image} is not from {Resource.ApiGroup.IMAGE_REGISTRY}"

    if not re.search(SHA256_DIGEST_PATTERN, image):
        return False, f"Image {image} does not use sha256 digest"

    return True, ""


@retry(
    wait_timeout=60,
    sleep=5,
    exceptions_dict={ResourceNotFoundError: [], UnexpectedResourceCountError: []},
)
def wait_for_pods_by_labels(
    admin_client: DynamicClient,
    namespace: str,
    label_selector: str,
    expected_num_pods: int,
) -> list[Pod]:
    """
    Get pods by label selector in a namespace.

    Args:
        admin_client: The admin client to use for pod retrieval
        namespace: The namespace to search in
        label_selector: The label selector to filter pods
        expected_num_pods: The expected number of pods to be found
    Returns:
        List of matching pods

    Raises:
        ResourceNotFoundError: If no pods are found
    """
    pods = list(
        Pod.get(
            dyn_client=admin_client,
            namespace=namespace,
            label_selector=label_selector,
        )
    )
    if not pods:
        raise ResourceNotFoundError(f"No pods found with label selector {label_selector} in namespace {namespace}")
    if len(pods) != expected_num_pods:
        raise UnexpectedResourceCountError(f"Expected {expected_num_pods} pods, found {len(pods)}")
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


def create_ig_pod_label_selector_str(ig: InferenceGraph) -> str:
    """
    Creates a pod label selector string for the given InferenceGraph.

    Args:
        ig (InferenceGraph): InferenceGraph object

    Returns:
        str: Label selector string for fetching IG pods

    """
    return f"serving.kserve.io/inferencegraph={ig.name}"


def generate_random_name(prefix: str = "", length: int = 8) -> str:
    """
    Generates a name with a required prefix and a random suffix derived from a UUID.

    The length of the random suffix can be controlled, defaulting to 8 characters.
    The suffix is taken from the beginning of a V4 UUID's hex representation.

    Args:
        prefix (str): The required prefix for the generated name.
        length (int, optional): The desired length for the UUID-derived suffix.
                               Defaults to 8. Must be between 1 and 32.

    Returns:
        str: A string in the format "prefix-uuid_suffix".

    Raises:
        ValueError: If prefix is empty, or if length is not between 1 and 32.
    """
    if not isinstance(length, int) or not (1 <= length <= 32):
        raise ValueError("suffix_length must be an integer between 1 and 32.")
    # Generate a new random UUID (version 4)
    random_uuid = uuid.uuid4()
    # Use the first 'length' characters of the hexadecimal representation of the UUID as the suffix.
    # random_uuid.hex is 32 characters long.
    suffix = random_uuid.hex[:length]
    return f"{prefix}-{suffix}" if prefix else suffix


@retry(
    wait_timeout=Timeout.TIMEOUT_15_SEC,
    sleep=1,
    exceptions_dict={ResourceValueMismatch: [], ResourceNotFoundError: [], NotFoundError: []},
)
def wait_for_container_status(pod: Pod, container_name: str, expected_status: str) -> bool:
    """
    Wait for a container to be in the expected status.

    Args:
        pod: The pod to wait for
        container_name: The name of the container to wait for
        expected_status: The expected status

    Returns:
        bool: True if the container is in the expected status, False otherwise

    Raises:
        ResourceValueMismatch: If the container is not in the expected status
    """

    container_status = None
    for cs in pod.instance.status.get("containerStatuses", []):
        if cs.name == container_name:
            container_status = cs
            break
    if container_status is None:
        raise ResourceValueMismatch(f"Container {container_name} not found in pod {pod.name}")

    if container_status.state.waiting:
        reason = container_status.state.waiting.reason
    elif container_status.state.terminated:
        reason = container_status.state.terminated.reason
    elif container_status.state.running:
        # Running container does not have a reason
        reason = "Running"
    else:
        raise ResourceValueMismatch(
            f"{container_name} in {pod.name} is in an unrecognized or transitional state: {container_status.state}"
        )

    if reason == expected_status:
        LOGGER.info(f"Container {container_name} is in the expected status {expected_status}")
        return True
    raise ResourceValueMismatch(f"Container {container_name} is not in the expected status {container_status.state}")
