import pytest
import time
from typing import Any, Generator
from huggingface_hub import HfApi
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from tests.model_registry.model_catalog.constants import HF_CUSTOM_MODE

from tests.model_registry.model_catalog.huggingface.utils import get_huggingface_model_from_api
from utilities.infra import create_ns
from utilities.operator_utils import get_cluster_service_version
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.pod import Pod
from pytest_testconfig import py_config
import base64
import portforward

LOGGER = get_logger(name=__name__)


def get_openvino_image_from_rhoai_csv(admin_client: DynamicClient) -> str:
    """
    Get the OpenVINO model server image from the RHOAI ClusterServiceVersion.

    Returns:
        str: The OpenVINO model server image URL from RHOAI CSV

    Raises:
        Exception: If unable to find the image in the CSV
    """
    # Get the RHOAI CSV using the utility function
    csv = get_cluster_service_version(
        client=admin_client, prefix="rhods-operator", namespace=py_config["applications_namespace"]
    )

    # Look for OpenVINO image in spec.relatedImages
    related_images = csv.instance.spec.get("relatedImages", [])

    for image_info in related_images:
        image_url = image_info.get("image", "")
        if "odh-openvino-model-server" in image_url:
            LOGGER.info(f"Found OpenVINO image from RHOAI CSV: {image_url}")
            return image_url

    raise Exception("Could not find odh-openvino-model-server image in RHOAI CSV relatedImages")


@pytest.fixture()
def huggingface_api():
    return HfApi()


@pytest.fixture()
def num_models_from_hf_api_with_matching_criteria(request: pytest.FixtureRequest, huggingface_api: HfApi) -> int:
    excluded_str = request.param.get("excluded_str")
    included_str = request.param.get("included_str")
    models = huggingface_api.list_models(author=request.param["org_name"], limit=10000)
    model_list = []
    for model in models:
        if excluded_str:
            if model.id.endswith(excluded_str):
                LOGGER.info(f"Skipping {model.id} due to {excluded_str}")
                continue
            else:
                LOGGER.info(f"Adding {model.id}")
                model_list.append(model.id)
        elif included_str:
            if model.id.startswith(included_str):
                LOGGER.info(f"Adding {model.id}")
                model_list.append(model.id)
            else:
                LOGGER.info(f"Skipping {model.id} due to {included_str}")
                continue
        else:
            model_list.append(model.id)
    return len(model_list)


@pytest.fixture(scope="module")
def epoch_time_before_config_map_update() -> float:
    """
    Return the current epoch time in milliseconds when the test class starts.
    Useful for comparing against timestamps created during test execution.
    """
    return float(time.time() * 1000)


@pytest.fixture(scope="function")
def initial_last_synced_values(
    request: pytest.FixtureRequest,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> str:
    """
    Collect initial last_synced values for a given model.
    """
    result = get_huggingface_model_from_api(
        model_registry_rest_headers=model_registry_rest_headers,
        model_catalog_rest_url=model_catalog_rest_url,
        model_name=request.param,
        source_id="hf_id",
    )

    return result["customProperties"]["last_synced"]["string_value"]


@pytest.fixture(scope="class")
def hugging_face_deployment_ns(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    """
    Create a dedicated namespace for Hugging Face model deployments and testing.
    Similar to oci_namespace but specifically for HuggingFace model serving tests.
    """
    with create_ns(
        name="hf-deployment-ns",
        admin_client=admin_client,
    ) as ns:
        LOGGER.info(f"Created Hugging Face deployment namespace: {ns.name}")
        yield ns


@pytest.fixture(scope="class")
def huggingface_connection_secret(
    admin_client: DynamicClient,
    hugging_face_deployment_ns: Namespace,
) -> Generator[Secret, Any, Any]:
    """
    Create a connection secret for the HuggingFace model URI.
    This secret is required by the ODH admission webhook when creating InferenceServices
    with the opendatahub.io/connections annotation.
    """
    resource_name = "hf-test-inference-service-connection"
    hf_model_uri = f"hf://{HF_CUSTOM_MODE}"

    # Base64 encode the HuggingFace URI
    encoded_uri = base64.b64encode(hf_model_uri.encode()).decode()

    # Annotations matching the connection secret structure
    annotations = {
        "opendatahub.io/connection-type-protocol": "uri",
        "opendatahub.io/connection-type-ref": "uri-v1",
        "openshift.io/display-name": resource_name,
    }

    # Labels for ODH integration
    labels = {
        "opendatahub.io/dashboard": "false",
    }

    with Secret(
        client=admin_client,
        name=resource_name,
        namespace=hugging_face_deployment_ns.name,
        annotations=annotations,
        label=labels,
        data_dict={"URI": encoded_uri},
        teardown=True,
    ) as connection_secret:
        LOGGER.info(
            f"Created HuggingFace connection secret: {resource_name} in namespace: {hugging_face_deployment_ns.name}"
        )
        yield connection_secret


@pytest.fixture(scope="class")
def huggingface_inference_service(
    admin_client: DynamicClient,
    hugging_face_deployment_ns: Namespace,
    huggingface_serving_runtime: ServingRuntime,
    huggingface_connection_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    """
    Create an InferenceService for testing Hugging Face models.
    Based on the manually created HF_CUSTOM_MODE example with comprehensive ODH dashboard integration.
    Includes ONNX model format, hf:// storage URI, and full annotation set.

    Note: Some annotations like 'opendatahub.io/model-type: predictive',
    'opendatahub.io/connections', and hardware profile annotations would require
    custom annotation support in create_isvc or using raw InferenceService constructor.
    """
    name = "hf-test-inference-service"
    hf_model_uri = f"hf://{HF_CUSTOM_MODE}"
    runtime_name = huggingface_serving_runtime.name

    # Resources matching the manually created example
    resources = {"limits": {"cpu": "2", "memory": "4Gi"}, "requests": {"cpu": "2", "memory": "4Gi"}}

    # Labels for ODH dashboard integration
    labels = {
        "opendatahub.io/dashboard": "true",
    }

    # Comprehensive annotations matching the manually created examples with full ODH integration
    annotations = {
        "opendatahub.io/connections": huggingface_connection_secret.name,  # Reference to connection secret
        "opendatahub.io/hardware-profile-name": "default-profile",
        "opendatahub.io/hardware-profile-namespace": "redhat-ods-applications",
        "opendatahub.io/model-type": "predictive",
        "openshift.io/description": "",
        "openshift.io/display-name": f"huggingface/{name}",
        "security.opendatahub.io/enable-auth": "false",
        "serving.kserve.io/deploymentMode": "RawDeployment",
    }

    # Predictor configuration matching the manually created example
    predictor_dict = {
        "automountServiceAccountToken": False,
        "deploymentStrategy": {"type": "RollingUpdate"},
        "maxReplicas": 1,
        "minReplicas": 1,
        "model": {
            "modelFormat": {"name": "onnx", "version": "1"},
            "name": "",
            "resources": resources,
            "runtime": runtime_name,
            "storageUri": hf_model_uri,
        },
    }

    with InferenceService(
        client=admin_client,
        name=name,
        namespace=hugging_face_deployment_ns.name,
        annotations=annotations,
        label=labels,
        predictor=predictor_dict,
        teardown=True,
    ) as inference_service:
        # Wait for InferenceService to become Ready (similar to create_isvc wait=True)
        inference_service.wait_for_condition(
            condition="Ready",
            status="True",
            timeout=600,  # 10 minutes timeout for model loading
        )
        LOGGER.info(f"Created HuggingFace InferenceService: {name} in namespace: {hugging_face_deployment_ns.name}")
        yield inference_service


@pytest.fixture(scope="class")
def huggingface_serving_runtime(
    admin_client: DynamicClient,
    hugging_face_deployment_ns: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    """
    Create a ServingRuntime for OpenVINO Model Server to support Hugging Face models.
    Based on the manually created examples with complete ODH dashboard integration.
    Includes all template metadata annotations for full compatibility.
    """
    runtime_name = "hf-test-runtime"

    # Complete annotations matching manually created examples
    annotations = {
        "opendatahub.io/apiProtocol": "REST",
        "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
        "opendatahub.io/runtime-version": "v2025.4",
        "opendatahub.io/serving-runtime-scope": "global",
        "opendatahub.io/template-display-name": "OpenVINO Model Server",
        "opendatahub.io/template-name": "kserve-ovms",
        "openshift.io/display-name": "OpenVINO Model Server",
    }

    # Labels for ODH dashboard integration
    labels = {
        "opendatahub.io/dashboard": "true",
    }

    # Supported model formats matching the example YAML
    supported_model_formats = [
        {"autoSelect": True, "name": "openvino_ir", "version": "opset13"},
        {"name": "onnx", "version": "1"},
        {"autoSelect": True, "name": "tensorflow", "version": "1"},
        {"autoSelect": True, "name": "tensorflow", "version": "2"},
        {"autoSelect": True, "name": "paddle", "version": "2"},
        {"autoSelect": True, "name": "pytorch", "version": "2"},
    ]

    # Complete ServingRuntime specification
    runtime_spec = {
        "annotations": {
            "opendatahub.io/kserve-runtime": "ovms",
            "prometheus.io/path": "/metrics",
            "prometheus.io/port": "8888",
        },
        "containers": [
            {
                "args": [
                    "--model_name={{.Name}}",
                    "--port=8001",
                    "--rest_port=8888",
                    "--model_path=/mnt/models",
                    "--file_system_poll_wait_seconds=0",
                    "--metrics_enable",
                ],
                "image": get_openvino_image_from_rhoai_csv(admin_client),
                "name": "kserve-container",
                "ports": [{"containerPort": 8888, "protocol": "TCP"}],
            }
        ],
        "multiModel": False,
        "protocolVersions": ["v2", "grpc-v2"],
        "supportedModelFormats": supported_model_formats,
    }

    # Create the ServingRuntime with complete configuration
    runtime_dict = {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": runtime_name,
            "namespace": hugging_face_deployment_ns.name,
            "annotations": annotations,
            "labels": labels,
        },
        "spec": runtime_spec,
    }

    with ServingRuntime(
        client=admin_client,
        kind_dict=runtime_dict,
        teardown=True,
    ) as serving_runtime:
        LOGGER.info(f"Created OpenVINO ServingRuntime: {runtime_name} in namespace: {hugging_face_deployment_ns.name}")
        yield serving_runtime


@pytest.fixture(scope="class")
def huggingface_predictor_pod(
    admin_client: DynamicClient,
    hugging_face_deployment_ns: Namespace,
    huggingface_inference_service: InferenceService,
) -> Pod:
    """
    Get the predictor pod for the HuggingFace InferenceService.
    """
    namespace = hugging_face_deployment_ns.name
    label_selector = f"serving.kserve.io/inferenceservice={huggingface_inference_service.name}"

    pods = Pod.get(
        client=admin_client,
        namespace=namespace,
        label_selector=label_selector,
    )

    predictor_pods = [pod for pod in pods if "predictor" in pod.name]
    if not predictor_pods:
        raise Exception(f"No predictor pods found for InferenceService {huggingface_inference_service.name}")

    pod = predictor_pods[0]  # Use the first predictor pod
    LOGGER.info(f"Found predictor pod: {pod.name} in namespace: {namespace}")
    return pod


@pytest.fixture(scope="class")
def huggingface_model_portforward(
    hugging_face_deployment_ns: Namespace,
    huggingface_inference_service: InferenceService,
    huggingface_predictor_pod: Pod,
) -> Generator[str, Any, None]:
    """
    Port-forwards the HuggingFace OpenVINO model server pod to access the model API locally.
    Equivalent CLI:
      oc -n hf-deployment-ns port-forward pod/<pod-name> 8080:8888
    """
    namespace = hugging_face_deployment_ns.name
    local_port = 9999
    remote_port = 8888  # OpenVINO Model Server REST port
    local_url = f"http://localhost:{local_port}/v1/models"

    try:
        with portforward.forward(
            pod_or_service=huggingface_predictor_pod.name,
            namespace=namespace,
            from_port=local_port,
            to_port=remote_port,
            waiting=20,
        ):
            LOGGER.info(f"HuggingFace model port-forward established: {local_url}")
            LOGGER.info(f"Test with: curl -s {local_url}{huggingface_inference_service.name}")
            yield local_url
    except Exception as expt:
        LOGGER.error(f"Failed to set up port forwarding for pod {huggingface_predictor_pod.name}: {expt}")
        raise
