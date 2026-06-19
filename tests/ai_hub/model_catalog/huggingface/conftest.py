import base64
import time
from collections.abc import Generator
from typing import Any

import portforward
import pytest
import structlog
from huggingface_hub import HfApi
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from tests.ai_hub.model_catalog.constants import HF_CUSTOM_MODE
from tests.ai_hub.model_catalog.huggingface.utils import get_huggingface_model_from_api
from tests.ai_hub.model_catalog.utils import get_models_from_catalog_api
from utilities.constants import RuntimeTemplates
from utilities.infra import create_ns
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = structlog.get_logger(name=__name__)


class PredictorPodNotFoundError(Exception):
    """Exception raised when predictor pods are not found for an InferenceService."""


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
def hf_source_filter(request: pytest.FixtureRequest) -> str:
    """
    Provide the HuggingFace source filter label for test classes.
    Can be overridden via indirect parameterization.
    """
    return request.param.get("source_filter", "HuggingFace Source mixed")


@pytest.fixture(scope="class")
def all_models_unfiltered(
    updated_catalog_config_map: ConfigMap,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    hf_source_filter: str,
) -> dict:
    """
    Fetch all models once at class scope to avoid redundant API calls.
    This cached result is shared across all parameterized test runs.

    Args:
        updated_catalog_config_map: The catalog ConfigMap (ensures catalog is updated)
        model_catalog_rest_url: The catalog REST API URL
        model_registry_rest_headers: Headers for API authentication
        hf_source_filter: The source label filter

    Returns:
        dict: API response containing all models from the specified source
    """
    LOGGER.info(f"Fetching all models from source '{hf_source_filter}'")
    return get_models_from_catalog_api(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label=hf_source_filter,
        page_size=1000,  # Large page size to get all models
    )


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
    """Create an OVMS ServingRuntime from the cluster template for Hugging Face models."""
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="hf-test-runtime",
        namespace=hugging_face_deployment_ns.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as serving_runtime:
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
        raise PredictorPodNotFoundError(
            f"No predictor pods found for InferenceService {huggingface_inference_service.name}"
        )

    pod = predictor_pods[0]  # Use the first predictor pod
    LOGGER.info(f"Found predictor pod: {pod.name} in namespace: {namespace}")
    return pod


@pytest.fixture(scope="class")
def huggingface_model_portforward(
    hugging_face_deployment_ns: Namespace,
    huggingface_inference_service: InferenceService,
    huggingface_predictor_pod: Pod,
) -> Generator[str, Any]:
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
