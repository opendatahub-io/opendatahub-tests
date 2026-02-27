import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.utils import get_hf_catalog_str

LOGGER = get_logger(name=__name__)

pytestmark = [pytest.mark.skip_on_disconnected]


@pytest.mark.parametrize(
    "updated_catalog_config_map",
    [
        pytest.param(
            {
                "sources_yaml": get_hf_catalog_str(ids=["custom"]),
            },
            id="validate_hf_fields",
            marks=pytest.mark.install,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("hugging_face_deployment_ns", "updated_catalog_config_map")
class TestHuggingFaceModelDeployment:
    """
    Test class for Hugging Face model deployment functionality.
    Tests the complete deployment workflow from ServingRuntime to InferenceService.
    """

    @pytest.mark.tier1
    def test_huggingface_model_deployment_end_to_end(
        self,
        admin_client: DynamicClient,
        hugging_face_deployment_ns: Namespace,
        huggingface_serving_runtime: ServingRuntime,
        huggingface_inference_service: InferenceService,
        huggingface_model_portforward: str,
    ) -> None:
        """
        Test HuggingFace model deployment API endpoints.
        Validates that the deployed model is accessible via the OpenVINO Model Server API.
        TODO: When adequate coverage us added ib dashboard tests, this would be removed
        """
        model_endpoint = f"{huggingface_model_portforward}/{huggingface_inference_service.name}"
        LOGGER.info(f"Testing model endpoint: {model_endpoint}")
        model_response = requests.get(model_endpoint, timeout=10)
        LOGGER.info(f"Model endpoint status: {model_response.status_code}")

        if model_response.status_code == 200:
            LOGGER.info(f"Model '{huggingface_inference_service.name}' details: {model_response.json()}")
        else:
            LOGGER.error(f"Model endpoint returned {model_response.status_code}: {model_response.text}")
            pytest.fail("Model may not be accessible via API yet")
