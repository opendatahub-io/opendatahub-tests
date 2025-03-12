import pytest

from tests.model_serving.model_server.utils import run_inference_multiple_times
from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelName,
    ModelStoragePath,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import Inference
from utilities.manifests.pytorch import PYTORCH_TGIS_INFERENCE_CONFIG
from utilities.manifests.tgis_grpc import TGIS_INFERENCE_CONFIG

pytestmark = [pytest.mark.serverless, pytest.mark.sanity]


@pytest.mark.polarion("ODS-2371")
@pytest.mark.parametrize(
    "model_namespace, serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "serverless-multi-tgis-models"},
            {
                "name": "tgis-runtime",
                "template-name": RuntimeTemplates.TGIS_GRPC_SERVING,
                "multi-model": False,
                "enable-http": False,
                "enable-grpc": True,
            },
            {
                "name": f"{ModelName.BLOOM_560M}-model",
                "deployment-mode": KServeDeploymentType.SERVERLESS,
                "model-dir": f"{ModelStoragePath.BLOOM_560M_CAIKIT}/artifacts",
                "external-route": True,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("s3_flan_small_hf_caikit_serverless_inference_service")
class TestServerlessMultipleProjectsInNamespace:
    def test_serverless_multi_tgis_models_inference_bloom(
        self,
        s3_models_inference_service,
    ):
        """Test inference with Bloom Caikit model when multiple models in the same namespace"""
        run_inference_multiple_times(
            isvc=s3_models_inference_service,
            inference_config=PYTORCH_TGIS_INFERENCE_CONFIG,
            model_name=ModelAndFormat.BLOOM_560M_CAIKIT,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.GRPC,
            run_in_parallel=True,
            iterations=5,
        )

    def test_serverless_multi_tgis_models_inference_flan(
        self, s3_flan_small_hf_caikit_serverless_inference_service, s3_models_inference_service
    ):
        """Test inference with Flan Caikit model when multiple models in the same namespace"""
        run_inference_multiple_times(
            isvc=s3_flan_small_hf_caikit_serverless_inference_service,
            inference_config=TGIS_INFERENCE_CONFIG,
            model_name=ModelAndFormat.FLAN_T5_SMALL_CAIKIT,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.GRPC,
            run_in_parallel=True,
            iterations=5,
        )
