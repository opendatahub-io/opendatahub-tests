import pytest

from tests.model_serving.model_server.utils import run_inference_multiple_times
from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelFormat,
    ModelInferenceRuntime,
    ModelStoragePath,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import Inference
from utilities.manifests.tgis_grpc import TGIS_INFERENCE_CONFIG

pytestmark = [pytest.mark.serverless, pytest.mark.sanity]


@pytest.mark.polarion("ODS-2371")
@pytest.mark.parametrize(
    "model_namespace, serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "serverless-multi-tgis-models"},
            {
                "name": f"{ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME}",
                "template-name": RuntimeTemplates.CAIKIT_TGIS_SERVING,
                "multi-model": False,
                "enable-http": False,
                "enable-grpc": True,
            },
            {
                "name": f"{ModelFormat.CAIKIT}-bloom",
                "deployment-mode": KServeDeploymentType.SERVERLESS,
                "model-dir": ModelStoragePath.BLOOM_560M_CAIKIT,
                "external-route": True,
            },
        )
    ],
    indirect=True,
)
class TestServerlessMultipleProjectsInNamespace:
    def test_serverless_multi_tgis_models_inference_bloom(
        self,
        s3_models_inference_service,
    ):
        """Test inference with Bloom Caikit model when multiple models in the same namespace"""
        run_inference_multiple_times(
            isvc=s3_models_inference_service,
            inference_config=TGIS_INFERENCE_CONFIG,
            model_name=ModelAndFormat.BLOOM_560M_CAIKIT,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.GRPC,
            run_in_parallel=True,
            iterations=5,
        )

    def test_serverless_multi_tgis_models_inference_flan(
        self, s3_flan_small_caikit_serverless_inference_service, s3_models_inference_service
    ):
        """Test inference with Flan Caikit model when multiple models in the same namespace"""
        run_inference_multiple_times(
            isvc=s3_flan_small_caikit_serverless_inference_service,
            inference_config=TGIS_INFERENCE_CONFIG,
            model_name=ModelAndFormat.FLAN_T5_SMALL_CAIKIT,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.GRPC,
            run_in_parallel=True,
            iterations=5,
        )
