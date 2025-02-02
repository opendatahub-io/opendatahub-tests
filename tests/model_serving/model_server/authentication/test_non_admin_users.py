import pytest

from tests.model_serving.model_server.utils import (
    run_inference_multiple_times,
    verify_inference_response,
)
from utilities.constants import ModelFormat, ModelName, ModelStoragePath, Protocols
from utilities.inference_utils import Inference
from utilities.manifests.caikit_tgis import CAIKIT_TGIS_INFERENCE_CONFIG
from utilities.manifests.tgis_grpc import TGIS_INFERENCE_CONFIG
from utilities.monitoring import validate_metrics_value


@pytest.mark.parametrize(
    "unprivileged_model_namespace, unprivileged_s3_caikit_serverless_inference_service",
    [
        pytest.param(
            {"name": "non-admin-serverless"},
            {"model-dir": ModelStoragePath.FLAN_T5_SMALL_CAIKIT},
        )
    ],
    indirect=True,
)
class TestServerlessUnprivilegedUser:
    @pytest.mark.smoke
    @pytest.mark.serverless
    @pytest.mark.polarion("ODS-2552")
    def test_non_admin_deploy_serverless_and_query_model(self, unprivileged_s3_caikit_serverless_inference_service):
        """Verify non admin can deploy a model and query using REST"""
        verify_inference_response(
            inference_service=unprivileged_s3_caikit_serverless_inference_service,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, unprivileged_s3_caikit_raw_inference_service",
    [
        pytest.param(
            {"name": "non-admin-raw"},
            {"model-dir": ModelStoragePath.FLAN_T5_SMALL_HF},
        )
    ],
    indirect=True,
)
class TestRawUnprivilegedUser:
    @pytest.mark.sanity
    @pytest.mark.rawdeployment
    @pytest.mark.polarion("ODS-2611")
    def test_non_admin_deploy_raw_and_query_model(
        self,
        unprivileged_s3_caikit_raw_inference_service,
    ):
        """Verify non admin can deploy a Raw model and query using REST"""
        verify_inference_response(
            inference_service=unprivileged_s3_caikit_raw_inference_service,
            inference_config=TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.GRPC,
            model_name=ModelName.FLAN_T5_SMALL_HF,
            use_default_query=True,
        )

    def test_raw_metrics(
        self,
        unprivileged_s3_caikit_raw_inference_service,
        prometheus,
        user_workload_monitoring_config_map,
    ):
        """Verify number of total model requests in OpenShift monitoring system (UserWorkloadMonitoring) metrics"""
        total_runs = 5

        run_inference_multiple_times(
            isvc=unprivileged_s3_caikit_raw_inference_service,
            inference_config=TGIS_INFERENCE_CONFIG,
            inference_type=Inference.STREAMING,
            protocol=Protocols.GRPC,
            model_name=ModelName.FLAN_T5_SMALL_HF,
            iterations=total_runs,
            run_in_parallel=True,
        )
        validate_metrics_value(
            prometheus=prometheus,
            metrics_query="tgi_request_count",
            expected_value=str(total_runs + 1),
        )
