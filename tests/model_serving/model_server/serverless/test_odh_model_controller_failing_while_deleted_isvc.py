import pytest
from ocp_utilities.infra import get_pods_by_name_prefix
from pytest_testconfig import py_config
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler, TimeoutExpiredError

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelFormat,
    ModelInferenceRuntime,
    ModelVersion,
    Protocols,
    Timeout,
)
from utilities.exceptions import PodLogMissMatchError
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.serverless,
    pytest.mark.sanity,
    pytest.mark.usefixtures("valid_aws_config"),
]

LOGGER = get_logger(name=__name__)


@pytest.mark.serverless
@pytest.mark.parametrize(
    "model_namespace, openvino_kserve_serving_runtime, ovms_serverless_inference_service",
    [
        pytest.param(
            {"name": "serverless-maistra"},
            {
                "runtime-name": ModelInferenceRuntime.ONNX_RUNTIME,
                "model-format": {ModelFormat.ONNX: ModelVersion.OPSET13},
            },
            {
                "name": ModelFormat.ONNX,
                "model-version": ModelVersion.OPSET13,
                "model-dir": "test-dir",
            },
        )
    ],
    indirect=True,
)
class TestLogForOdHModelController:
    def test_odh_model_controller_failing_while_deleted_isvc(self, ovms_serverless_inference_service):
        """Verify model can be queried"""
        verify_inference_response(
            inference_service=ovms_serverless_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    def test_moniter_odh_logs_doe_maistra(self, deleted_isvc, admin_client):
        """Delete isvc and check pod logs for unexpected 'Maistra' string"""
        search_string = "maistra.io"
        pod = get_pods_by_name_prefix(
            client=admin_client, namespace=py_config["applications_namespace"], pod_prefix="odh-model-controller"
        )[0]

        try:
            log_sampler = TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_1MIN,
                sleep=5,
                func=lambda: pod.log(container="manager", tail_lines=100, timestamps=True),
            )

            for log_output in log_sampler:
                LOGGER.info("Log output fetched during sampling:")
                if search_string in log_output:
                    raise PodLogMissMatchError(f"'{search_string}' was found in pod logs")
            LOGGER.info(f"'{search_string}' was not found in pod logs within timeout")

        except TimeoutExpiredError as e:
            LOGGER.error(f"Timeout while sampling logs: {str(e)}")
