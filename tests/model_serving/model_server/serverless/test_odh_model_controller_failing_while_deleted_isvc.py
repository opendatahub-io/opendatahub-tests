import pytest
import re
from ocp_utilities.infra import get_pods_by_name_prefix
from pytest_testconfig import py_config
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler, TimeoutExpiredError

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelVersion,
    Protocols,
    RunTimeConfigs,
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


@pytest.mark.parametrize(
    "model_namespace, ovms_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "serverless-maistra"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                "name": ModelFormat.ONNX,
                "model-version": ModelVersion.OPSET13,
                "model-dir": "test-dir",
                "deployment-mode": KServeDeploymentType.SERVERLESS,
            },
        )
    ],
    indirect=True,
)
class TestNoMaistraErrorInLogs:
    def test_inference_before_isvc_deletion(self, ovms_kserve_inference_service):
        """Verify model can be queried"""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    def test_pod_logs_do_not_contain_maistra_error(self, deleted_isvc, admin_client):
        """Delete isvc and check pod logs for 'ServiceMeshMemberRoll' error"""
        regex_pattern = r'"error":\s*"no matches for kind \\"ServiceMeshMemberRoll\\" in version \\"maistra\.io/v1\\""'
        pod = get_pods_by_name_prefix(
            client=admin_client, namespace=py_config["applications_namespace"], pod_prefix="odh-model-controller"
        )[0]

        try:
            log_sampler = TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_4MIN,
                sleep=5,
                func=pod.log,
                container="manager",
                tail_lines=500,
                timestamps=True,
            )

            for log_output in log_sampler:
                LOGGER.info("Log output fetched during sampling:")
                if re.search(regex_pattern, log_output):
                    raise PodLogMissMatchError("ServiceMeshMemberRoll error found in pod logs")

        except TimeoutExpiredError:
            LOGGER.info(f"Error {regex_pattern} not found in {pod.name} logs")
