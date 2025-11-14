import pytest

from tests.model_serving.model_server.kserve.inference_service_configuration.constants import (
    BASE_ISVC_CONFIG,
    ISVC_ENV_VARS,
)
from tests.model_serving.model_server.kserve.inference_service_configuration.utils import (
    verify_env_vars_in_isvc_pods,
)
from utilities.constants import KServeDeploymentType, RunTimeConfigs

pytestmark = [pytest.mark.sanity, pytest.mark.usefixtures("valid_aws_config")]


ISVC_ENV_VARS_CONFIG = {
    "env-vars": ISVC_ENV_VARS,
}
RAW_DEPLOYMENT_ISVC_CONFIG = {
    **BASE_ISVC_CONFIG,
    **ISVC_ENV_VARS_CONFIG,
    "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
}


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "test-raw-update"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            RAW_DEPLOYMENT_ISVC_CONFIG,
        ),
        pytest.param(
            {"name": "test-raw-multi-update"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                **RAW_DEPLOYMENT_ISVC_CONFIG,
                "min-replicas": 4,
            },
        ),
    ],
    indirect=True,
)
class TestRawISVCEnvVarsUpdates:
    def test_raw_with_isvc_env_vars(self, ovms_kserve_inference_service):
        """Test adding environment variables to the inference service"""
        verify_env_vars_in_isvc_pods(isvc=ovms_kserve_inference_service, env_vars=ISVC_ENV_VARS, vars_exist=True)

    def test_raw_remove_isvc_env_vars(self, removed_isvc_env_vars):
        """Test removing environment variables from the inference service"""
        verify_env_vars_in_isvc_pods(isvc=removed_isvc_env_vars, env_vars=ISVC_ENV_VARS, vars_exist=False)
