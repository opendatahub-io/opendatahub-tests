import pytest

from tests.model_serving.model_runtime.mlserver.utils import run_mlserver_inference, validate_deterministic_snapshot
from tests.model_serving.model_server.utils import (
    verify_inference_response,
)
from utilities.constants import Protocols
from utilities.inference_utils import Inference


@pytest.mark.parametrize(
    "unprivileged_model_namespace, unprivileged_s3_ovms_raw_inference_service",
    [
        pytest.param(
            {"name": "test-non-admin-raw"},
            {"model-dir": "/test-dir/"},
        )
    ],
    indirect=True,
)
@pytest.mark.smoke
@pytest.mark.rawdeployment
@pytest.mark.arch_runtime
class TestRawUnprivilegedUser:
    """Validate that a non-admin user can deploy and query a KServe raw deployment model.

    Steps:
        1. Create a namespace with unprivileged user credentials.
        2. Deploy OVMS on x86 or MLServer on ARM64 as a raw deployment using the non-admin user.
        3. Query the deployed model via REST and verify a successful inference response.
    """

    def test_non_admin_deploy_raw_and_query_model(
        self,
        unprivileged_s3_ovms_raw_inference_service,
        auth_model_config,
    ):
        """Verify non admin can deploy a Raw model and query using REST"""
        if "request" in auth_model_config:
            response = run_mlserver_inference(
                isvc=unprivileged_s3_ovms_raw_inference_service,
                input_data=auth_model_config["request"],
                model_version="",
                protocol=Protocols.REST,
            )
            validate_deterministic_snapshot(response=response)
            return

        verify_inference_response(
            inference_service=unprivileged_s3_ovms_raw_inference_service,
            inference_config=auth_model_config["inference"],
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )
