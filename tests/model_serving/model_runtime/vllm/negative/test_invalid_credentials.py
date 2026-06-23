"""vLLM deployment with invalid S3 credentials."""

import pytest
from ocp_resources.inference_service import InferenceService
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG
from tests.model_serving.model_runtime.vllm.negative.utils import get_isvc_condition_messages
from utilities.constants import Timeout

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.tier3
@pytest.mark.negative
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, negative_serving_runtime, negative_isvc_bad_creds",
    [
        pytest.param(
            {"name": "vllm-neg-badcreds"},
            {"model-dir": "vllm/granite-7b-lab"},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                "name": "vllm-bad-creds",
                **BASE_RAW_DEPLOYMENT_CONFIG,
            },
            id="vllm-invalid-credentials-negative",
        ),
    ],
    indirect=True,
)
class TestInvalidCredentials:
    def test_isvc_fails_with_auth_error(
        self,
        negative_isvc_bad_creds: InferenceService,
    ) -> None:
        """Verify ISVC reports an authentication error when S3 credentials are invalid."""
        for sample in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_5MIN,
            sleep=10,
            func=get_isvc_condition_messages,
            isvc=negative_isvc_bad_creds,
        ):
            if sample:
                messages_lower = " ".join(sample).lower()
                if any(
                    keyword in messages_lower
                    for keyword in ("accessdenied", "forbidden", "credential", "auth", "403", "invalid", "error")
                ):
                    return

        pytest.fail(
            f"ISVC {negative_isvc_bad_creds.name} did not report an auth error within timeout. "
            f"Conditions: {get_isvc_condition_messages(isvc=negative_isvc_bad_creds)}"
        )
