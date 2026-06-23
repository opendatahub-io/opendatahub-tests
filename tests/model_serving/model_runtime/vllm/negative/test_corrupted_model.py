"""vLLM deployment with corrupted/missing model in S3."""

import pytest
from ocp_resources.inference_service import InferenceService
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG
from tests.model_serving.model_runtime.vllm.negative.utils import get_isvc_condition_messages
from utilities.constants import Timeout

CORRUPTED_MODEL_PATH = "vllm/non-existent-model-path"

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.tier3
@pytest.mark.negative
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, negative_serving_runtime, negative_isvc_no_wait",
    [
        pytest.param(
            {"name": "vllm-neg-corrupt"},
            {"model-dir": CORRUPTED_MODEL_PATH},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                "name": "vllm-corrupt-model",
                **BASE_RAW_DEPLOYMENT_CONFIG,
            },
            id="vllm-corrupted-model-negative",
        ),
    ],
    indirect=True,
)
class TestCorruptedModelDeployment:
    """Validate ISVC failure behavior when S3 model path is invalid."""

    def test_isvc_fails_with_storage_error(
        self,
        negative_isvc_no_wait: InferenceService,
    ) -> None:
        """Verify ISVC reports a storage error for a non-existent S3 model path."""
        for sample in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_5MIN,
            sleep=10,
            func=get_isvc_condition_messages,
            isvc=negative_isvc_no_wait,
        ):
            if sample:
                messages_lower = " ".join(sample).lower()
                if any(
                    keyword in messages_lower
                    for keyword in ("storage", "download", "not found", "nosuchkey", "error", "fail")
                ):
                    return

        pytest.fail(
            f"ISVC {negative_isvc_no_wait.name} did not report a storage error within timeout. "
            f"Conditions: {get_isvc_condition_messages(isvc=negative_isvc_no_wait)}"
        )
