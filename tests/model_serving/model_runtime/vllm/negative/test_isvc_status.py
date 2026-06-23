"""ISVC status messages for invalid configurations."""

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
    "model_namespace, s3_models_storage_uri, negative_serving_runtime, negative_isvc_no_wait",
    [
        pytest.param(
            {"name": "vllm-neg-status"},
            {"model-dir": "vllm/granite-7b-lab"},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                "name": "vllm-bad-config",
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "runtime_argument": ["--model=/mnt/models", "--dtype=invalid_dtype"],
            },
            id="vllm-invalid-config-status",
        ),
    ],
    indirect=True,
)
class TestISVCStatusMessages:
    """Validate ISVC status conditions for invalid runtime configurations."""

    def test_isvc_reports_failure_conditions(
        self,
        negative_isvc_no_wait: InferenceService,
    ) -> None:
        """Verify ISVC reports failure conditions when created with an invalid runtime argument."""
        condition_found = False
        for sample in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_5MIN,
            sleep=10,
            func=get_isvc_condition_messages,
            isvc=negative_isvc_no_wait,
        ):
            if sample:
                messages_lower = " ".join(sample).lower()
                has_error_indicators = any(
                    keyword in messages_lower
                    for keyword in ("error", "fail", "crash", "backoff", "invalid", "not ready", "false")
                )
                if has_error_indicators:
                    condition_found = True
                    break

        assert condition_found, (
            f"ISVC {negative_isvc_no_wait.name} did not report failure conditions within timeout. "
            f"Conditions: {get_isvc_condition_messages(isvc=negative_isvc_no_wait)}"
        )
