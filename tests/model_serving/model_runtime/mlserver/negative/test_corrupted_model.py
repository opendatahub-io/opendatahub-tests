"""TC-NEG-007: MLServer deployment with corrupted/missing model in S3.

Validates that the ISVC surfaces a descriptive error when the S3 storage URI
points to a non-existent or corrupted model directory.
"""

import pytest
from ocp_resources.inference_service import InferenceService
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_runtime.mlserver.constant import BASE_RAW_DEPLOYMENT_CONFIG
from tests.model_serving.model_runtime.mlserver.negative.utils import get_isvc_condition_messages
from tests.model_serving.model_runtime.mlserver.utils import get_model_namespace_dict
from utilities.constants import Timeout

CORRUPTED_MODEL_PATH = "mlserver/non-existent-model-path"

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.tier3
@pytest.mark.negative
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, negative_mlserver_serving_runtime, negative_mlserver_isvc_no_wait",
    [
        pytest.param(
            get_model_namespace_dict(model_format_name="mlserver-neg-corrupt"),
            {"model-dir": CORRUPTED_MODEL_PATH},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                "name": "mlserver-corrupt-model",
                **BASE_RAW_DEPLOYMENT_CONFIG,
            },
            id="mlserver-corrupted-model-negative",
        ),
    ],
    indirect=[
        "model_namespace",
        "s3_models_storage_uri",
        "negative_mlserver_serving_runtime",
        "negative_mlserver_isvc_no_wait",
    ],
)
class TestCorruptedModelDeployment:
    def test_isvc_fails_with_storage_error(
        self,
        negative_mlserver_isvc_no_wait: InferenceService,
    ) -> None:
        """Given an MLServer ISVC pointing to a non-existent S3 model path,
        When the ISVC is created,
        Then it should not become Ready and should report a storage-related error.
        """
        for sample in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_5MIN,
            sleep=10,
            func=get_isvc_condition_messages,
            isvc=negative_mlserver_isvc_no_wait,
        ):
            if sample:
                messages_lower = " ".join(sample).lower()
                if any(
                    keyword in messages_lower
                    for keyword in ("storage", "download", "not found", "nosuchkey", "error", "fail")
                ):
                    return

        pytest.fail(
            f"ISVC {negative_mlserver_isvc_no_wait.name} did not report a storage error within timeout. "
            f"Conditions: {get_isvc_condition_messages(isvc=negative_mlserver_isvc_no_wait)}"
        )
