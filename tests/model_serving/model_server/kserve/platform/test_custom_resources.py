import pytest
import structlog
from ocp_resources.inference_service import InferenceService
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    RuntimeTemplates,
)

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [pytest.mark.tier3, pytest.mark.slow, pytest.mark.usefixtures("valid_aws_config")]


def wait_for_isvc_model_status(isvc: InferenceService, target_model_state: str, transition_status: str) -> None:
    LOGGER.info(
        f"Wait for {isvc.name} target model state {target_model_state} and transition status {transition_status}."
    )

    samples = TimeoutSampler(wait_timeout=60 * 25, sleep=5, func=lambda: isvc.instance.status.modelStatus)

    sample = None
    try:
        for sample in samples:
            if sample.states.targetModelState == target_model_state and sample.transitionStatus == transition_status:
                return

    except TimeoutExpiredError:
        LOGGER.error(f"Status of {isvc.name} is {sample}")
        raise


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, invalid_s3_models_inference_service",
    [
        pytest.param(
            {"name": "test-non-existing-models-path"},
            {
                "name": ModelFormat.ONNX,
                "template-name": RuntimeTemplates.OVMS_KSERVE,
                "multi-model": False,
            },
            {
                "name": "missing-path",
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
            },
        )
    ],
    indirect=True,
)
class TestInferenceServiceCustomResources:
    """Validate InferenceService status transitions when the model storage path is invalid and then corrected.

    Steps:
        1. Deploy an ISVC with a non-existing S3 model path.
        2. Verify the model status transitions to FailedToLoad / BlockedByFailedLoad.
        3. Update the ISVC with a valid S3 model path.
        4. Verify the model status transitions to Loaded / UpToDate.
    """

    @pytest.mark.dependency(name="test_isvc_with_invalid_models_s3_path")
    def test_isvc_with_invalid_models_s3_path(self, invalid_s3_models_inference_service):
        """Test ISVC status with invalid models storage path"""
        wait_for_isvc_model_status(
            isvc=invalid_s3_models_inference_service,
            target_model_state="FailedToLoad",
            transition_status="BlockedByFailedLoad",
        )

    @pytest.mark.parametrize(
        "s3_models_storage_uri",
        [pytest.param({"model-dir": "test-dir"})],
        indirect=True,
    )
    @pytest.mark.dependency(depends=["test_isvc_with_invalid_models_s3_path"])
    def test_isvc_with_updated_valid_models_s3_path(self, s3_models_storage_uri, updated_s3_models_inference_service):
        """Test inference status after updating the model storage path"""
        wait_for_isvc_model_status(
            isvc=updated_s3_models_inference_service,
            target_model_state="Loaded",
            transition_status="UpToDate",
        )
