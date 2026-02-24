import pytest

from tests.model_serving.model_server.kserve.inference_service_configuration.constants import (
    ORIGINAL_PULL_SECRET,
    UPDATED_PULL_SECRET,
)
from tests.model_serving.model_server.kserve.inference_service_configuration.utils import verify_pull_secret
from utilities.constants import ModelName, ModelStorage, RuntimeTemplates


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, model_car_raw_inference_service_with_pull_secret",
    [
        pytest.param(
            {"name": "test-pull-secret-modelcar"},
            {
                "name": f"{ModelName.MNIST}-runtime",
                "template-name": RuntimeTemplates.OVMS_KSERVE,
                "multi-model": False,
            },
            {"storage-uri": ModelStorage.OCI.MNIST_8_1},
        )
    ],
    indirect=True,
)
class TestISVCPullSecretUpdate:
    @pytest.mark.smoke
    def test_initial_pull_secret_set(self, model_car_raw_inference_service_with_pull_secret):
        """Ensure initial pull secret is correctly set in the pod"""
        verify_pull_secret(
            isvc=model_car_raw_inference_service_with_pull_secret, pull_secret=ORIGINAL_PULL_SECRET, secret_exists=True
        )

    def test_update_pull_secret(self, updated_isvc_pull_secret):
        """Update the pull secret and verify it is reflected in the new pod"""
        verify_pull_secret(isvc=updated_isvc_pull_secret, pull_secret=UPDATED_PULL_SECRET, secret_exists=True)

    def test_remove_pull_secret(self, updated_isvc_remove_pull_secret):
        verify_pull_secret(isvc=updated_isvc_remove_pull_secret, pull_secret=UPDATED_PULL_SECRET, secret_exists=False)
