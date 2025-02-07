import time

import pytest


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-fairness"},
        )
    ],
    indirect=True,
)
class TestFairnessMetrics:
    def test_dummy(self, admin_client, model_namespace, trustyai_service_with_pvc_storage, onnx_loan_model):
        time.sleep(1200)  # noqa: FCN001
