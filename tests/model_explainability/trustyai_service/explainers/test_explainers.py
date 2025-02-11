import pytest

from tests.model_explainability.trustyai_service.constants import MODEL_DATA_PATH
from tests.model_explainability.trustyai_service.explainers.utils import verify_shap_explanation
from tests.model_explainability.trustyai_service.utils import send_inference_requests_and_verify_trustyai_service
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG

BASE_DATA_PATH: str = f"{MODEL_DATA_PATH}/fairness/"


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-shap"},
        )
    ],
    indirect=True,
)
def test_explainers_shap(
    admin_client, current_client_token, model_namespace, trustyai_service_with_pvc_storage, onnx_loan_model
):
    send_inference_requests_and_verify_trustyai_service(
        client=admin_client,
        token=current_client_token,
        data_path=f"{BASE_DATA_PATH}",
        trustyai_service=trustyai_service_with_pvc_storage,
        inference_service=onnx_loan_model,
        inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
    )

    verify_shap_explanation(
        client=admin_client,
        token=current_client_token,
        trustyai_service=trustyai_service_with_pvc_storage,
        inference_service=onnx_loan_model,
    )
