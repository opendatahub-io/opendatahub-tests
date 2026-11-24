"""Local LIME and SHAP explainer contract tests against a deployed model.

Scope and limits:

- Enablement is test-side. The explainers are off by default and the
  TrustyAIService CR exposes no field for the transport allowlist, so the
  fixture patches the generated Deployment. These tests prove service behaviour
  when configured; they do not prove that an administrator can enable the
  feature through the operator.
- Model access is the narrow case: unauthenticated, plain HTTP, cluster
  internal. Auth tokens, client certificates, and custom CA bundles are covered
  by the provider unit tests, not here.
- Every explanation requests all stored features, so attributions are asserted
  as an exact feature set rather than a subset.
- ``prediction_output`` is compared against an independent KServe V2 call to the
  deployed model for the same stored row.
"""

import math
from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.trustyai_service import TrustyAIService

from tests.ai_safety.trustyai_service.constants import (
    EXPLAINABILITY_N_TRAINING_ROWS,
    GAUSSIAN_CREDIT_MODEL_FEATURE_COUNT,
    GAUSSIAN_CREDIT_MODEL_INPUT_NAME,
    LOCAL_EXPLAINER_TIMEOUT,
)
from tests.ai_safety.trustyai_service.explainability.utils import (
    assert_common_explanation_response,
    assert_finite_attributions,
    assert_finite_number,
    assert_matches_deployed_model,
    build_local_explanation_payload,
)
from tests.ai_safety.trustyai_service.trustyai_service_utils import TrustyAIServiceClient

LOCAL_EXPLAINER_FEATURE_NAMES: set[str] = {
    f"{GAUSSIAN_CREDIT_MODEL_INPUT_NAME}-{index}" for index in range(GAUSSIAN_CREDIT_MODEL_FEATURE_COUNT)
}
LOCAL_EXPLAINER_HTTP_TIMEOUT: int = LOCAL_EXPLAINER_TIMEOUT + 60
UNKNOWN_PREDICTION_ID: str = "missing-prediction-id"
CREDENTIAL_BEARING_MODEL_URL: str = "https://user:secret@model.example"  # pragma: allowlist secret


def _client(
    admin_client: DynamicClient,
    current_client_token: str,
    trustyai_service: TrustyAIService,
) -> TrustyAIServiceClient:
    """Create an authenticated TrustyAI client for one explanation request."""
    return TrustyAIServiceClient(
        token=current_client_token,
        service=trustyai_service,
        client=admin_client,
    )


def _lime_explainer_config() -> dict[str, Any]:
    """Return bounded LIME options without confidence bootstrap work."""
    return {
        "num_samples": 100,
        "num_features": GAUSSIAN_CREDIT_MODEL_FEATURE_COUNT,
        "n_training_rows": EXPLAINABILITY_N_TRAINING_ROWS,
        "timeout": LOCAL_EXPLAINER_TIMEOUT,
        "seed": 123,
    }


def _shap_explainer_config() -> dict[str, Any]:
    """Return bounded KernelSHAP options with exact identity-link additivity."""
    return {
        "n_samples": 100,
        "n_training_rows": EXPLAINABILITY_N_TRAINING_ROWS,
        "timeout": LOCAL_EXPLAINER_TIMEOUT,
        "seed": 123,
        "link": "IDENTITY",
        "regularizer": 0,
    }


def _request_lime_explanation(
    admin_client: DynamicClient,
    current_client_token: str,
    trustyai_service: TrustyAIService,
    model_name: str,
    model_url: str,
    prediction_id: str,
    expected_prediction: float,
) -> dict[str, Any]:
    """Request one LIME explanation and assert its regression contract.

    Returns:
        dict[str, Any]: Parsed LIME response.
    """
    response = _client(
        admin_client=admin_client,
        current_client_token=current_client_token,
        trustyai_service=trustyai_service,
    ).request_local_explanation(
        algorithm="lime",
        payload=build_local_explanation_payload(
            model_url=model_url,
            model_name=model_name,
            prediction_id=prediction_id,
            explainer_config=_lime_explainer_config(),
        ),
        timeout=LOCAL_EXPLAINER_HTTP_TIMEOUT,
    )

    assert response.status_code == HTTPStatus.OK, response.text
    response_body: dict[str, Any] = response.json()
    assert_common_explanation_response(
        response_body=response_body,
        model_name=model_name,
        prediction_id=prediction_id,
    )
    assert_finite_attributions(
        response_body=response_body,
        expected_feature_names=LOCAL_EXPLAINER_FEATURE_NAMES,
    )
    for field_name in ("score", "local_prediction", "intercept", "prediction_output"):
        assert_finite_number(value=response_body[field_name])
    assert_matches_deployed_model(response_body=response_body, expected_prediction=expected_prediction)
    return response_body


def _request_shap_explanation(
    admin_client: DynamicClient,
    current_client_token: str,
    trustyai_service: TrustyAIService,
    model_name: str,
    model_url: str,
    prediction_id: str,
    expected_prediction: float,
) -> dict[str, Any]:
    """Request one SHAP explanation and assert its identity-link regression contract.

    Returns:
        dict[str, Any]: Parsed SHAP response.
    """
    response = _client(
        admin_client=admin_client,
        current_client_token=current_client_token,
        trustyai_service=trustyai_service,
    ).request_local_explanation(
        algorithm="shap",
        payload=build_local_explanation_payload(
            model_url=model_url,
            model_name=model_name,
            prediction_id=prediction_id,
            explainer_config=_shap_explainer_config(),
        ),
        timeout=LOCAL_EXPLAINER_HTTP_TIMEOUT,
    )

    assert response.status_code == HTTPStatus.OK, response.text
    response_body: dict[str, Any] = response.json()
    assert_common_explanation_response(
        response_body=response_body,
        model_name=model_name,
        prediction_id=prediction_id,
    )
    assert_finite_attributions(
        response_body=response_body,
        expected_feature_names=LOCAL_EXPLAINER_FEATURE_NAMES,
    )
    prediction_output = response_body["prediction_output"]
    shap_base_value = response_body["shap_base_value"]
    linked_prediction_output = response_body["linked_prediction_output"]
    for value in (prediction_output, shap_base_value, linked_prediction_output):
        assert_finite_number(value=value)

    attribution_sum = sum(float(attribution["importance"]) for attribution in response_body["attributions"])
    assert math.isclose(
        shap_base_value + attribution_sum,
        linked_prediction_output,
        rel_tol=1e-4,
        abs_tol=1e-3,
    )
    assert math.isclose(
        prediction_output,
        linked_prediction_output,
        rel_tol=1e-4,
        abs_tol=1e-3,
    )
    assert_matches_deployed_model(response_body=response_body, expected_prediction=expected_prediction)
    return response_body


@pytest.mark.parametrize(
    "model_namespace, trustyai_service",
    [
        pytest.param(
            {"name": "test-local-explainers-pvc"},
            {"storage": "pvc"},
            id="test_pvc_storage",
        ),
    ],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.tier2
@pytest.mark.rawdeployment
class TestLocalExplainersPvc:
    """Verify local explanation contracts against PVC-backed stored data."""

    def test_local_lime_returns_regression_attributions(
        self,
        admin_client: DynamicClient,
        current_client_token: str,
        explainability_trustyai_service: TrustyAIService,
        explainability_gaussian_credit_model: InferenceService,
        explainability_model_url: str,
        explainability_prediction_id: str,
        explainability_model_prediction: float,
    ) -> None:
        """Given stored target data, when LIME is requested, then finite attributions are returned."""
        _request_lime_explanation(
            admin_client=admin_client,
            current_client_token=current_client_token,
            trustyai_service=explainability_trustyai_service,
            model_name=explainability_gaussian_credit_model.name,
            model_url=explainability_model_url,
            prediction_id=explainability_prediction_id,
            expected_prediction=explainability_model_prediction,
        )

    def test_local_shap_preserves_identity_additivity(
        self,
        admin_client: DynamicClient,
        current_client_token: str,
        explainability_trustyai_service: TrustyAIService,
        explainability_gaussian_credit_model: InferenceService,
        explainability_model_url: str,
        explainability_prediction_id: str,
        explainability_model_prediction: float,
    ) -> None:
        """Given stored target data, when SHAP is requested, then identity-link additivity holds."""
        _request_shap_explanation(
            admin_client=admin_client,
            current_client_token=current_client_token,
            trustyai_service=explainability_trustyai_service,
            model_name=explainability_gaussian_credit_model.name,
            model_url=explainability_model_url,
            prediction_id=explainability_prediction_id,
            expected_prediction=explainability_model_prediction,
        )

    @pytest.mark.parametrize("algorithm", ["lime", "shap"])
    def test_local_explanation_unknown_prediction_returns_data_missing(
        self,
        admin_client: DynamicClient,
        current_client_token: str,
        explainability_trustyai_service: TrustyAIService,
        explainability_gaussian_credit_model: InferenceService,
        explainability_model_url: str,
        explainability_prediction_id: str,
        explainability_model_prediction: float,
        algorithm: str,
    ) -> None:
        """Given stored data and an unknown ID, when explained, then the mapped error is data_missing.

        The upload fixture keeps stored data present, so `data_missing` can only
        mean the prediction ID is unknown, not that the model has no data.
        """
        assert explainability_prediction_id != UNKNOWN_PREDICTION_ID
        response = _client(
            admin_client=admin_client,
            current_client_token=current_client_token,
            trustyai_service=explainability_trustyai_service,
        ).request_local_explanation(
            algorithm=algorithm,
            payload=build_local_explanation_payload(
                model_url=explainability_model_url,
                model_name=explainability_gaussian_credit_model.name,
                prediction_id=UNKNOWN_PREDICTION_ID,
                explainer_config=_lime_explainer_config() if algorithm == "lime" else _shap_explainer_config(),
            ),
            timeout=LOCAL_EXPLAINER_HTTP_TIMEOUT,
        )

        assert response.status_code == HTTPStatus.NOT_FOUND, response.text
        error_detail = response.json().get("detail")
        assert isinstance(error_detail, dict), response.text
        assert error_detail.get("code") == "data_missing", response.text

    @pytest.mark.parametrize("algorithm", ["lime", "shap"])
    def test_local_explanation_does_not_reflect_model_credentials(
        self,
        admin_client: DynamicClient,
        current_client_token: str,
        explainability_trustyai_service: TrustyAIService,
        explainability_gaussian_credit_model: InferenceService,
        algorithm: str,
    ) -> None:
        """Given a credential-bearing model URL, when explained, then the 422 response is sanitized."""
        payload = build_local_explanation_payload(
            model_url=CREDENTIAL_BEARING_MODEL_URL,
            model_name=explainability_gaussian_credit_model.name,
            prediction_id=UNKNOWN_PREDICTION_ID,
            explainer_config=_lime_explainer_config() if algorithm == "lime" else _shap_explainer_config(),
        )
        response = _client(
            admin_client=admin_client,
            current_client_token=current_client_token,
            trustyai_service=explainability_trustyai_service,
        ).request_local_explanation(
            algorithm=algorithm,
            payload=payload,
            timeout=LOCAL_EXPLAINER_HTTP_TIMEOUT,
        )

        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, response.text
        assert "secret" not in response.text.lower()
        validation_errors = response.json().get("detail")
        assert isinstance(validation_errors, list), response.text
        for error in validation_errors:
            assert "input" not in error
            assert "ctx" not in error


@pytest.mark.parametrize(
    "model_namespace, trustyai_service",
    [
        pytest.param(
            {"name": "test-local-explainers-db"},
            {"storage": "db"},
            id="test_db_storage",
        ),
    ],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.tier2
@pytest.mark.rawdeployment
class TestLocalExplainersDb:
    """Verify the local explanation storage read path against database-backed data."""

    def test_local_lime_returns_regression_attributions(
        self,
        admin_client: DynamicClient,
        current_client_token: str,
        explainability_trustyai_service: TrustyAIService,
        explainability_gaussian_credit_model: InferenceService,
        explainability_model_url: str,
        explainability_prediction_id: str,
        explainability_model_prediction: float,
    ) -> None:
        """Given database-backed target data, when LIME is requested, then finite attributions are returned."""
        _request_lime_explanation(
            admin_client=admin_client,
            current_client_token=current_client_token,
            trustyai_service=explainability_trustyai_service,
            model_name=explainability_gaussian_credit_model.name,
            model_url=explainability_model_url,
            prediction_id=explainability_prediction_id,
            expected_prediction=explainability_model_prediction,
        )

    def test_local_shap_preserves_identity_additivity(
        self,
        admin_client: DynamicClient,
        current_client_token: str,
        explainability_trustyai_service: TrustyAIService,
        explainability_gaussian_credit_model: InferenceService,
        explainability_model_url: str,
        explainability_prediction_id: str,
        explainability_model_prediction: float,
    ) -> None:
        """Given database-backed target data, when SHAP is requested, then identity-link additivity holds."""
        _request_shap_explanation(
            admin_client=admin_client,
            current_client_token=current_client_token,
            trustyai_service=explainability_trustyai_service,
            model_name=explainability_gaussian_credit_model.name,
            model_url=explainability_model_url,
            prediction_id=explainability_prediction_id,
            expected_prediction=explainability_model_prediction,
        )
