"""
End-to-end tests for AutoGluon models served from S3 via KServe Standard deployment mode.

Validates tabular (V1/V2) and timeseries (V1) predictors with fuzzy response checks.
"""

from typing import cast

import pytest
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.autogluon.constant import MODEL_CONFIGS, PredictorType
from tests.model_serving.model_runtime.autogluon.utils import (
    ProtocolVersionLiteral,
    get_deployment_config_dict,
    get_model_namespace_dict,
    get_model_storage_uri_dict,
    get_test_case_id,
    validate_inference_request,
)
from utilities.constants import KServeDeploymentType

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.parametrize(
    (
        "model_namespace",
        "autogluon_inference_service",
        "s3_models_storage_uri",
        "autogluon_serving_runtime",
        "predictor_type",
    ),
    [
        pytest.param(
            get_model_namespace_dict(predictor_type=PredictorType.TABULAR_V2),
            get_deployment_config_dict(
                predictor_type=PredictorType.TABULAR_V2,
                deployment_mode=KServeDeploymentType.STANDARD,
            ),
            get_model_storage_uri_dict(predictor_type=PredictorType.TABULAR_V2),
            get_deployment_config_dict(
                predictor_type=PredictorType.TABULAR_V2,
                deployment_mode=KServeDeploymentType.STANDARD,
            ),
            PredictorType.TABULAR_V2,
            id=get_test_case_id(
                predictor_type=PredictorType.TABULAR_V2,
                deployment_mode=KServeDeploymentType.STANDARD,
            ),
            marks=pytest.mark.tier1,
        ),
        pytest.param(
            get_model_namespace_dict(predictor_type=PredictorType.TABULAR_V1),
            get_deployment_config_dict(
                predictor_type=PredictorType.TABULAR_V1,
                deployment_mode=KServeDeploymentType.STANDARD,
            ),
            get_model_storage_uri_dict(predictor_type=PredictorType.TABULAR_V1),
            get_deployment_config_dict(
                predictor_type=PredictorType.TABULAR_V1,
                deployment_mode=KServeDeploymentType.STANDARD,
            ),
            PredictorType.TABULAR_V1,
            id=get_test_case_id(
                predictor_type=PredictorType.TABULAR_V1,
                deployment_mode=KServeDeploymentType.STANDARD,
            ),
            marks=pytest.mark.tier1,
        ),
        pytest.param(
            get_model_namespace_dict(predictor_type=PredictorType.TIMESERIES_V1),
            get_deployment_config_dict(
                predictor_type=PredictorType.TIMESERIES_V1,
                deployment_mode=KServeDeploymentType.STANDARD,
            ),
            get_model_storage_uri_dict(predictor_type=PredictorType.TIMESERIES_V1),
            get_deployment_config_dict(
                predictor_type=PredictorType.TIMESERIES_V1,
                deployment_mode=KServeDeploymentType.STANDARD,
            ),
            PredictorType.TIMESERIES_V1,
            id=get_test_case_id(
                predictor_type=PredictorType.TIMESERIES_V1,
                deployment_mode=KServeDeploymentType.STANDARD,
            ),
            marks=pytest.mark.tier1,
        ),
    ],
    indirect=[
        "model_namespace",
        "autogluon_inference_service",
        "s3_models_storage_uri",
        "autogluon_serving_runtime",
    ],
)
class TestAutoGluonS3Models:
    """AutoGluon inference from S3-backed models using KServe Standard deployment mode."""

    def test_autogluon_model_inference(
        self,
        autogluon_inference_service: InferenceService,
        predictor_type: str,
    ) -> None:
        """
        Given an AutoGluon model in S3 and a Standard InferenceService,
        When an inference request is sent using the configured protocol version,
        Then the response structure and payload data are valid.
        """
        if predictor_type not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported predictor type: {predictor_type}")

        model_config = MODEL_CONFIGS[predictor_type]
        protocol_version = cast(ProtocolVersionLiteral, model_config["protocol_version"])

        validate_inference_request(
            isvc=autogluon_inference_service,
            input_payload=model_config["input_payload"],
            protocol_version=protocol_version,
            model_version=model_config["model_version"],
            model_output_type=model_config["output_type"],
        )
