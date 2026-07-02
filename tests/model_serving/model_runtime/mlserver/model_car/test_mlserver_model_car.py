"""
Test module for MLServer model car (OCI image) deployment.

This module validates MLServer inference using model car OCI images
for sklearn, xgboost, and lightgbm formats.
"""

from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.mlserver.constant import MODEL_CONFIGS
from tests.model_serving.model_runtime.mlserver.utils import (
    get_deployment_config_dict,
    get_model_namespace_dict,
    get_model_storage_uri_dict,
    get_test_case_id,
    validate_inference_request,
)
from utilities.constants import ModelFormat, Protocols
from utilities.infra import get_pods_by_isvc_label


@pytest.mark.parametrize(
    (
        "model_namespace",
        "mlserver_model_car_inference_service",
        "mlserver_serving_runtime",
    ),
    [
        pytest.param(
            get_model_namespace_dict(model_format_name=ModelFormat.SKLEARN, modelcar=True),
            {
                **get_model_storage_uri_dict(model_format_name=ModelFormat.SKLEARN, modelcar=True),
                **get_deployment_config_dict(model_format_name=ModelFormat.SKLEARN),
            },
            get_deployment_config_dict(model_format_name=ModelFormat.SKLEARN),
            id=get_test_case_id(model_format_name=ModelFormat.SKLEARN, modelcar=True),
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            get_model_namespace_dict(model_format_name=ModelFormat.XGBOOST, modelcar=True),
            {
                **get_model_storage_uri_dict(model_format_name=ModelFormat.XGBOOST, modelcar=True),
                **get_deployment_config_dict(model_format_name=ModelFormat.XGBOOST),
            },
            get_deployment_config_dict(model_format_name=ModelFormat.XGBOOST),
            id=get_test_case_id(model_format_name=ModelFormat.XGBOOST, modelcar=True),
            marks=pytest.mark.tier1,
        ),
        pytest.param(
            get_model_namespace_dict(model_format_name=ModelFormat.LIGHTGBM, modelcar=True),
            {
                **get_model_storage_uri_dict(model_format_name=ModelFormat.LIGHTGBM, modelcar=True),
                **get_deployment_config_dict(model_format_name=ModelFormat.LIGHTGBM),
            },
            get_deployment_config_dict(model_format_name=ModelFormat.LIGHTGBM),
            id=get_test_case_id(model_format_name=ModelFormat.LIGHTGBM, modelcar=True),
            marks=pytest.mark.tier1,
        ),
        pytest.param(
            {"name": f"{ModelFormat.LIGHTGBM}-model-car-text-type"},
            {
                **get_model_storage_uri_dict(
                    model_format_name=ModelFormat.LIGHTGBM,
                    modelcar=True,
                    env_variables=[{"name": "MLSERVER_MODEL_URI", "value": "/mnt/models/model.txt"}],
                ),
                **get_deployment_config_dict(model_format_name=ModelFormat.LIGHTGBM),
            },
            get_deployment_config_dict(model_format_name=ModelFormat.LIGHTGBM),
            id=get_test_case_id(model_format_name=ModelFormat.LIGHTGBM, modelcar=True) + "_text_type",
            marks=pytest.mark.tier1,
        ),
        pytest.param(
            get_model_namespace_dict(model_format_name=ModelFormat.ONNX, modelcar=True),
            {
                **get_model_storage_uri_dict(model_format_name=ModelFormat.ONNX, modelcar=True),
                **get_deployment_config_dict(model_format_name=ModelFormat.ONNX),
            },
            get_deployment_config_dict(model_format_name=ModelFormat.ONNX),
            id=get_test_case_id(model_format_name=ModelFormat.ONNX, modelcar=True),
            marks=pytest.mark.tier1,
        ),
    ],
    indirect=[
        "model_namespace",
        "mlserver_model_car_inference_service",
        "mlserver_serving_runtime",
    ],
)
class TestMLServerModelCar:
    """
    Test class for MLServer model car (OCI image) inference.

    Validates inference functionality using OCI images for sklearn,
    xgboost, and lightgbm model formats.
    """

    def test_mlserver_model_car_inference(
        self,
        mlserver_model_car_inference_service: InferenceService,
        mlserver_response_snapshot: Any,
    ) -> None:
        """
        Test model inference using MLServer model car with OCI images.

        Validates that MLServer can load models from OCI images and
        perform inference using REST protocol.

        Args:
            mlserver_model_car_inference_service: Deployed inference service.
            mlserver_response_snapshot: Expected response for validation.
        """
        # Extract model format from InferenceService spec
        model_format = mlserver_model_car_inference_service.instance.spec.predictor.model.modelFormat.name

        if model_format not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model format: {model_format}")

        model_format_config = MODEL_CONFIGS[model_format]

        # Get pod directly from inference service (following kserve model_car pattern)
        pods = get_pods_by_isvc_label(
            client=mlserver_model_car_inference_service.client,
            isvc=mlserver_model_car_inference_service,
        )
        if not pods:
            raise RuntimeError(f"No pods found for InferenceService {mlserver_model_car_inference_service.name}")
        pod = pods[0]

        validate_inference_request(
            pod_name=pod.name,
            isvc=mlserver_model_car_inference_service,
            response_snapshot=mlserver_response_snapshot,
            input_query=model_format_config["rest_query"],
            model_version="",
            model_output_type=model_format_config["output_type"],
            protocol=Protocols.REST,
        )
