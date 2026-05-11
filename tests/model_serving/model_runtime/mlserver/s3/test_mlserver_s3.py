"""
Test module for model deployment using the MLServer runtime.

This module contains parameterized tests to validate model inference
across REST protocol and RawDeployment mode.
"""

from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.mlserver.constant import (
    MODEL_CONFIGS,
)
from tests.model_serving.model_runtime.mlserver.utils import (
    get_deployment_config_dict,
    get_model_namespace_dict,
    get_model_storage_uri_dict,
    get_test_case_id,
    validate_inference_request,
)
from utilities.constants import KServeDeploymentType, ModelFormat, Protocols

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.parametrize(
    (
        "model_namespace",
        "mlserver_inference_service",
        "s3_models_storage_uri",
        "mlserver_serving_runtime",
        "model_format",
    ),
    [
        pytest.param(
            get_model_namespace_dict(model_format_name=ModelFormat.LIGHTGBM),
            get_deployment_config_dict(
                model_format_name=ModelFormat.LIGHTGBM, deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT
            ),
            get_model_storage_uri_dict(model_format_name=ModelFormat.LIGHTGBM),
            get_deployment_config_dict(
                model_format_name=ModelFormat.LIGHTGBM, deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT
            ),
            ModelFormat.LIGHTGBM,
            id=get_test_case_id(
                model_format_name=ModelFormat.LIGHTGBM,
                deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            ),
            marks=pytest.mark.tier1,
        ),
        pytest.param(
            get_model_namespace_dict(model_format_name=ModelFormat.ONNX),
            get_deployment_config_dict(
                model_format_name=ModelFormat.ONNX, deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT
            ),
            get_model_storage_uri_dict(model_format_name=ModelFormat.ONNX),
            get_deployment_config_dict(
                model_format_name=ModelFormat.ONNX, deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT
            ),
            ModelFormat.ONNX,
            id=get_test_case_id(
                model_format_name=ModelFormat.ONNX,
                deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            ),
            marks=pytest.mark.tier1,
        ),
        pytest.param(
            get_model_namespace_dict(model_format_name=ModelFormat.SKLEARN),
            get_deployment_config_dict(
                model_format_name=ModelFormat.SKLEARN, deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT
            ),
            get_model_storage_uri_dict(model_format_name=ModelFormat.SKLEARN),
            get_deployment_config_dict(
                model_format_name=ModelFormat.SKLEARN, deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT
            ),
            ModelFormat.SKLEARN,
            id=get_test_case_id(
                model_format_name=ModelFormat.SKLEARN,
                deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            ),
            marks=pytest.mark.tier1,
        ),
        pytest.param(
            get_model_namespace_dict(model_format_name=ModelFormat.XGBOOST),
            get_deployment_config_dict(
                model_format_name=ModelFormat.XGBOOST, deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT
            ),
            get_model_storage_uri_dict(model_format_name=ModelFormat.XGBOOST),
            get_deployment_config_dict(
                model_format_name=ModelFormat.XGBOOST, deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT
            ),
            ModelFormat.XGBOOST,
            id=get_test_case_id(
                model_format_name=ModelFormat.XGBOOST,
                deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            ),
            marks=pytest.mark.tier1,
        ),
    ],
    indirect=[
        "model_namespace",
        "mlserver_inference_service",
        "s3_models_storage_uri",
        "mlserver_serving_runtime",
    ],
)
class TestMLServerModels:
    """
    Test class for model inference using the MLServer runtime.

    This class validates inference functionality across multiple configurations:
    - Protocols: REST
    - Deployment modes: Raw
    - Response validation against predefined snapshots
    """

    def test_mlserver_model_inference(
        self,
        mlserver_inference_service: InferenceService,
        mlserver_pod_resource: Pod,
        mlserver_response_snapshot: Any,
        model_format: str,
    ) -> None:
        """
        Test model inference using MLServer with REST protocol and RawDeployment mode.

        This test sends inference requests using REST protocol and compares
        the actual response with the expected snapshot for validation.

        Args:
            mlserver_inference_service (InferenceService): The deployed inference service instance.
            mlserver_pod_resource (Pod): The Kubernetes pod running the MLServer.
            mlserver_response_snapshot (Any): The expected model response for snapshot-based validation.
            model_format (str): Identifier for the model framework (e.g., "sklearn").
        """
        if model_format not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model format: {model_format}")

        model_format_config = MODEL_CONFIGS[model_format]

        validate_inference_request(
            pod_name=mlserver_pod_resource.name,
            isvc=mlserver_inference_service,
            response_snapshot=mlserver_response_snapshot,
            input_query=model_format_config["rest_query"],
            model_version=model_format_config["model_version"],
            model_output_type=model_format_config["output_type"],
            protocol=Protocols.REST,
        )
