"""
Test module for model deployment using the MLServer runtime.

This module contains parameterized tests to validate model inference
across REST protocol and raw deployment type.
"""

from typing import Any

import pytest
from simple_logger.logger import get_logger

from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from utilities.constants import Protocols

from tests.model_serving.model_runtime.mlserver.constant import (
    MODEL_CONFIGS,
    RAW_DEPLOYMENT_TYPE,
    REST_PROTOCOL_TYPE_DICT,
    LIGHTGBM_MODEL_FORMAT_NAME,
    SKLEARN_MODEL_FORMAT_NAME,
    XGBOOST_MODEL_FORMAT_NAME,
)

from tests.model_serving.model_runtime.mlserver.utils import (
    validate_inference_request,
    get_model_storage_uri_dict,
    get_model_namespace_dict,
    get_deployment_config_dict,
    get_test_case_id,
)

LOGGER = get_logger(name=__name__)

pytestmark = pytest.mark.usefixtures("root_dir", "valid_aws_config", "mlserver_rest_serving_runtime_template")


@pytest.mark.smoke
@pytest.mark.parametrize(
    (
        "protocol",
        "model_namespace",
        "mlserver_inference_service",
        "s3_models_storage_uri",
        "mlserver_serving_runtime",
        "model_format",
    ),
    [
        pytest.param(
            REST_PROTOCOL_TYPE_DICT,
            get_model_namespace_dict(
                model_format_name=LIGHTGBM_MODEL_FORMAT_NAME,
                deployment_type=RAW_DEPLOYMENT_TYPE,
                protocol_type=Protocols.REST,
            ),
            get_deployment_config_dict(
                model_format_name=LIGHTGBM_MODEL_FORMAT_NAME, deployment_type=RAW_DEPLOYMENT_TYPE
            ),
            get_model_storage_uri_dict(model_format_name=LIGHTGBM_MODEL_FORMAT_NAME),
            get_deployment_config_dict(
                model_format_name=LIGHTGBM_MODEL_FORMAT_NAME, deployment_type=RAW_DEPLOYMENT_TYPE
            ),
            LIGHTGBM_MODEL_FORMAT_NAME,
            id=get_test_case_id(
                model_format_name=LIGHTGBM_MODEL_FORMAT_NAME,
                deployment_type=RAW_DEPLOYMENT_TYPE,
                protocol_type=Protocols.REST,
            ),
        ),
        pytest.param(
            REST_PROTOCOL_TYPE_DICT,
            get_model_namespace_dict(
                model_format_name=SKLEARN_MODEL_FORMAT_NAME,
                deployment_type=RAW_DEPLOYMENT_TYPE,
                protocol_type=Protocols.REST,
            ),
            get_deployment_config_dict(
                model_format_name=SKLEARN_MODEL_FORMAT_NAME, deployment_type=RAW_DEPLOYMENT_TYPE
            ),
            get_model_storage_uri_dict(model_format_name=SKLEARN_MODEL_FORMAT_NAME),
            get_deployment_config_dict(
                model_format_name=SKLEARN_MODEL_FORMAT_NAME, deployment_type=RAW_DEPLOYMENT_TYPE
            ),
            SKLEARN_MODEL_FORMAT_NAME,
            id=get_test_case_id(
                model_format_name=SKLEARN_MODEL_FORMAT_NAME,
                deployment_type=RAW_DEPLOYMENT_TYPE,
                protocol_type=Protocols.REST,
            ),
        ),
        pytest.param(
            REST_PROTOCOL_TYPE_DICT,
            get_model_namespace_dict(
                model_format_name=XGBOOST_MODEL_FORMAT_NAME,
                deployment_type=RAW_DEPLOYMENT_TYPE,
                protocol_type=Protocols.REST,
            ),
            get_deployment_config_dict(
                model_format_name=XGBOOST_MODEL_FORMAT_NAME, deployment_type=RAW_DEPLOYMENT_TYPE
            ),
            get_model_storage_uri_dict(model_format_name=XGBOOST_MODEL_FORMAT_NAME),
            get_deployment_config_dict(
                model_format_name=XGBOOST_MODEL_FORMAT_NAME, deployment_type=RAW_DEPLOYMENT_TYPE
            ),
            XGBOOST_MODEL_FORMAT_NAME,
            id=get_test_case_id(
                model_format_name=XGBOOST_MODEL_FORMAT_NAME,
                deployment_type=RAW_DEPLOYMENT_TYPE,
                protocol_type=Protocols.REST,
            ),
        ),
    ],
    indirect=[
        "protocol",
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
        protocol: str,
        root_dir: str,
        model_format: str,
    ) -> None:
        """
        Test model inference using MLServer across REST protocol and raw deployment type.

        This test sends inference requests using REST protocol and compares
        the actual response with the expected snapshot for validation.

        Args:
            mlserver_inference_service (InferenceService): The deployed inference service instance.
            mlserver_pod_resource (Pod): The Kubernetes pod running the MLServer.
            mlserver_response_snapshot (Any): The expected model response for snapshot-based validation.
            protocol (str): The communication protocol to use ("rest").
            root_dir (str): Path to the test root directory containing snapshots or test data.
            model_format (str): Identifier for the model framework (e.g., "sklearn").
        """

        if model_format not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model format: {model_format}")

        model_format_config = MODEL_CONFIGS[model_format]

        query_key = "rest_query"
        input_query = model_format_config[query_key]

        validate_inference_request(
            pod_name=mlserver_pod_resource.name,
            isvc=mlserver_inference_service,
            response_snapshot=mlserver_response_snapshot,
            input_query=input_query,
            model_version=model_format_config["model_version"],
            model_framework=model_format,
            model_output_type=model_format_config["output_type"],
            protocol=protocol,
            root_dir=root_dir,
        )
