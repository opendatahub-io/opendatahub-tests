"""
Test module for MLServer inference with PVC (Persistent Volume Claim) storage.

This module validates that MLServer can successfully load models from PVC
storage and perform inference, as an alternative to S3 storage.
"""

from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.mlserver.constant import MODEL_CONFIGS
from tests.model_serving.model_runtime.mlserver.utils import validate_inference_request
from utilities.constants import KServeDeploymentType, ModelFormat, Protocols
from utilities.infra import get_pods_by_isvc_label

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.parametrize(
    (
        "model_namespace",
        "mlserver_model_pvc",
        "mlserver_pvc_downloaded_model_data",
        "mlserver_serving_runtime",
        "mlserver_pvc_inference_service",
    ),
    [
        pytest.param(
            {"name": "sklearn-pvc"},
            {"pvc-size": "5Gi"},
            {"model-dir": "mlserver/model_repository/sklearn"},
            {"name": ModelFormat.SKLEARN, "deployment_mode": KServeDeploymentType.STANDARD},
            {"name": "sklearn", "deployment_mode": KServeDeploymentType.STANDARD},
            id="sklearn-pvc-Standard",
            marks=pytest.mark.smoke,
        ),
    ],
    indirect=True,
)
class TestMLServerPvcInference:
    """
    Test class for MLServer inference using PVC storage.

    Validates that MLServer can load sklearn models from PVC storage
    and perform successful inference using REST protocol.
    """

    def test_mlserver_pvc_sklearn_inference(
        self,
        mlserver_pvc_inference_service: InferenceService,
        mlserver_response_snapshot: Any,
    ) -> None:
        """
        Test MLServer inference with sklearn model stored on PVC.

        This test validates the complete PVC storage workflow:
        1. PVC created and model downloaded from S3 to PVC
        2. InferenceService deployed with pvc:// storage URI
        3. Model loads successfully from PVC
        4. Inference request returns valid response

        Args:
            mlserver_pvc_inference_service: Deployed InferenceService with PVC storage
            mlserver_response_snapshot: Expected response for validation
        """
        # Get model configuration for sklearn
        model_format = ModelFormat.SKLEARN
        if model_format not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model format: {model_format}")

        model_format_config = MODEL_CONFIGS[model_format]

        # Get pod from InferenceService
        pods = get_pods_by_isvc_label(
            client=mlserver_pvc_inference_service.client,
            isvc=mlserver_pvc_inference_service,
        )
        if not pods:
            raise RuntimeError(f"No pods found for InferenceService {mlserver_pvc_inference_service.name}")
        pod = pods[0]

        # Validate inference using REST protocol
        validate_inference_request(
            pod_name=pod.name,
            isvc=mlserver_pvc_inference_service,
            response_snapshot=mlserver_response_snapshot,
            input_query=model_format_config["rest_query"],
            model_version=model_format_config["model_version"],
            model_output_type=model_format_config["output_type"],
            protocol=Protocols.REST,
        )
