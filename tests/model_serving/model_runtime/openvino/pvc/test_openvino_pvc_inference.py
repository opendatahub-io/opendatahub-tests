"""
Test module for OpenVINO PVC-based model deployments.

Validates OpenVINO inference using PVC storage for models.
"""

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.openvino.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    MODEL_CONFIGS,
    MODEL_PATH_PREFIX,
)
from tests.model_serving.model_runtime.openvino.utils import (
    get_input_query,
    validate_inference_request,
)
from utilities.constants import ModelFormat, Protocols

MODEL_STORAGE_URI_DICT = {"model-dir": f"{MODEL_PATH_PREFIX}/{ModelFormat.OPENVINO}"}

pytestmark = pytest.mark.usefixtures(
    "root_dir",
    "valid_aws_config",
)


@pytest.mark.tier1
@pytest.mark.parametrize(
    (
        "model_namespace, openvino_model_pvc, "
        "openvino_pvc_downloaded_model_data, openvino_pvc_serving_runtime, "
        "openvino_pvc_inference_service"
    ),
    [
        pytest.param(
            {"name": "openvino-pvc-ovms"},
            {"pvc-size": "10Gi"},
            MODEL_STORAGE_URI_DICT,
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 0,
                "name": "openvino-pvc-standard",
            },
            id="test_openvino_pvc_ovms_standard",
        ),
    ],
    indirect=True,
)
class TestOpenVINOPvcOvmsInference:
    """Validate OpenVINO model inference from PVC-backed storage.

    Steps:
        1. Create a PVC and download the OpenVINO model from S3 into it.
        2. Deploy an OpenVINO InferenceService using PVC storage.
        3. Run REST inference requests.
        4. Validate that inference responses contain expected content.
    """

    def test_openvino_pvc_ovms_inference(
        self,
        openvino_pvc_inference_service: InferenceService,
        openvino_pvc_pod_resource: Pod,
    ) -> None:
        """Given an OpenVINO ISVC backed by PVC storage with the OpenVINO model,
        When REST inference requests are sent,
        Then the model returns valid responses.
        """
        model_format_config = MODEL_CONFIGS[ModelFormat.OPENVINO]
        model_version = model_format_config.get("model_version", "")
        input_query = get_input_query(model_format_config=model_format_config, protocol=Protocols.REST)

        validate_inference_request(
            pod_name=openvino_pvc_pod_resource.name,
            isvc=openvino_pvc_inference_service,
            response_snapshot=None,
            input_query=input_query,
            model_version=model_version,
        )
