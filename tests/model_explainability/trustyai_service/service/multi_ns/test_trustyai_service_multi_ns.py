
from typing import Generator, Any, Optional, List, Dict

import pytest
import logging
from ocp_resources.namespace import Namespace
from ocp_resources.trustyai_service import TrustyAIService

from tests.model_explainability.trustyai_service.constants import (
    DRIFT_BASE_DATA_PATH,
    TRUSTYAI_DB_MIGRATION_PATCH,
)
from tests.model_explainability.trustyai_service.trustyai_service_utils import (
    send_inferences_and_verify_trustyai_service_registered,
    verify_upload_data_to_trustyai_service,
    TrustyAIServiceMetrics,
    verify_trustyai_service_metric_scheduling_request,
    verify_trustyai_service_metric_delete_request,
)
from tests.model_explainability.trustyai_service.utils import (
    validate_trustyai_service_db_conn_failure,
    validate_trustyai_service_images,
)
from tests.model_explainability.trustyai_service.service.utils import (
    wait_for_trustyai_db_migration_complete_log,
    patch_trustyai_service_cr,
)
from utilities.constants import MinIo
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG
from ocp_resources.inference_service import InferenceService
logger = logging.getLogger(__name__)


@pytest.mark.usefixtures("minio_pod")
@pytest.mark.parametrize(
    "model_namespaces, minio_pod, minio_data_connection_multi_ns",
    [
        pytest.param(
            [
                {"name": "test-trustyaiservice-multins-1"},
                {"name": "test-trustyaiservice-multins-2"},
            ],
            MinIo.PodConfig.MODEL_MESH_MINIO_CONFIG,
            [
                {"bucket": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS},
                {"bucket": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS},
            ]
        ),
    ],
    indirect=True,
)
class TestTrustyAIServiceMultipleNS:
    """Verifies TrustyAIService operations across multiple namespaces."""

    @pytest.mark.parametrize("ns_index", [0, 1])
    def test_drift_send_inference_and_verify_trustyai_service_multiple_ns(
        self,
        ns_index: int,
        admin_client,
        current_client_token,
        trustyai_service_with_pvc_storage_multi_ns: List[TrustyAIService],
        gaussian_credit_model_multi_ns: List[InferenceService],
        isvc_getter_token_multi_ns: List[str]):
        trustyaiservice = trustyai_service_with_pvc_storage_multi_ns[ns_index]
        inference_model = gaussian_credit_model_multi_ns[ns_index]
        inference_token = isvc_getter_token_multi_ns[ns_index]

        logger.info(f"Running drift test for namespace index {ns_index} (namespace: {trustyaiservice.namespace})")

        send_inferences_and_verify_trustyai_service_registered(
            client=admin_client,
            token=current_client_token,
            data_path=f"{DRIFT_BASE_DATA_PATH}/data_batches",
            trustyai_service=trustyaiservice,
            inference_service=inference_model,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_token=inference_token,
        )
