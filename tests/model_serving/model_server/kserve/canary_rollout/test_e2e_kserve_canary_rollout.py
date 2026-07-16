"""TC-E2E-001: Complete canary rollout with traffic verification and promotion."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.canary_rollout.constants import (
    CANARY_MODEL_FORMAT,
    CANARY_STORAGE_URI,
    DEFAULT_CANARY_TRAFFIC_PERCENT,
)
from tests.model_serving.model_server.kserve.canary_rollout.utils import (
    assert_canary_traffic_by_status_codes,
    assert_route_traffic_weights,
    get_isvc_deployments,
    promote_canary_to_stable,
    wait_for_canary_ready_condition,
)
from utilities.infra import get_model_route

pytestmark = pytest.mark.usefixtures("canary_e2e_inference_service")


class TestCanaryRolloutE2E:
    """End-to-end canary rollout lifecycle with promotion."""

    def test_tc_e2e_001_canary_rollout_promotion(
        self,
        admin_client: DynamicClient,
        canary_e2e_inference_service: InferenceService,
    ) -> None:
        """
        Given a canary rollout at 10% traffic on a RawDeployment InferenceService
        When traffic is verified and the canary revision is promoted
        Then only one Deployment remains and Route alternateBackends are cleared
        """
        runtime_name = canary_e2e_inference_service.instance.spec.predictor["model"]["runtime"]
        wait_for_canary_ready_condition(canary_e2e_inference_service)

        deployments = get_isvc_deployments(
            client=admin_client,
            isvc=canary_e2e_inference_service,
            runtime_name=runtime_name,
            expected_count=2,
        )
        assert len(deployments) == 2

        assert_route_traffic_weights(
            canary_e2e_inference_service,
            stable_weight=100 - DEFAULT_CANARY_TRAFFIC_PERCENT,
            canary_weight=DEFAULT_CANARY_TRAFFIC_PERCENT,
        )

        assert_canary_traffic_by_status_codes(
            canary_e2e_inference_service,
            expected_percent=DEFAULT_CANARY_TRAFFIC_PERCENT,
            sample_size=200,
        )

        promote_canary_to_stable(
            canary_e2e_inference_service,
            promoted_storage_uri=CANARY_STORAGE_URI,
            runtime=runtime_name,
            model_format=CANARY_MODEL_FORMAT,
        )

        post_promotion_deployments = get_isvc_deployments(
            client=admin_client,
            isvc=canary_e2e_inference_service,
            runtime_name=runtime_name,
            expected_count=1,
        )
        assert len(post_promotion_deployments) == 1

        route = get_model_route(client=admin_client, isvc=canary_e2e_inference_service)
        alternate_backends = route.instance.spec.get("alternateBackends") or []
        assert not alternate_backends, "Expected Route alternateBackends to be cleared after promotion"
