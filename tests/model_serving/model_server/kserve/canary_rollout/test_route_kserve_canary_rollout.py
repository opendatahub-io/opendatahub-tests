"""TC-ROUTE-001: alternateBackends traffic split at 10/90 ratio."""

import pytest
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.canary_rollout.constants import DEFAULT_CANARY_TRAFFIC_PERCENT
from tests.model_serving.model_server.kserve.canary_rollout.utils import (
    assert_canary_traffic_by_status_codes,
    assert_route_traffic_weights,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.usefixtures("canary_sklearn_inference_service"),
]


class TestCanaryRolloutRoute:
    """Validate OpenShift Route alternateBackends traffic splitting."""

    def test_tc_route_001_alternate_backends_split_traffic(
        self,
        canary_sklearn_inference_service: InferenceService,
    ) -> None:
        """
        Given an InferenceService with 10% canary traffic configured
        When inference requests are sent through the exposed Route
        Then Route weights are 90/10 and HTTP status fingerprint matches within tolerance
        """
        stable_weight = 100 - DEFAULT_CANARY_TRAFFIC_PERCENT
        assert_route_traffic_weights(
            isvc=canary_sklearn_inference_service,
            stable_weight=stable_weight,
            canary_weight=DEFAULT_CANARY_TRAFFIC_PERCENT,
        )

        assert_canary_traffic_by_status_codes(
            isvc=canary_sklearn_inference_service,
            expected_percent=DEFAULT_CANARY_TRAFFIC_PERCENT,
        )
