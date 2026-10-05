"""TC-CRD-001: Valid canary array field accepted on RawDeployment ISVC."""

import pytest
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.canary_rollout.constants import DEFAULT_CANARY_TRAFFIC_PERCENT

pytestmark = pytest.mark.usefixtures("canary_sklearn_inference_service")


class TestCanaryRolloutCrd:
    """Validate InferenceService CRD accepts the canary array schema."""

    def test_tc_crd_001_canary_array_field_persisted(
        self,
        canary_sklearn_inference_service: InferenceService,
    ) -> None:
        """
        Given an InferenceService with RawDeployment mode and a canary array entry
        When the resource is created on the cluster
        Then the canary array is persisted with the configured traffic percentage
        """
        canary_spec = canary_sklearn_inference_service.instance.spec.get("canary") or []
        assert isinstance(canary_spec, list), "Expected spec.canary to be an array"
        assert canary_spec, "Expected at least one canary array entry"
        assert canary_spec[0]["trafficPercent"] == DEFAULT_CANARY_TRAFFIC_PERCENT
        assert canary_spec[0]["predictor"]["name"] == "canary"
