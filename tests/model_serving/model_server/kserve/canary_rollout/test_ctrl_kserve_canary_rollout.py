"""TC-CTRL-001: kserve-controller-manager creates canary Deployment."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.canary_rollout.constants import CANARY_MODEL_DIR
from tests.model_serving.model_server.kserve.canary_rollout.utils import (
    deployment_contains_storage_uri,
    get_isvc_deployments,
)

pytestmark = pytest.mark.usefixtures("canary_ctrl_inference_service")


class TestCanaryRolloutController:
    """Validate KServe controller creates stable and canary Deployments."""

    def test_tc_ctrl_001_canary_deployment_created(
        self,
        admin_client: DynamicClient,
        canary_ctrl_inference_service: InferenceService,
    ) -> None:
        """
        Given an InferenceService with a canary array entry
        When the controller reconciles the resource
        Then stable and canary Deployments exist for the InferenceService
        """
        runtime_name = canary_ctrl_inference_service.instance.spec.predictor["model"]["runtime"]
        deployments = get_isvc_deployments(
            client=admin_client,
            isvc=canary_ctrl_inference_service,
            runtime_name=runtime_name,
            expected_count=2,
        )
        assert len(deployments) == 2

        canary_storage_fragment = CANARY_MODEL_DIR.split("/")[-1]
        assert any(
            deployment_contains_storage_uri(deployment, canary_storage_fragment) for deployment in deployments
        ), "Expected one Deployment to reference the canary model storage"

        running_pods = [
            deployment for deployment in deployments if deployment.instance.status.get("readyReplicas", 0) >= 1
        ]
        assert running_pods, "Expected at least one canary Deployment pod to be running"
