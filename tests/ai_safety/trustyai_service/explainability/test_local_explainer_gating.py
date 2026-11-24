from http import HTTPStatus

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.trustyai_service import TrustyAIService

from tests.ai_safety.trustyai_service.explainability.utils import (
    local_explainers_enabled_by_deployment,
)
from tests.ai_safety.trustyai_service.trustyai_service_utils import TrustyAIServiceClient
from utilities.constants import TRUSTYAI_SERVICE_NAME


@pytest.mark.parametrize(
    "model_namespace, trustyai_service",
    [
        pytest.param(
            {"name": "test-local-explainer-gate"},
            {"storage": "pvc"},
            id="test_pvc_storage",
        ),
    ],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.tier2
@pytest.mark.rawdeployment
class TestLocalExplainerGating:
    """Verify local explainer routes stay unregistered unless explicitly enabled."""

    @pytest.mark.parametrize("algorithm", ["lime", "shap"])
    def test_local_explainer_is_not_registered_by_default(
        self,
        admin_client: DynamicClient,
        current_client_token: str,
        model_namespace: Namespace,
        trustyai_service: TrustyAIService,
        algorithm: str,
    ) -> None:
        """Given default feature flags, when a local explanation is requested, then the route is absent."""
        deployment = Deployment(
            client=admin_client,
            name=TRUSTYAI_SERVICE_NAME,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        if local_explainers_enabled_by_deployment(deployment=deployment):
            pytest.skip("Local explainers are enabled by deployment configuration")

        client = TrustyAIServiceClient(
            token=current_client_token,
            service=trustyai_service,
            client=admin_client,
        )
        response = client.request_local_explanation(
            algorithm=algorithm,
            payload={"predictionId": "unused", "config": {}},
            timeout=30,
        )

        assert response.status_code == HTTPStatus.NOT_FOUND, response.text
