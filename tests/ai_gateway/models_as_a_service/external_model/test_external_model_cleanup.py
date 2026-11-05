from __future__ import annotations

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.ai_gateway.models_as_a_service.external_model.utils import (
    EXTERNAL_MODEL_HTTPROUTE_TIMEOUT,
    EXTERNAL_PROVIDER_NAME,
    external_provider_ref,
    wait_for_external_model_httproute,
    wait_for_httproute_deleted,
)
from utilities.resources.external_model import ExternalModel

LOGGER = structlog.get_logger(name=__name__)

CLEANUP_TEST_MODEL_NAME = "e2e-cleanup-test"


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "external_model_credential_secret",
    "external_provider_cr",
)
class TestExternalModelCleanup:
    """Verify resource cleanup when ExternalModel CRs are deleted."""

    @pytest.mark.tier2
    def test_delete_removes_httproute(
        self,
        admin_client: DynamicClient,
        maas_unprivileged_model_namespace: Namespace,
    ) -> None:
        """Given a temporary ExternalModel with an HTTPRoute, when the CR is deleted, then the HTTPRoute is removed."""
        namespace = maas_unprivileged_model_namespace.name
        http_route_name = ""

        with ExternalModel(
            client=admin_client,
            name=CLEANUP_TEST_MODEL_NAME,
            namespace=namespace,
            external_provider_refs=[external_provider_ref(provider_name=EXTERNAL_PROVIDER_NAME)],
            teardown=True,
            wait_for_resource=True,
        ) as external_model:
            http_route_name = wait_for_external_model_httproute(
                client=admin_client,
                external_model=external_model,
                namespace=namespace,
                timeout=EXTERNAL_MODEL_HTTPROUTE_TIMEOUT,
            )
            LOGGER.info(f"HTTPRoute '{http_route_name}' confirmed before deletion")

        wait_for_httproute_deleted(
            client=admin_client,
            name=http_route_name,
            namespace=namespace,
            timeout=EXTERNAL_MODEL_HTTPROUTE_TIMEOUT,
        )
        LOGGER.info(f"HTTPRoute '{http_route_name}' correctly garbage-collected after ExternalModel deletion")
