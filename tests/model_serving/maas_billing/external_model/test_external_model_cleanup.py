from __future__ import annotations

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.model_serving.maas_billing.external_model.utils import (
    EXTERNAL_PROVIDER_NAME,
    external_provider_ref,
    wait_for_httproute,
    wait_for_httproute_deleted,
)
from utilities.resources.external_model import ExternalModel

LOGGER = structlog.get_logger(name=__name__)

CLEANUP_TEST_MODEL_NAME = "e2e-cleanup-test"


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
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

        with ExternalModel(
            client=admin_client,
            name=CLEANUP_TEST_MODEL_NAME,
            namespace=namespace,
            external_provider_refs=[external_provider_ref(provider_name=EXTERNAL_PROVIDER_NAME)],
            teardown=True,
            wait_for_resource=True,
        ):
            wait_for_httproute(
                client=admin_client,
                name=CLEANUP_TEST_MODEL_NAME,
                namespace=namespace,
                timeout=60,
            )
            LOGGER.info(f"HTTPRoute '{CLEANUP_TEST_MODEL_NAME}' confirmed before deletion")

        wait_for_httproute_deleted(
            client=admin_client,
            name=CLEANUP_TEST_MODEL_NAME,
            namespace=namespace,
            timeout=60,
        )
        LOGGER.info(f"HTTPRoute '{CLEANUP_TEST_MODEL_NAME}' correctly garbage-collected after ExternalModel deletion")
