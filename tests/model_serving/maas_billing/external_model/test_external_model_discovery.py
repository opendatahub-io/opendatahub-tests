from __future__ import annotations

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.namespace import Namespace

from tests.model_serving.maas_billing.external_model.utils import (
    get_httproute,
    get_service,
)
from utilities.resources.external_model import ExternalModel
from utilities.resources.external_provider import ExternalProvider

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "external_provider_cr",
    "external_model_cr",
    "external_model_ref",
)
class TestExternalModelDiscovery:
    """Verify ExternalModel reconciler creates HTTPRoute and ExternalProvider reconciler creates Service."""

    @pytest.mark.tier1
    def test_external_model_cr_exists(
        self,
        external_model_cr: ExternalModel,
    ) -> None:
        """Given a deployed ExternalModel CR, when checking the cluster, then the CR exists."""
        assert external_model_cr.exists, f"ExternalModel '{external_model_cr.name}' was not created"
        LOGGER.info(f"ExternalModel '{external_model_cr.name}' exists")

    @pytest.mark.tier1
    def test_maas_model_ref_created(
        self,
        external_model_ref: MaaSModelRef,
    ) -> None:
        """Given an ExternalModel, when checking MaaSModelRef, then a ref to the model exists."""
        assert external_model_ref.exists, f"MaaSModelRef '{external_model_ref.name}' not found"
        LOGGER.info(f"MaaSModelRef '{external_model_ref.name}' exists")

    @pytest.mark.tier1
    def test_reconciler_created_httproute(
        self,
        admin_client: DynamicClient,
        external_model_cr: ExternalModel,
        maas_unprivileged_model_namespace: Namespace,
    ) -> None:
        """Given a reconciled ExternalModel, when listing HTTPRoutes, then one named after the model exists."""
        route = get_httproute(
            client=admin_client,
            name=external_model_cr.name,
            namespace=maas_unprivileged_model_namespace.name,
        )
        assert route is not None, (
            f"HTTPRoute '{external_model_cr.name}' not found in namespace '{maas_unprivileged_model_namespace.name}'"
        )
        LOGGER.info(f"HTTPRoute '{external_model_cr.name}' created by reconciler")

    @pytest.mark.tier1
    def test_reconciler_created_backend_service(
        self,
        admin_client: DynamicClient,
        external_provider_cr: ExternalProvider,
        maas_unprivileged_model_namespace: Namespace,
    ) -> None:
        """Given a reconciled ExternalProvider, when listing Services, then one named after the provider exists."""
        svc = get_service(
            client=admin_client,
            name=external_provider_cr.name,
            namespace=maas_unprivileged_model_namespace.name,
        )
        assert svc is not None, (
            f"Service '{external_provider_cr.name}' not found in namespace '{maas_unprivileged_model_namespace.name}'"
        )
        LOGGER.info(f"Service '{external_provider_cr.name}' created by reconciler")
