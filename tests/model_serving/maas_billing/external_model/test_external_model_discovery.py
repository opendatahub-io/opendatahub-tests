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

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
)
class TestExternalModelDiscovery:
    """Verify ExternalModel reconciler creates the expected routing resources."""

    @pytest.mark.tier1
    def test_external_model_cr_exists(
        self,
        external_model_cr: ExternalModel,
    ) -> None:
        """Given an ExternalModel CR, verify it exists on the cluster."""
        assert external_model_cr.exists, f"ExternalModel '{external_model_cr.name}' was not created"
        LOGGER.info(f"ExternalModel '{external_model_cr.name}' exists")

    @pytest.mark.tier1
    def test_maas_model_ref_created(
        self,
        external_model_ref: MaaSModelRef,
    ) -> None:
        """Given an ExternalModel, verify MaaSModelRef referencing it exists."""
        assert external_model_ref.exists, f"MaaSModelRef '{external_model_ref.name}' not found"
        LOGGER.info(f"MaaSModelRef '{external_model_ref.name}' exists")

    @pytest.mark.tier1
    def test_reconciler_created_httproute(
        self,
        admin_client: DynamicClient,
        external_model_ref: MaaSModelRef,
        maas_unprivileged_model_namespace: Namespace,
    ) -> None:
        """Given a reconciled ExternalModel, verify an HTTPRoute was created."""
        status = external_model_ref.instance.status or {}
        httproute_name = status.get("httpRouteName")
        assert httproute_name, f"MaaSModelRef '{external_model_ref.name}' has no httpRouteName in status"

        route = get_httproute(
            client=admin_client,
            name=httproute_name,
            namespace=maas_unprivileged_model_namespace.name,
        )
        assert route is not None, (
            f"HTTPRoute '{httproute_name}' not found in namespace '{maas_unprivileged_model_namespace.name}'"
        )
        LOGGER.info(f"HTTPRoute '{httproute_name}' created by reconciler")

    @pytest.mark.tier1
    def test_reconciler_created_backend_service(
        self,
        admin_client: DynamicClient,
        external_model_ref: MaaSModelRef,
        maas_unprivileged_model_namespace: Namespace,
    ) -> None:
        """Given a reconciled ExternalModel, verify a backend Service was created."""
        svc = get_service(
            client=admin_client,
            name=external_model_ref.name,
            namespace=maas_unprivileged_model_namespace.name,
        )
        assert svc is not None, (
            f"Service '{external_model_ref.name}' not found in namespace '{maas_unprivileged_model_namespace.name}'"
        )
        LOGGER.info(f"Service '{external_model_ref.name}' created by reconciler")
