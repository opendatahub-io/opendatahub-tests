from __future__ import annotations

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.namespace import Namespace

from tests.ai_gateway.models_as_a_service.external_model.utils import (
    EXTERNAL_MODEL_HTTPROUTE_TIMEOUT,
    get_external_model_http_route_name,
    get_httproute,
    wait_for_external_provider_transport,
)
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE
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
    """Verify ExternalModel HTTPRoute and ExternalProvider egress transport after reconciliation."""

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
        """Given a reconciled ExternalModel, when checking status, then its HTTPRoute exists in the namespace."""
        http_route_name = get_external_model_http_route_name(external_model=external_model_cr)
        assert http_route_name, f"ExternalModel '{external_model_cr.name}' did not publish an HTTPRoute name in status"
        route = get_httproute(
            client=admin_client,
            name=http_route_name,
            namespace=maas_unprivileged_model_namespace.name,
        )
        assert route is not None, (
            f"HTTPRoute '{http_route_name}' not found in namespace '{maas_unprivileged_model_namespace.name}'"
        )
        LOGGER.info(f"HTTPRoute '{http_route_name}' created by reconciler")

    @pytest.mark.tier1
    def test_reconciler_created_backend_service(
        self,
        admin_client: DynamicClient,
        external_provider_cr: ExternalProvider,
    ) -> None:
        """Given reconciled ExternalProvider and attached ExternalModel, provider egress transport exists.

        Legacy IPP creates a Service in the model namespace; AGC creates a labeled DestinationRule on the gateway.
        """
        transport_path = wait_for_external_provider_transport(
            client=admin_client,
            provider=external_provider_cr,
            gateway_name=MAAS_GATEWAY_NAME,
            gateway_namespace=MAAS_GATEWAY_NAMESPACE,
            timeout=EXTERNAL_MODEL_HTTPROUTE_TIMEOUT,
        )
        LOGGER.info(
            f"ExternalProvider '{external_provider_cr.name}' transport verified via {transport_path}",
        )
