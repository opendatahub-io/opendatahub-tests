"""Utilities and verification helpers for body-based routing (BBR) tests."""

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.service import Service

from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.resources.destination_rule import DestinationRule

LOGGER = structlog.get_logger(name=__name__)

BBR_PRE_PROCESSING_DEPLOYMENT_NAME: str = "payload-pre-processing"
BBR_PRE_PROCESSING_SERVICE_NAME: str = "payload-pre-processing"
BBR_PRE_PROCESSING_GRPC_PORT: int = 9004
BBR_PRE_PROCESSING_DESTINATION_RULE_NAME: str = "payload-pre-processing"


def verify_bbr_pre_processing_deployment_ready(
    admin_client: DynamicClient,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> None:
    """Assert the payload-pre-processing Deployment exists and all replicas are ready."""
    deployment = Deployment(
        client=admin_client,
        name=BBR_PRE_PROCESSING_DEPLOYMENT_NAME,
        namespace=gateway_namespace,
    )
    assert deployment.exists, (
        f"Deployment '{gateway_namespace}/{BBR_PRE_PROCESSING_DEPLOYMENT_NAME}' not found — "
        "payload-pre-processing must be deployed by the controller after PR #948 is merged"
    )
    ready_replicas: int = deployment.instance.status.readyReplicas or 0
    desired_replicas: int = deployment.instance.spec.replicas or 1
    assert ready_replicas >= desired_replicas, (
        f"Deployment '{BBR_PRE_PROCESSING_DEPLOYMENT_NAME}' has {ready_replicas}/{desired_replicas} ready replicas "
        f"in namespace '{gateway_namespace}'"
    )
    LOGGER.info(
        f"Deployment '{gateway_namespace}/{BBR_PRE_PROCESSING_DEPLOYMENT_NAME}' is ready "
        f"({ready_replicas}/{desired_replicas} replicas)"
    )


def verify_bbr_pre_processing_service_port(
    admin_client: DynamicClient,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> None:
    """Assert the payload-pre-processing Service exists and exposes the expected gRPC port."""
    service = Service(
        client=admin_client,
        name=BBR_PRE_PROCESSING_SERVICE_NAME,
        namespace=gateway_namespace,
    )
    assert service.exists, f"Service '{gateway_namespace}/{BBR_PRE_PROCESSING_SERVICE_NAME}' not found"
    service_ports = service.instance.spec.ports or []
    exposed_port_numbers = [port.port for port in service_ports]
    assert BBR_PRE_PROCESSING_GRPC_PORT in exposed_port_numbers, (
        f"Service '{gateway_namespace}/{BBR_PRE_PROCESSING_SERVICE_NAME}' does not expose "
        f"port {BBR_PRE_PROCESSING_GRPC_PORT} — found: {exposed_port_numbers!r}"
    )
    LOGGER.info(
        f"Service '{gateway_namespace}/{BBR_PRE_PROCESSING_SERVICE_NAME}' "
        f"exposes gRPC port {BBR_PRE_PROCESSING_GRPC_PORT}"
    )


def verify_bbr_pre_processing_destination_rule_exists(
    admin_client: DynamicClient,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> None:
    """Assert the payload-pre-processing DestinationRule exists in the gateway namespace."""
    destination_rule = DestinationRule(
        client=admin_client,
        name=BBR_PRE_PROCESSING_DESTINATION_RULE_NAME,
        namespace=gateway_namespace,
    )
    assert destination_rule.exists, (
        f"DestinationRule '{gateway_namespace}/{BBR_PRE_PROCESSING_DESTINATION_RULE_NAME}' not found — "
        "expected to be created by the controller after reconciliation"
    )
    LOGGER.info(f"DestinationRule '{gateway_namespace}/{BBR_PRE_PROCESSING_DESTINATION_RULE_NAME}' exists")
