from __future__ import annotations

from typing import Any, Literal

import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError, ResourceNotFoundError
from ocp_resources.service import Service
from timeout_sampler import TimeoutSampler

from tests.ai_gateway.models_as_a_service.utils import (
    get_external_model_http_route_name,
    get_httproute,
    wait_for_external_model_httproute,
    wait_for_httproute,
)
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE, ApiGroups
from utilities.resources.destination_rule import DestinationRule
from utilities.resources.external_provider import ExternalProvider

ExternalProviderTransportPath = Literal["ipp-service", "agc-destinationrule"]

EXTERNAL_PROVIDER_LABEL = "inference.opendatahub.io/external-provider"
EXTERNAL_GATEWAY_LABEL = "inference.opendatahub.io/external-gateway"

LOGGER = structlog.get_logger(name=__name__)

EXTERNAL_MODEL_NAME = "e2e-external-model"
EXTERNAL_PROVIDER_NAME = "e2e-external-provider"
EXTERNAL_ENDPOINT = "httpbin.org"
EXTERNAL_TARGET_MODEL = "gpt-3.5-turbo"
EXTERNAL_API_FORMAT = "openai-chat"
EXTERNAL_PROVIDER_PATH = "/v1/chat/completions"
EXTERNAL_AUTH_POLICY_NAME = "e2e-external-access"
EXTERNAL_SUBSCRIPTION_NAME = "e2e-external-subscription"
EXTERNAL_SECRET_NAME = f"{EXTERNAL_MODEL_NAME}-api-key"
INFERENCE_EXTERNAL_MODEL_API_GROUP = ApiGroups.INFERENCE_OPENDATAHUB_IO
EXTERNAL_MODEL_HTTPROUTE_TIMEOUT = 120

__all__ = [
    "EXTERNAL_API_FORMAT",
    "EXTERNAL_AUTH_POLICY_NAME",
    "EXTERNAL_ENDPOINT",
    "EXTERNAL_GATEWAY_LABEL",
    "EXTERNAL_MODEL_HTTPROUTE_TIMEOUT",
    "EXTERNAL_MODEL_NAME",
    "EXTERNAL_PROVIDER_LABEL",
    "EXTERNAL_PROVIDER_NAME",
    "EXTERNAL_PROVIDER_PATH",
    "EXTERNAL_SECRET_NAME",
    "EXTERNAL_SUBSCRIPTION_NAME",
    "EXTERNAL_TARGET_MODEL",
    "INFERENCE_EXTERNAL_MODEL_API_GROUP",
    "ExternalProviderTransportPath",
    "external_provider_ref",
    "get_external_model_http_route_name",
    "get_httproute",
    "get_ipp_external_provider_service",
    "get_service",
    "list_agc_provider_destination_rules",
    "wait_for_external_model_httproute",
    "wait_for_external_provider_transport",
    "wait_for_httproute",
    "wait_for_httproute_deleted",
]


def external_provider_ref(provider_name: str, *, target_model: str = EXTERNAL_TARGET_MODEL) -> dict[str, Any]:
    """Build an externalProviderRefs entry for an ExternalModel spec."""
    return {
        "ref": {"name": provider_name},
        "targetModel": target_model,
        "apiFormat": EXTERNAL_API_FORMAT,
        "path": EXTERNAL_PROVIDER_PATH,
    }


def get_service(
    client: DynamicClient,
    name: str,
    namespace: str,
) -> Service | None:
    """Look up a Service by name/namespace. Returns None if not found."""
    try:
        svc = Service(client=client, name=name, namespace=namespace)
        if svc.exists:
            return svc
    except NotFoundError, ResourceNotFoundError:
        LOGGER.debug(f"Service {namespace}/{name} not found")
    return None


def get_ipp_external_provider_service(
    client: DynamicClient,
    provider: ExternalProvider,
) -> Service | None:
    """Return the legacy IPP Service named after the ExternalProvider CR, if present."""
    return get_service(client=client, name=provider.name, namespace=provider.namespace)


def list_agc_provider_destination_rules(
    client: DynamicClient,
    *,
    provider_name: str,
    gateway_name: str,
    gateway_namespace: str,
) -> list[DestinationRule]:
    """List AGC provider DestinationRules in the gateway namespace for a provider and gateway."""
    label_selector = f"{EXTERNAL_PROVIDER_LABEL}={provider_name},{EXTERNAL_GATEWAY_LABEL}={gateway_name}"
    return list(
        DestinationRule.get(
            client=client,
            namespace=gateway_namespace,
            label_selector=label_selector,
        )
    )


def _resolve_external_provider_transport_path(
    client: DynamicClient,
    provider: ExternalProvider,
    gateway_name: str,
    gateway_namespace: str,
) -> ExternalProviderTransportPath | None:
    """Return which transport path is present, or None when neither legacy nor AGC transport exists."""
    if get_ipp_external_provider_service(client=client, provider=provider) is not None:
        return "ipp-service"
    destination_rules = list_agc_provider_destination_rules(
        client=client,
        provider_name=provider.name,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
    )
    if destination_rules:
        return "agc-destinationrule"
    return None


def wait_for_external_provider_transport(
    client: DynamicClient,
    provider: ExternalProvider,
    *,
    gateway_name: str = MAAS_GATEWAY_NAME,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
    timeout: int = EXTERNAL_MODEL_HTTPROUTE_TIMEOUT,
) -> ExternalProviderTransportPath:
    """Poll until IPP Service or AGC DestinationRule transport exists for the provider."""
    for transport_path in TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        func=_resolve_external_provider_transport_path,
        client=client,
        provider=provider,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
    ):
        if transport_path == "ipp-service":
            LOGGER.info(
                "ExternalProvider transport ready via legacy IPP Service",
                provider=f"{provider.namespace}/{provider.name}",
            )
            return transport_path
        if transport_path == "agc-destinationrule":
            LOGGER.info(
                "ExternalProvider transport ready via AGC DestinationRule",
                provider=f"{provider.namespace}/{provider.name}",
                gateway=f"{gateway_namespace}/{gateway_name}",
            )
            return transport_path

    raise TimeoutError(
        f"ExternalProvider '{provider.namespace}/{provider.name}' did not get transport "
        f"(IPP Service in model namespace or AGC DestinationRule on gateway) within {timeout}s"
    )


def wait_for_httproute_deleted(
    client: DynamicClient,
    name: str,
    namespace: str,
    timeout: int = 60,
) -> None:
    """Poll until the HTTPRoute no longer exists, or raise on timeout."""
    for route in TimeoutSampler(
        wait_timeout=timeout,
        sleep=3,
        func=get_httproute,
        client=client,
        name=name,
        namespace=namespace,
    ):
        if route is None:
            return
