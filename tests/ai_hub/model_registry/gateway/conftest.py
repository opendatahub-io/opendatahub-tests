import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config

from tests.ai_hub.model_registry.gateway.utils import (
    DATA_SCIENCE_GATEWAY_NAME,
    DATA_SCIENCE_GATEWAY_NAMESPACE,
    get_gateway_domain_from_operator,
    get_model_registry_httproutes,
)
from utilities.resources.http_route import HTTPRoute
from utilities.resources.reference_grant import ReferenceGrant

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def gateway_domain(admin_client: DynamicClient) -> str:
    """Get GATEWAY_DOMAIN from the model-registry-operator deployment."""
    domain = get_gateway_domain_from_operator(
        client=admin_client,
        namespace=py_config["applications_namespace"],
    )
    assert domain, "GATEWAY_DOMAIN not set on model-registry-operator — gateway routing not enabled"
    LOGGER.info(f"GATEWAY_DOMAIN: {domain}")
    return domain


@pytest.fixture(scope="class")
def model_registry_httproutes(
    admin_client: DynamicClient,
    gateway_domain: str,
    model_registry_instance: list,
) -> list[HTTPRoute]:
    """Get HTTPRoutes in applications namespace that route to model registry via data-science-gateway."""
    routes = get_model_registry_httproutes(client=admin_client, namespace=py_config["applications_namespace"])
    assert routes, (
        f"No model registry HTTPRoutes found via data-science-gateway in {py_config['applications_namespace']}"
    )
    LOGGER.info(f"Found {len(routes)} model registry HTTPRoutes via data-science-gateway")
    return routes


@pytest.fixture(scope="class")
def httproutes_without_gateway_ref(
    model_registry_httproutes: list[HTTPRoute],
) -> list[str]:
    """HTTPRoute names missing parentRef to data-science-gateway."""
    return [
        route.name
        for route in model_registry_httproutes
        if not any(
            ref.name == DATA_SCIENCE_GATEWAY_NAME and ref.namespace == DATA_SCIENCE_GATEWAY_NAMESPACE
            for ref in route.instance.spec.parentRefs
        )
    ]


@pytest.fixture(scope="class")
def httproutes_without_path_match(
    model_registry_httproutes: list[HTTPRoute],
) -> list[str]:
    """HTTPRoute names missing a PathPrefix match containing 'model-registry'."""
    missing = []
    for route in model_registry_httproutes:
        has_path_match = any(
            match.get("path", {}).get("type") == "PathPrefix"
            and "model-registry" in match.get("path", {}).get("value", "")
            for rule in route.instance.spec.get("rules", [])
            for match in rule.get("matches", [])
        )
        if not has_path_match:
            missing.append(route.name)

    return missing


@pytest.fixture(scope="class")
def gateway_model_registry_url(
    gateway_domain: str,
    model_registry_httproutes: list[HTTPRoute],
) -> str:
    """Gateway URL with model registry path prefix from the first HTTPRoute."""
    for route in model_registry_httproutes:
        for rule in route.instance.spec.get("rules", []):
            for match in rule.get("matches", []):
                path = match.get("path", {})
                if path.get("type") == "PathPrefix" and path.get("value"):
                    url = f"https://{gateway_domain}{path['value']}"
                    LOGGER.info(f"Gateway model registry URL: {url}")
                    return url

    pytest.fail("No PathPrefix found in any model registry HTTPRoute")


@pytest.fixture(scope="class")
def model_registry_reference_grants(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> list[ReferenceGrant]:
    """Get ReferenceGrants in model registry namespace."""
    grants = list(ReferenceGrant.get(client=admin_client, namespace=model_registry_namespace))
    assert grants, f"No ReferenceGrants found in {model_registry_namespace}"
    LOGGER.info(f"Found {len(grants)} ReferenceGrants in {model_registry_namespace}")
    return grants


@pytest.fixture(scope="class")
def cross_namespace_reference_grant(
    model_registry_reference_grants: list[ReferenceGrant],
) -> ReferenceGrant | None:
    """ReferenceGrant allowing HTTPRoutes from applications namespace to reference Services."""
    for grant in model_registry_reference_grants:
        from_rules = grant.instance.spec.get("from", [])
        to_rules = grant.instance.spec.get("to", [])

        allows_httproute_from_apps = any(
            rule.get("group") == "gateway.networking.k8s.io"
            and rule.get("kind") == "HTTPRoute"
            and rule.get("namespace") == py_config["applications_namespace"]
            for rule in from_rules
        )
        allows_service_target = any(rule.get("kind") == "Service" for rule in to_rules)

        if allows_httproute_from_apps and allows_service_target:
            LOGGER.info(f"ReferenceGrant '{grant.name}' allows HTTPRoute->Service cross-namespace access")
            return grant

    return None
