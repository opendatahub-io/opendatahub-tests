import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment

from utilities.resources.http_route import HTTPRoute

LOGGER = structlog.get_logger(name=__name__)

DATA_SCIENCE_GATEWAY_NAME = "data-science-gateway"
DATA_SCIENCE_GATEWAY_NAMESPACE = "openshift-ingress"


def get_gateway_domain_from_operator(client: DynamicClient, namespace: str) -> str | None:
    """Extract GATEWAY_DOMAIN from the model-registry-operator deployment env vars."""
    deployments = list(
        Deployment.get(
            client=client,
            namespace=namespace,
            label_selector="control-plane=model-registry-operator",
        )
    )
    if not deployments:
        return None

    containers = deployments[0].instance.spec.template.spec.containers
    for container in containers:
        for env_var in container.env or []:
            if env_var.name == "GATEWAY_DOMAIN" and env_var.value:
                return env_var.value

    return None


def get_model_registry_httproutes(client: DynamicClient, namespace: str) -> list[HTTPRoute]:
    """Get HTTPRoutes in a namespace that route to model registry via data-science-gateway."""
    all_routes = list(HTTPRoute.get(client=client, namespace=namespace))

    matched = []
    for route in all_routes:
        parent_refs = route.instance.spec.get("parentRefs", [])
        has_gateway_parent = any(
            ref.get("name") == DATA_SCIENCE_GATEWAY_NAME and ref.get("namespace") == DATA_SCIENCE_GATEWAY_NAMESPACE
            for ref in parent_refs
        )
        if not has_gateway_parent:
            continue

        has_model_registry_backend = any(
            backend_ref.get("name", "").startswith("model-registry")
            for rule in route.instance.spec.get("rules", [])
            for backend_ref in rule.get("backendRefs", [])
        )
        if has_model_registry_backend:
            matched.append(route)

    return matched
