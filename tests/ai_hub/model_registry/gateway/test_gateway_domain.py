import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.route import Route
from pytest_testconfig import config as py_config

from tests.ai_hub.model_registry.gateway.utils import DATA_SCIENCE_GATEWAY_NAME, DATA_SCIENCE_GATEWAY_NAMESPACE
from utilities.resources.http_route import HTTPRoute
from utilities.resources.reference_grant import ReferenceGrant

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.tier1,
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    ),
]


class TestModelRegistryGatewayDomain:
    """Tests for GATEWAY_DOMAIN-based HTTPRoute routing for model registry (RHOAIENG-49128)."""

    def test_httproute_created_for_model_registry(
        self,
        gateway_domain: str,
        model_registry_httproutes: list[HTTPRoute],
        httproutes_without_gateway_ref: list[str],
    ):
        """Given GATEWAY_DOMAIN is set on the model-registry-operator
        When listing HTTPRoutes in the applications namespace
        Then each HTTPRoute has a parentRef pointing to data-science-gateway
        """
        assert not httproutes_without_gateway_ref, (
            f"HTTPRoutes not referencing {DATA_SCIENCE_GATEWAY_NAMESPACE}/{DATA_SCIENCE_GATEWAY_NAME}: "
            f"{httproutes_without_gateway_ref}"
        )

        LOGGER.info(f"Validated {len(model_registry_httproutes)} HTTPRoutes reference {DATA_SCIENCE_GATEWAY_NAME}")

    def test_httproute_path_matches_registry_name(
        self,
        gateway_domain: str,
        model_registry_httproutes: list[HTTPRoute],
        httproutes_without_path_match: list[str],
    ):
        """Given HTTPRoutes exist for model registry
        When inspecting the HTTPRoute spec
        Then each HTTPRoute has a path match rule with a prefix containing model-registry
        """
        assert not httproutes_without_path_match, (
            f"HTTPRoutes missing PathPrefix match containing 'model-registry': {httproutes_without_path_match}"
        )

        LOGGER.info(f"Validated PathPrefix match for {len(model_registry_httproutes)} HTTPRoutes")

    def test_httproute_backend_refs_target_model_registry_service(
        self,
        gateway_domain: str,
        model_registry_httproutes: list[HTTPRoute],
        model_registry_namespace: str,
    ):
        """Given HTTPRoutes exist for model registry
        When inspecting the backendRefs in each HTTPRoute rule
        Then each backendRef targets a Service in the model registry namespace
        """
        validation_errors = []
        for route in model_registry_httproutes:
            rules = route.instance.spec.get("rules", [])
            for rule in rules:
                backend_refs = rule.get("backendRefs", [])
                if not backend_refs:
                    validation_errors.append(f"HTTPRoute '{route.name}' has a rule with no backendRefs")
                    continue

                for backend_ref in backend_refs:
                    ref_namespace = backend_ref.get("namespace", "")
                    ref_kind = backend_ref.get("kind", "Service")
                    if ref_kind != "Service":
                        validation_errors.append(
                            f"HTTPRoute '{route.name}' backendRef kind is '{ref_kind}', expected 'Service'"
                        )
                    if ref_namespace != model_registry_namespace:
                        validation_errors.append(
                            f"HTTPRoute '{route.name}' backendRef namespace is '{ref_namespace}', "
                            f"expected '{model_registry_namespace}'"
                        )

        assert not validation_errors, "BackendRef validation failed:\n" + "\n".join(validation_errors)

        LOGGER.info(
            f"Validated backendRefs target {model_registry_namespace} for {len(model_registry_httproutes)} HTTPRoutes"
        )

    def test_reference_grant_allows_cross_namespace_access(
        self,
        gateway_domain: str,
        model_registry_namespace: str,
        cross_namespace_reference_grant: ReferenceGrant | None,
    ):
        """Given model registries in rhoai-model-registries and HTTPRoutes in redhat-ods-applications
        When checking ReferenceGrants in the model registry namespace
        Then a ReferenceGrant exists allowing HTTPRoutes from the applications namespace to reference Services
        """
        assert cross_namespace_reference_grant, (
            f"No ReferenceGrant in {model_registry_namespace} allows HTTPRoutes "
            f"from {py_config['applications_namespace']} to reference Services"
        )

        LOGGER.info(
            f"ReferenceGrant '{cross_namespace_reference_grant.name}' allows cross-namespace access "
            f"in {model_registry_namespace}"
        )

    @pytest.mark.sanity
    def test_model_registry_reachable_via_gateway_domain(
        self,
        gateway_model_registry_url: str,
    ):
        """Given GATEWAY_DOMAIN is set and HTTPRoutes exist for model registry
        When sending a request to the gateway domain with the model registry path
        Then the gateway routes the request successfully
        """
        LOGGER.info(f"Testing gateway connectivity: {gateway_model_registry_url}")
        response = requests.get(gateway_model_registry_url, verify=False, timeout=30)

        assert response.ok, f"Gateway returned {response.status_code} at {gateway_model_registry_url}"
        LOGGER.info(f"Gateway responded with HTTP {response.status_code} at {gateway_model_registry_url}")

    def test_no_standalone_routes_with_gateway_domain(
        self,
        admin_client: DynamicClient,
        gateway_domain: str,
        model_registry_namespace: str,
    ):
        """Given GATEWAY_DOMAIN is set on the operator
        When checking OpenShift Routes in the model registry namespace
        Then no Route has a host matching the gateway domain
        """
        routes = list(Route.get(client=admin_client, namespace=model_registry_namespace))

        gateway_domain_routes = [route for route in routes if gateway_domain in (route.instance.spec.host or "")]

        assert not gateway_domain_routes, (
            f"Found {len(gateway_domain_routes)} standalone Route(s) using gateway domain '{gateway_domain}' "
            f"in {model_registry_namespace}. Routes should use HTTPRoutes via data-science-gateway instead: "
            f"{[route.name for route in gateway_domain_routes]}"
        )

        LOGGER.info(
            f"Confirmed no standalone Routes with gateway domain in {model_registry_namespace} "
            f"({len(routes)} total Routes checked)"
        )
