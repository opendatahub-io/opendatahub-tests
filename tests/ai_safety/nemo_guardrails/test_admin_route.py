"""Tests for NeMo Guardrails admin route auth segregation."""

import http

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.nemo_guardrails.utils import verify_auth_required
from utilities.guardrails import get_auth_headers
from utilities.resources.nemo_guardrails import NemoGuardrails

_ADMIN_TEST_PATH = "/admin/test"
_ADMIN_CONTAINER_NAME = "kube-rbac-proxy-admin"
_USER_CONTAINER_NAME = "kube-rbac-proxy"
_ADMIN_DOWNSTREAM_PORT = 8444
_ADMIN_HEALTH_PORT = 9445


@pytest.mark.tier1
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoAdminRouteWithAuth:
    """
    Tests for NeMo Guardrails admin route auth segregation when auth is enabled.

    This test class validates:
    1. Admin kube-rbac-proxy sidecar is present and correctly configured
    2. Admin RBAC ConfigMap is created
    3. Service exposes the admin port
    4. Route targets the correct port for authenticated traffic
    5. User proxy restricts /admin/* paths
    6. Admin proxy passes authenticated requests and blocks unauthenticated ones
    """

    def test_admin_rbac_proxy_sidecar_present(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_presidio_auth: NemoGuardrails,
    ) -> None:
        """
        Test that the admin kube-rbac-proxy sidecar is present when auth is enabled.

        Given: NemoGuardrails CR with auth enabled
        When: Deployment pod containers are inspected
        Then: A container named kube-rbac-proxy-admin is present
        """
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={nemo_guardrails_presidio_auth.name}",
            )
        )
        assert pods, f"No pods found for {nemo_guardrails_presidio_auth.name}"

        container_names = [c.name for c in pods[0].instance.spec.containers]
        assert _ADMIN_CONTAINER_NAME in container_names, (
            f"Expected container '{_ADMIN_CONTAINER_NAME}', found: {container_names}"
        )

    def test_admin_rbac_proxy_sidecar_ports(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_presidio_auth: NemoGuardrails,
    ) -> None:
        """
        Test that the admin sidecar exposes the correct ports.

        Given: NemoGuardrails CR with auth enabled
        When: Admin sidecar container ports are inspected
        Then: Port 8444 (downstream) and 9445 (health) are exposed
        """
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={nemo_guardrails_presidio_auth.name}",
            )
        )
        assert pods, f"No pods found for {nemo_guardrails_presidio_auth.name}"

        admin_container = next(
            (c for c in pods[0].instance.spec.containers if c.name == _ADMIN_CONTAINER_NAME),
            None,
        )
        assert admin_container is not None, f"Container '{_ADMIN_CONTAINER_NAME}' not found"

        exposed_ports = {p.containerPort for p in admin_container.ports}
        assert _ADMIN_DOWNSTREAM_PORT in exposed_ports, (
            f"Expected admin downstream port {_ADMIN_DOWNSTREAM_PORT}, found: {exposed_ports}"
        )
        assert _ADMIN_HEALTH_PORT in exposed_ports, (
            f"Expected admin health port {_ADMIN_HEALTH_PORT}, found: {exposed_ports}"
        )

    def test_user_proxy_path_restriction(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_presidio_auth: NemoGuardrails,
    ) -> None:
        """
        Test that the user proxy restricts allowed paths to /v1/* only.

        Given: NemoGuardrails CR with auth enabled
        When: User proxy container args are inspected
        Then: --allow-paths includes /v1/* but not /admin/*
        """
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={nemo_guardrails_presidio_auth.name}",
            )
        )
        assert pods, f"No pods found for {nemo_guardrails_presidio_auth.name}"

        user_container = next(
            (c for c in pods[0].instance.spec.containers if c.name == _USER_CONTAINER_NAME),
            None,
        )
        assert user_container is not None, f"Container '{_USER_CONTAINER_NAME}' not found"

        args = user_container.args or []
        allow_paths_arg = next((arg for arg in args if arg.startswith("--allow-paths=")), None)
        assert allow_paths_arg is not None, f"--allow-paths not found in {_USER_CONTAINER_NAME} args: {args}"
        paths = allow_paths_arg.split("=", 1)[1]
        assert "/v1/" in paths, f"Expected /v1/ in user proxy allow-paths, got: {paths}"
        assert "/admin/" not in paths, f"User proxy should not allow /admin/ paths, got: {paths}"

    def test_admin_proxy_path_allowlist(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_presidio_auth: NemoGuardrails,
    ) -> None:
        """
        Test that the admin proxy allows both /v1/* and /admin/* paths.

        Given: NemoGuardrails CR with auth enabled
        When: Admin proxy container args are inspected
        Then: --allow-paths includes both /v1/* and /admin/*
        """
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={nemo_guardrails_presidio_auth.name}",
            )
        )
        assert pods, f"No pods found for {nemo_guardrails_presidio_auth.name}"

        admin_container = next(
            (c for c in pods[0].instance.spec.containers if c.name == _ADMIN_CONTAINER_NAME),
            None,
        )
        assert admin_container is not None, f"Container '{_ADMIN_CONTAINER_NAME}' not found"

        args = admin_container.args or []
        allow_paths_arg = next((arg for arg in args if arg.startswith("--allow-paths=")), None)
        assert allow_paths_arg is not None, f"--allow-paths not found in {_ADMIN_CONTAINER_NAME} args: {args}"
        paths = allow_paths_arg.split("=", 1)[1]
        assert "/v1/" in paths, f"Expected /v1/ in admin proxy allow-paths, got: {paths}"
        assert "/admin/" in paths, f"Expected /admin/ in admin proxy allow-paths, got: {paths}"

    def test_admin_rbac_configmap_exists(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_presidio_auth: NemoGuardrails,
    ) -> None:
        """
        Test that the admin RBAC ConfigMap is created when auth is enabled.

        Given: NemoGuardrails CR with auth enabled
        When: ConfigMaps in the namespace are inspected
        Then: {name}-rbac-proxy-admin-config ConfigMap exists with a config.yaml key
        """
        cm_name = f"{nemo_guardrails_presidio_auth.name}-rbac-proxy-admin-config"
        cm = ConfigMap(
            client=admin_client,
            name=cm_name,
            namespace=model_namespace.name,
        )

        assert cm.exists, f"ConfigMap '{cm_name}' not found in namespace '{model_namespace.name}'"
        assert "config.yaml" in dict(cm.instance.data or {}), f"Expected 'config.yaml' key in ConfigMap '{cm_name}'"

    def test_service_has_admin_port(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_presidio_auth: NemoGuardrails,
    ) -> None:
        """
        Test that the Service exposes the admin port when auth is enabled.

        Given: NemoGuardrails CR with auth enabled
        When: Service ports are inspected
        Then: A port named {name}-admin exists with port 444 targeting 8444
        """
        svc = Service(
            client=admin_client,
            name=nemo_guardrails_presidio_auth.name,
            namespace=model_namespace.name,
        )
        assert svc.exists, f"Service '{nemo_guardrails_presidio_auth.name}' not found"

        admin_port_name = f"{nemo_guardrails_presidio_auth.name}-admin"
        ports = {p.name: p for p in svc.instance.spec.ports}
        assert admin_port_name in ports, f"Expected admin port '{admin_port_name}', found: {list(ports.keys())}"
        admin_port = ports[admin_port_name]
        assert admin_port.port == 444, f"Expected admin service port 444, got {admin_port.port}"
        assert str(admin_port.targetPort) == str(_ADMIN_DOWNSTREAM_PORT), (
            f"Expected admin targetPort {_ADMIN_DOWNSTREAM_PORT}, got {admin_port.targetPort}"
        )

    def test_user_route_blocks_admin_path(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_guardrails_presidio_auth_route: Route,
        nemo_guardrails_presidio_auth_healthcheck: None,
    ) -> None:
        """
        Test that the user route blocks /admin/* paths at the proxy.

        Given: NemoGuardrails with auth enabled and a valid user token
        When: A request to /admin/test is made via the user-facing route
        Then: The kube-rbac-proxy path filter returns a non-JSON 404 (path not in allow list)
        """
        url = f"https://{nemo_guardrails_presidio_auth_route.host}{_ADMIN_TEST_PATH}"
        response = requests.get(
            url=url,
            headers=get_auth_headers(token=current_client_token),
            verify=openshift_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == http.HTTPStatus.NOT_FOUND, (
            f"Expected 404 from proxy path filter on user route, got {response.status_code}"
        )
        assert "application/json" not in response.headers.get("Content-Type", ""), (
            "Expected non-JSON response from kube-rbac-proxy path filter, but got JSON"
        )

    def test_admin_route_authenticated_not_blocked(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_guardrails_presidio_admin_route: Route,
        nemo_guardrails_presidio_admin_route_healthcheck: None,
    ) -> None:
        """
        Test that authenticated admin users are not blocked at the admin proxy.

        Given: NemoGuardrails with auth enabled and a token with create permission
        When: A request to /admin/test is made via the admin route
        Then: The admin proxy forwards the request to NeMo, which returns a JSON 404
              (proving the proxy forwarded the request rather than blocking it)
        """
        url = f"https://{nemo_guardrails_presidio_admin_route.instance.spec.host}{_ADMIN_TEST_PATH}"
        response = requests.get(
            url=url,
            headers=get_auth_headers(token=current_client_token),
            verify=openshift_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == http.HTTPStatus.NOT_FOUND, (
            f"Expected 404 from NeMo (proxy forwarded the request), got {response.status_code}"
        )
        assert "application/json" in response.headers.get("Content-Type", ""), (
            "Expected JSON response from NeMo, got non-JSON (request may have been blocked by proxy)"
        )

    def test_admin_route_unauthenticated_blocked(
        self,
        openshift_ca_bundle_file: str,
        nemo_guardrails_presidio_auth: NemoGuardrails,
        nemo_guardrails_presidio_admin_route: Route,
        nemo_guardrails_presidio_admin_route_healthcheck: None,
    ) -> None:
        """
        Test that unauthenticated requests are blocked at the admin proxy.

        Given: NemoGuardrails with auth enabled and no auth token
        When: A request to /admin/test is made via the admin route without a token
        Then: The admin proxy returns 401 Unauthorized
        """
        url = f"https://{nemo_guardrails_presidio_admin_route.instance.spec.host}{_ADMIN_TEST_PATH}"
        response = requests.get(
            url=url,
            verify=openshift_ca_bundle_file,
            timeout=30,
        )
        verify_auth_required(response=response)


@pytest.mark.tier2
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoAdminRouteWithoutAuth:
    """
    Tests for NeMo Guardrails when auth is disabled — admin infrastructure should be absent.

    This test class validates:
    1. Admin sidecar is not injected without auth
    2. Admin RBAC ConfigMap is not created without auth
    3. Service does not expose the admin port without auth
    4. Route targets the plain HTTP port without auth
    """

    def test_admin_sidecar_absent_without_auth(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_presidio: NemoGuardrails,
    ) -> None:
        """
        Test that the admin sidecar is not present when auth is disabled.

        Given: NemoGuardrails CR without auth annotation
        When: Pod containers are inspected
        Then: No container named kube-rbac-proxy-admin exists
        """
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={nemo_guardrails_presidio.name}",
            )
        )
        assert pods, f"No pods found for {nemo_guardrails_presidio.name}"

        container_names = [c.name for c in pods[0].instance.spec.containers]
        assert _ADMIN_CONTAINER_NAME not in container_names, (
            f"Admin sidecar '{_ADMIN_CONTAINER_NAME}' should not exist without auth, found: {container_names}"
        )

    def test_admin_configmap_absent_without_auth(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_presidio: NemoGuardrails,
    ) -> None:
        """
        Test that the admin RBAC ConfigMap is not created when auth is disabled.

        Given: NemoGuardrails CR without auth annotation
        When: ConfigMaps are inspected
        Then: {name}-rbac-proxy-admin-config does not exist
        """
        cm_name = f"{nemo_guardrails_presidio.name}-rbac-proxy-admin-config"
        cm = ConfigMap(
            client=admin_client,
            name=cm_name,
            namespace=model_namespace.name,
        )
        assert not cm.exists, f"ConfigMap '{cm_name}' should not exist when auth is disabled"

    def test_service_no_admin_port_without_auth(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_presidio: NemoGuardrails,
    ) -> None:
        """
        Test that the Service does not expose an admin port when auth is disabled.

        Given: NemoGuardrails CR without auth annotation
        When: Service ports are inspected
        Then: No port named {name}-admin exists
        """
        svc = Service(
            client=admin_client,
            name=nemo_guardrails_presidio.name,
            namespace=model_namespace.name,
        )
        assert svc.exists, f"Service '{nemo_guardrails_presidio.name}' not found"

        admin_port_name = f"{nemo_guardrails_presidio.name}-admin"
        port_names = [p.name for p in svc.instance.spec.ports]
        assert admin_port_name not in port_names, (
            f"Admin port '{admin_port_name}' should not exist without auth, found ports: {port_names}"
        )
