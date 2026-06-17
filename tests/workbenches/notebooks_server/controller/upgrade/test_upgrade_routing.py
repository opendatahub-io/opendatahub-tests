from typing import Any

import pytest
from ocp_resources.notebook import Notebook
from pytest_testconfig import config as py_config

from utilities.resources.http_route import HTTPRoute
from utilities.resources.reference_grant import ReferenceGrant

KUBE_RBAC_PROXY_PORT = 8443
KUBE_RBAC_PROXY_SERVICE_SUFFIX = "-kube-rbac-proxy"
GATEWAY_NAME = "data-science-gateway"
GATEWAY_NAMESPACE = "openshift-ingress"


def _assert_notebook_backend(route: HTTPRoute, notebook: Notebook) -> None:
    """Assert the HTTPRoute targets the correct kube-rbac-proxy service, namespace, port, and path."""
    expected_service = f"{notebook.name}{KUBE_RBAC_PROXY_SERVICE_SUFFIX}"
    expected_path = f"/notebook/{notebook.namespace}/{notebook.name}"

    for rule in route.instance.spec.get("rules", []):
        for backend_ref in rule.get("backendRefs", []):
            if (
                backend_ref.get("name") == expected_service
                and backend_ref.get("namespace") == notebook.namespace
                and backend_ref.get("kind", "Service") == "Service"
                and backend_ref.get("port") == KUBE_RBAC_PROXY_PORT
                and any(match.get("path", {}).get("value") == expected_path for match in rule.get("matches", []))
            ):
                return

    pytest.fail(
        f"HTTPRoute '{route.name}' does not target "
        f"Service '{notebook.namespace}/{expected_service}:{KUBE_RBAC_PROXY_PORT}' "
        f"with path '{expected_path}'."
    )


@pytest.mark.usefixtures("capture_notebook_baseline")
class TestPreUpgradeNotebookRouting:
    """Verify notebook routing resources exist before the platform upgrade.

    Steps:
        1. Verify the HTTPRoute for the notebook exists in the applications namespace.
        2. Verify the HTTPRoute references the data-science-gateway in openshift-ingress.
        3. Verify the HTTPRoute routes to the correct kube-rbac-proxy service, port, and path.
        4. Verify the ReferenceGrant exists in the notebook namespace.
    """

    @pytest.mark.pre_upgrade
    def test_httproute_exists_before_upgrade(
        self,
        upgrade_notebook_httproute: HTTPRoute,
    ) -> None:
        """Given a Notebook CR is created before upgrade,
        When the ODH controller reconciles routing,
        Then an HTTPRoute should exist for the notebook in the applications namespace.
        """
        assert upgrade_notebook_httproute.exists, (
            f"HTTPRoute '{upgrade_notebook_httproute.name}' does not exist "
            f"in namespace '{upgrade_notebook_httproute.namespace}'"
        )

    @pytest.mark.pre_upgrade
    def test_httproute_has_gateway_parent_before_upgrade(
        self,
        upgrade_notebook_httproute: HTTPRoute,
    ) -> None:
        """Given the notebook HTTPRoute exists,
        When inspecting its parentRefs,
        Then it should reference the data-science-gateway in the openshift-ingress namespace.
        """
        parent_refs = upgrade_notebook_httproute.instance.spec.get("parentRefs", [])
        assert parent_refs, f"HTTPRoute '{upgrade_notebook_httproute.name}' has no parentRefs"

        has_gateway = any(
            ref.get("name") == GATEWAY_NAME and ref.get("namespace") == GATEWAY_NAMESPACE for ref in parent_refs
        )
        assert has_gateway, (
            f"HTTPRoute '{upgrade_notebook_httproute.name}' does not reference "
            f"'{GATEWAY_NAMESPACE}/{GATEWAY_NAME}'. parentRefs: {parent_refs}"
        )

    @pytest.mark.pre_upgrade
    def test_httproute_backend_ref_before_upgrade(
        self,
        upgrade_notebook: Notebook,
        upgrade_notebook_httproute: HTTPRoute,
    ) -> None:
        """Given the notebook HTTPRoute exists,
        When inspecting its rules,
        Then it should route to the kube-rbac-proxy service on port 8443
        with the correct path prefix and cross-namespace reference.
        """
        _assert_notebook_backend(route=upgrade_notebook_httproute, notebook=upgrade_notebook)

    @pytest.mark.pre_upgrade
    def test_reference_grant_exists_before_upgrade(
        self,
        upgrade_notebook_reference_grant: ReferenceGrant,
    ) -> None:
        """Given a Notebook CR is created before upgrade,
        When the ODH controller reconciles routing,
        Then a ReferenceGrant should exist in the notebook namespace.
        """
        assert upgrade_notebook_reference_grant.exists, (
            f"ReferenceGrant '{upgrade_notebook_reference_grant.name}' does not exist "
            f"in namespace '{upgrade_notebook_reference_grant.namespace}'"
        )


class TestPostUpgradeNotebookRouting:
    """Verify notebook routing survived the platform upgrade.

    Steps:
        1. Verify HTTPRoute still exists and references the Gateway.
        2. Verify HTTPRoute spec was not modified (generation unchanged).
        3. Verify HTTPRoute still routes to the correct kube-rbac-proxy service, port, and path.
        4. Verify no duplicate HTTPRoutes exist for this notebook.
        5. Verify ReferenceGrant still exists in the notebook namespace.
    """

    @pytest.mark.post_upgrade
    def test_httproute_exists_after_upgrade(
        self,
        upgrade_notebook_httproute: HTTPRoute,
    ) -> None:
        """Given a notebook HTTPRoute existed before upgrade,
        When the upgrade completes,
        Then the HTTPRoute should still exist.
        """
        assert upgrade_notebook_httproute.exists, (
            f"HTTPRoute '{upgrade_notebook_httproute.name}' no longer exists after upgrade"
        )

    @pytest.mark.post_upgrade
    def test_httproute_has_gateway_parent_after_upgrade(
        self,
        upgrade_notebook_httproute: HTTPRoute,
    ) -> None:
        """Given a notebook HTTPRoute existed before upgrade,
        When the upgrade completes,
        Then it should still reference the data-science-gateway in the openshift-ingress namespace.
        """
        assert upgrade_notebook_httproute.exists, (
            f"HTTPRoute '{upgrade_notebook_httproute.name}' no longer exists after upgrade"
        )
        parent_refs = upgrade_notebook_httproute.instance.spec.get("parentRefs", [])
        assert parent_refs, f"HTTPRoute '{upgrade_notebook_httproute.name}' has no parentRefs after upgrade"

        has_gateway = any(
            ref.get("name") == GATEWAY_NAME and ref.get("namespace") == GATEWAY_NAMESPACE for ref in parent_refs
        )
        assert has_gateway, (
            f"HTTPRoute '{upgrade_notebook_httproute.name}' lost '{GATEWAY_NAMESPACE}/{GATEWAY_NAME}' parentRef "
            f"after upgrade. parentRefs: {parent_refs}"
        )

    @pytest.mark.post_upgrade
    def test_httproute_not_modified_after_upgrade(
        self,
        upgrade_notebook_httproute: HTTPRoute,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given a notebook HTTPRoute existed before upgrade,
        When the upgrade completes,
        Then the HTTPRoute generation should be unchanged.
        """
        assert upgrade_notebook_httproute.exists, (
            f"HTTPRoute '{upgrade_notebook_httproute.name}' no longer exists after upgrade"
        )
        current_generation = upgrade_notebook_httproute.instance.metadata.generation
        saved_generation = upgrade_notebook_baseline["httproute_generation"]

        assert current_generation == saved_generation, (
            f"HTTPRoute was modified during upgrade. "
            f"Pre-upgrade generation: {saved_generation}, "
            f"post-upgrade generation: {current_generation}"
        )

    @pytest.mark.post_upgrade
    def test_httproute_backend_ref_after_upgrade(
        self,
        upgrade_notebook: Notebook,
        upgrade_notebook_httproute: HTTPRoute,
    ) -> None:
        """Given a notebook HTTPRoute existed before upgrade,
        When the upgrade completes,
        Then it should still route to the kube-rbac-proxy service on port 8443
        with the correct path prefix and cross-namespace reference.
        """
        _assert_notebook_backend(route=upgrade_notebook_httproute, notebook=upgrade_notebook)

    @pytest.mark.post_upgrade
    def test_no_duplicate_httproutes_after_upgrade(
        self,
        admin_client: Any,
        upgrade_notebook: Notebook,
    ) -> None:
        """Given a notebook HTTPRoute existed before upgrade,
        When the upgrade completes,
        Then there should be exactly one HTTPRoute for this notebook.
        """
        apps_ns = py_config["applications_namespace"]
        all_httproutes = list(HTTPRoute.get(dyn_client=admin_client, namespace=apps_ns))

        matching_routes = [
            route
            for route in all_httproutes
            if route.instance.metadata.labels
            and route.instance.metadata.labels.get("notebook-name") == upgrade_notebook.name
            and route.instance.metadata.labels.get("notebook-namespace") == upgrade_notebook.namespace
        ]

        assert len(matching_routes) == 1, (
            f"Expected exactly 1 HTTPRoute for notebook '{upgrade_notebook.name}' "
            f"in namespace '{apps_ns}', found {len(matching_routes)}. "
            f"Routes: {[route.name for route in matching_routes]}"
        )

    @pytest.mark.post_upgrade
    def test_reference_grant_exists_after_upgrade(
        self,
        upgrade_notebook_reference_grant: ReferenceGrant,
    ) -> None:
        """Given a ReferenceGrant existed before upgrade,
        When the upgrade completes,
        Then the ReferenceGrant should still exist in the notebook namespace.
        """
        assert upgrade_notebook_reference_grant.exists, (
            f"ReferenceGrant '{upgrade_notebook_reference_grant.name}' no longer exists "
            f"in namespace '{upgrade_notebook_reference_grant.namespace}' after upgrade"
        )
