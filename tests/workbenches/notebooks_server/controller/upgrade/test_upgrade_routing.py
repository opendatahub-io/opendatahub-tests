from typing import Any

import pytest

from utilities.resources.http_route import HTTPRoute
from utilities.resources.reference_grant import ReferenceGrant


@pytest.mark.usefixtures("capture_notebook_baseline")
class TestPreUpgradeNotebookRouting:
    """Verify notebook routing resources exist before the platform upgrade.

    Steps:
        1. Verify the HTTPRoute for the notebook exists in the applications namespace.
        2. Verify the HTTPRoute references the correct Gateway parent.
        3. Verify the ReferenceGrant exists in the notebook namespace.
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
        Then it should reference the data-science-gateway.
        """
        parent_refs = upgrade_notebook_httproute.instance.spec.get("parentRefs", [])
        assert parent_refs, f"HTTPRoute '{upgrade_notebook_httproute.name}' has no parentRefs"

        has_gateway = any(ref.get("name") == "data-science-gateway" for ref in parent_refs)
        assert has_gateway, (
            f"HTTPRoute '{upgrade_notebook_httproute.name}' does not reference data-science-gateway. "
            f"parentRefs: {parent_refs}"
        )

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
        2. Verify HTTPRoute spec was not modified.
        3. Verify ReferenceGrant still exists in the notebook namespace.
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
        Then it should still reference the data-science-gateway.
        """
        parent_refs = upgrade_notebook_httproute.instance.spec.get("parentRefs", [])
        assert parent_refs, f"HTTPRoute '{upgrade_notebook_httproute.name}' has no parentRefs after upgrade"

        has_gateway = any(ref.get("name") == "data-science-gateway" for ref in parent_refs)
        assert has_gateway, (
            f"HTTPRoute '{upgrade_notebook_httproute.name}' lost data-science-gateway parentRef after upgrade. "
            f"parentRefs: {parent_refs}"
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
        current_generation = upgrade_notebook_httproute.instance.metadata.generation
        saved_generation = upgrade_notebook_baseline["httproute_generation"]

        assert current_generation == saved_generation, (
            f"HTTPRoute was modified during upgrade. "
            f"Pre-upgrade generation: {saved_generation}, "
            f"post-upgrade generation: {current_generation}"
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
