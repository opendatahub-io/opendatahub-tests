import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.ai_safety.evalhub.constants import (
    EVALHUB_API_GROUP,
    EVALHUB_PLURAL,
)
from tests.ai_safety.evalhub.utils import EvalHubV1, EvalHubV1Alpha1, validate_evalhub_health


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-upgrade"},
        ),
    ],
    indirect=True,
)
@pytest.mark.ai_safety
class TestPreUpgradeEvalHub:
    """Pre-upgrade tests: deploy EvalHub and verify it works before the platform upgrade.

    EvalHub was introduced with v1alpha1 and gained v1 (storage version) in 3.5EA2.
    The conversion webhook is only available from 3.5EA2 onwards (not in 2.25 or 3.4),
    so these tests only apply to 3.x -> 3.x upgrades where both source and target are 3.5EA2+.
    """

    @pytest.mark.pre_upgrade
    def test_evalhub_pre_upgrade_health(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Verify EvalHub health endpoint responds before upgrade."""
        validate_evalhub_health(
            host=evalhub_route.host,
            token=current_client_token,
            ca_bundle_file=evalhub_ca_bundle_file,
        )

    @pytest.mark.pre_upgrade
    def test_evalhub_pre_upgrade_crd_versions(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Verify the EvalHub CRD serves both v1alpha1 and v1 before upgrade."""
        crd_name = f"{EVALHUB_PLURAL}.{EVALHUB_API_GROUP}"
        crd = CustomResourceDefinition(
            client=admin_client,
            name=crd_name,
            ensure_exists=True,
        )
        assert crd.exists, f"CRD {crd_name} not found"

        served_versions = {version["name"] for version in crd.instance.spec.versions if version.get("served", False)}
        assert "v1alpha1" in served_versions
        assert "v1" in served_versions

    @pytest.mark.pre_upgrade
    def test_evalhub_pre_upgrade_conversion_works(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
    ) -> None:
        """Verify the conversion webhook works before upgrade by reading the CR as v1."""
        result = EvalHubV1(
            client=admin_client,
            name=evalhub_cr.name,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        assert result.instance.spec.database.type == "sqlite"


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-upgrade"},
        ),
    ],
    indirect=True,
)
@pytest.mark.ai_safety
class TestPostUpgradeEvalHub:
    """Post-upgrade tests: verify EvalHub survived the platform upgrade.

    Validates that:
    - The CRD still serves both API versions
    - The conversion webhook still functions
    - The EvalHub deployment is healthy
    - Pre-existing CR status fields survived
    """

    @pytest.mark.post_upgrade
    def test_evalhub_post_upgrade_crd_versions(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Verify the EvalHub CRD still serves both v1alpha1 and v1 after upgrade."""
        crd_name = f"{EVALHUB_PLURAL}.{EVALHUB_API_GROUP}"
        crd = CustomResourceDefinition(
            client=admin_client,
            name=crd_name,
            ensure_exists=True,
        )
        assert crd.exists, f"CRD {crd_name} not found after upgrade"

        served_versions = {version["name"] for version in crd.instance.spec.versions if version.get("served", False)}
        assert "v1alpha1" in served_versions, f"v1alpha1 no longer served after upgrade; versions: {served_versions}"
        assert "v1" in served_versions, f"v1 no longer served after upgrade; versions: {served_versions}"

    @pytest.mark.post_upgrade
    def test_evalhub_post_upgrade_health(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Verify EvalHub health endpoint still responds after upgrade."""
        validate_evalhub_health(
            host=evalhub_route.host,
            token=current_client_token,
            ca_bundle_file=evalhub_ca_bundle_file,
        )

    @pytest.mark.post_upgrade
    def test_evalhub_post_upgrade_cr_status_survived(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
    ) -> None:
        """Verify the EvalHub CR status fields survived the upgrade."""
        result = EvalHubV1(
            client=admin_client,
            name=evalhub_cr.name,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        assert result.instance.status, "EvalHub status is empty after upgrade"
        assert result.instance.status.phase, "EvalHub phase is empty after upgrade"

    @pytest.mark.post_upgrade
    def test_evalhub_post_upgrade_conversion_still_works(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
    ) -> None:
        """Verify the conversion webhook still functions after upgrade.

        Read the pre-existing CR via both API versions to confirm
        the webhook is still converting correctly.
        """
        v1_obj = EvalHubV1(
            client=admin_client,
            name=evalhub_cr.name,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        assert v1_obj.instance.spec.database.type == "sqlite"

        v1alpha1_obj = EvalHubV1Alpha1(
            client=admin_client,
            name=evalhub_cr.name,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        assert v1alpha1_obj.instance.spec.database.type == "sqlite"

    @pytest.mark.post_upgrade
    def test_evalhub_post_upgrade_deployment_available(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_cr: EvalHub,
    ) -> None:
        """Verify the EvalHub deployment is still available after upgrade."""
        deployment = Deployment(
            client=admin_client,
            name=evalhub_cr.name,
            namespace=model_namespace.name,
        )
        assert deployment.exists, (
            f"EvalHub deployment '{evalhub_cr.name}' not found in namespace '{model_namespace.name}' after upgrade"
        )

        available = any(
            condition.type == "Available" and condition.status == "True"
            for condition in (deployment.instance.status.conditions or [])
        )
        assert available, "EvalHub deployment is not Available after upgrade"
