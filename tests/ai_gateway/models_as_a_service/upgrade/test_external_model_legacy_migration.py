import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace

from tests.ai_gateway.models_as_a_service.upgrade.utils import (
    LegacyMigrationBaseline,
    verify_inference_external_model_programmed,
    verify_maas_auth_policy_exists,
    verify_maas_model_ref_exists,
    verify_maas_subscription_ready,
    verify_no_legacy_owned_httproutes,
    wait_for_inference_external_model_httproute,
    wait_for_legacy_maas_networking_present,
    wait_for_legacy_maas_prefixed_networking_deleted,
)
from utilities.resources.external_model import ExternalModel
from utilities.resources.legacy_external_model import LegacyExternalModel

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "capture_legacy_migration_baseline_fixture",
)
@pytest.mark.pre_upgrade
class TestPreUpgradeLegacyExternalModelMigration:
    """Deploy legacy ExternalModel networking state before upgrade.

    Steps:
        1. Verify legacy maas.opendatahub.io ExternalModel exists.
        2. Verify maas-controller created maas-* networking children.
        3. Verify MaaSModelRef, auth policy, and subscription exist.
        4. Capture baseline for post-upgrade assertions.
    """

    def test_legacy_external_model_created(
        self,
        legacy_migration_external_model: LegacyExternalModel,
    ) -> None:
        """Given pre-upgrade cluster, when legacy ExternalModel is deployed, then the CR exists."""
        assert legacy_migration_external_model.exists, (
            f"Legacy ExternalModel '{legacy_migration_external_model.name}' was not created"
        )

    def test_legacy_maas_networking_present(
        self,
        admin_client: DynamicClient,
        legacy_migration_external_model: LegacyExternalModel,
        legacy_migration_namespace: Namespace,
    ) -> None:
        """Given a reconciled legacy ExternalModel, when checking networking, then legacy children exist."""
        resource_name = wait_for_legacy_maas_networking_present(
            client=admin_client,
            model_name=legacy_migration_external_model.name,
            namespace=legacy_migration_namespace.name,
        )
        LOGGER.info(f"Legacy networking present for '{resource_name}'")

    def test_legacy_maas_model_ref_created(
        self,
        legacy_migration_model_ref: MaaSModelRef,
    ) -> None:
        """Given a legacy ExternalModel, when checking MaaSModelRef, then a ref to the model exists."""
        verify_maas_model_ref_exists(model_ref=legacy_migration_model_ref)

    def test_legacy_maas_auth_policy_created(
        self,
        legacy_migration_auth_policy: MaaSAuthPolicy,
    ) -> None:
        """Given a legacy external model stack, when checking auth policy, then it exists."""
        verify_maas_auth_policy_exists(auth_policy=legacy_migration_auth_policy)

    def test_legacy_maas_subscription_ready(
        self,
        legacy_migration_subscription: MaaSSubscription,
    ) -> None:
        """Given a legacy external model stack, when checking subscription, then it exists."""
        verify_maas_subscription_ready(subscription=legacy_migration_subscription)


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "legacy_migration_credential_secret",
    "legacy_migration_external_model",
    "capture_legacy_migration_baseline_fixture",
)
@pytest.mark.post_upgrade
class TestPostUpgradeLegacyExternalModelMigration:
    """Validate legacy ExternalModel supersede behavior after upgrade.

    Steps:
        1. Verify inference ExternalModel is programmed.
        2. Verify legacy maas-* networking children are removed.
        3. Verify inference HTTPRoute remains available.
        4. Verify MaaSModelRef, auth policy, and subscription survived upgrade.
    """

    def test_inference_external_model_programmed(
        self,
        require_inference_external_model_crd: None,
        legacy_migration_inference_external_model: ExternalModel,
    ) -> None:
        """Given upgrade completed, when migration creates inference ExternalModel, then its route is programmed."""
        http_route_name = verify_inference_external_model_programmed(
            external_model=legacy_migration_inference_external_model,
        )
        LOGGER.info(f"Inference ExternalModel programmed HTTPRoute '{http_route_name}'")

    @pytest.mark.dependency(name="test_legacy_maas_networking_removed")
    def test_legacy_maas_networking_removed(
        self,
        admin_client: DynamicClient,
        legacy_migration_namespace: Namespace,
        legacy_migration_baseline_fixture: LegacyMigrationBaseline,
    ) -> None:
        """Given inference EM is programmed, then maas-prefixed legacy networking is removed."""
        wait_for_legacy_maas_prefixed_networking_deleted(
            client=admin_client,
            resource_name=legacy_migration_baseline_fixture["legacy_resource_name"],
            namespace=legacy_migration_namespace.name,
        )
        LOGGER.info(
            f"Legacy maas-prefixed networking removed for '{legacy_migration_baseline_fixture['legacy_resource_name']}'"
        )

    def test_inference_httproute_present(
        self,
        require_inference_external_model_crd: None,
        admin_client: DynamicClient,
        legacy_migration_namespace: Namespace,
        legacy_migration_inference_external_model: ExternalModel,
    ) -> None:
        """Given inference ExternalModel is programmed, when checking HTTPRoutes, then the inference route exists."""
        wait_for_inference_external_model_httproute(
            client=admin_client,
            external_model=legacy_migration_inference_external_model,
            namespace=legacy_migration_namespace.name,
        )

    def test_legacy_maas_model_ref_survives_upgrade(
        self,
        legacy_migration_model_ref: MaaSModelRef,
    ) -> None:
        """Given upgrade completed, when checking MaaSModelRef, then the legacy migration ref still exists."""
        verify_maas_model_ref_exists(model_ref=legacy_migration_model_ref)

    def test_legacy_maas_auth_policy_survives_upgrade(
        self,
        legacy_migration_auth_policy: MaaSAuthPolicy,
    ) -> None:
        """Given upgrade completed, when checking auth policy, then the legacy migration policy still exists."""
        verify_maas_auth_policy_exists(auth_policy=legacy_migration_auth_policy)

    def test_legacy_maas_subscription_survives_upgrade(
        self,
        legacy_migration_subscription: MaaSSubscription,
    ) -> None:
        """Given upgrade completed, when checking subscription, then the legacy migration subscription still exists."""
        verify_maas_subscription_ready(subscription=legacy_migration_subscription)

    @pytest.mark.dependency(depends=["test_legacy_maas_networking_removed"])
    def test_no_duplicate_legacy_httproute(
        self,
        admin_client: DynamicClient,
        legacy_migration_namespace: Namespace,
        legacy_migration_baseline_fixture: LegacyMigrationBaseline,
    ) -> None:
        """Given supersede completed, when checking HTTPRoutes, then none are owned by the legacy ExternalModel."""
        wait_for_legacy_maas_prefixed_networking_deleted(
            client=admin_client,
            resource_name=legacy_migration_baseline_fixture["legacy_resource_name"],
            namespace=legacy_migration_namespace.name,
        )
        verify_no_legacy_owned_httproutes(
            client=admin_client,
            namespace=legacy_migration_namespace.name,
            model_name=legacy_migration_baseline_fixture["model_name"],
        )
