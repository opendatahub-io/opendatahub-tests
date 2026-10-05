import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace

from tests.model_serving.maas_billing.upgrade.utils import (
    verify_maas_auth_policy_exists,
    verify_maas_model_ref_exists,
    verify_maas_subscription_ready,
    wait_for_legacy_maas_networking_present,
)
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
        4. Capture baseline for post-upgrade assertions on 3.5+.
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
