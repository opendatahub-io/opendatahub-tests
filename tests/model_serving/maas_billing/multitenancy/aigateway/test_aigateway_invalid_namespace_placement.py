import pytest
import structlog

from tests.model_serving.maas_billing.multitenancy.aigateway.utils import verify_aigateway_invalid_placement
from utilities.resources.aigateway import AIGateway

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest", "aigateway_infra_namespace")
class TestAIGatewayInvalidNamespacePlacement:
    """Tier3 tests for AIGateway invalid namespace placement (InvalidPlacement)."""

    @pytest.mark.tier3
    @pytest.mark.parametrize(
        "invalid_placement_aigateway",
        [
            pytest.param(
                ("test_cr_in_applications_namespace", "applications_namespace", None),
                id="test_cr_in_applications_namespace",
            ),
            pytest.param(
                ("test_cr_in_legacy_tenant_namespace", "legacy_tenant_namespace", None),
                id="test_cr_in_legacy_tenant_namespace",
            ),
            pytest.param(
                (
                    "test_tenant_namespace_equals_applications_namespace",
                    "infra_namespace",
                    "applications_namespace",
                ),
                id="test_tenant_namespace_equals_applications_namespace",
            ),
            pytest.param(
                (
                    "test_tenant_namespace_equals_legacy_namespace",
                    "infra_namespace",
                    "legacy_tenant_namespace",
                ),
                id="test_tenant_namespace_equals_legacy_namespace",
            ),
            pytest.param(
                ("test_tenant_namespace_equals_infra_namespace", "infra_namespace", "infra_namespace"),
                id="test_tenant_namespace_equals_infra_namespace",
            ),
        ],
        indirect=True,
    )
    def test_aigateway_invalid_namespace_placement_rejected(
        self,
        invalid_placement_aigateway: AIGateway,
    ) -> None:
        """Verify invalid namespace placement fails with InvalidPlacement."""
        verify_aigateway_invalid_placement(aigateway=invalid_placement_aigateway)
        LOGGER.info(
            f"AIGateway invalid namespace placement rejected as expected for '{invalid_placement_aigateway.name}'"
        )
