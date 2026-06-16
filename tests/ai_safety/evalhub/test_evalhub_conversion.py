"""EvalHub CRD conversion webhook tests (RHOAI 3.5EA2+).

The v1alpha1 <-> v1 conversion webhook was introduced in RHOAI 3.5EA2.
It is not available in RHOAI 2.25 or 3.4. These tests only apply to
3.x -> 3.x upgrades where the target version is 3.5EA2 or later.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.namespace import Namespace

from tests.ai_safety.evalhub.constants import (
    EVALHUB_API_GROUP,
    EVALHUB_PLURAL,
)
from tests.ai_safety.evalhub.utils import EvalHubV1, EvalHubV1Alpha1


@pytest.mark.smoke
@pytest.mark.ai_safety
def test_evalhub_crd_serves_both_versions(
    admin_client: DynamicClient,
) -> None:
    """Verify the EvalHub CRD advertises both v1alpha1 and v1 versions."""
    crd_name = f"{EVALHUB_PLURAL}.{EVALHUB_API_GROUP}"
    crd = CustomResourceDefinition(
        client=admin_client,
        name=crd_name,
        ensure_exists=True,
    )
    assert crd.exists, f"CRD {crd_name} not found"

    served_versions = {version["name"] for version in crd.instance.spec.versions if version.get("served", False)}
    assert "v1alpha1" in served_versions, f"v1alpha1 not served; served versions: {served_versions}"
    assert "v1" in served_versions, f"v1 not served; served versions: {served_versions}"


@pytest.mark.smoke
@pytest.mark.ai_safety
def test_evalhub_crd_conversion_strategy_is_webhook(
    admin_client: DynamicClient,
) -> None:
    """Verify the EvalHub CRD uses the Webhook conversion strategy."""
    crd_name = f"{EVALHUB_PLURAL}.{EVALHUB_API_GROUP}"
    crd = CustomResourceDefinition(
        client=admin_client,
        name=crd_name,
        ensure_exists=True,
    )
    assert crd.exists, f"CRD {crd_name} not found"

    conversion = crd.instance.spec.get("conversion", {})
    assert conversion.get("strategy") == "Webhook", (
        f"Expected conversion strategy 'Webhook', got '{conversion.get('strategy')}'"
    )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-conversion"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubCRDConversion:
    """Tests for the EvalHub v1alpha1 <-> v1 CRD conversion webhook."""

    def test_create_v1alpha1_read_as_v1(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Create an EvalHub via v1alpha1 and read it back as v1."""
        with EvalHubV1Alpha1(
            client=admin_client,
            name="conv-v1alpha1-to-v1",
            namespace=model_namespace.name,
            database={"type": "sqlite"},
            providers=["lm-evaluation-harness"],
            collections=["leaderboard-v2"],
        ) as evalhub:
            result = EvalHubV1(
                client=admin_client,
                name=evalhub.name,
                namespace=model_namespace.name,
                ensure_exists=True,
            )
            assert result.instance.spec.database.type == "sqlite"
            assert "lm-evaluation-harness" in result.instance.spec.providers
            assert "leaderboard-v2" in result.instance.spec.collections

    def test_create_v1_read_as_v1alpha1(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Create an EvalHub via v1 and read it back as v1alpha1."""
        with EvalHubV1(
            client=admin_client,
            name="conv-v1-to-v1alpha1",
            namespace=model_namespace.name,
            database={"type": "sqlite"},
            providers=["lm-evaluation-harness"],
            collections=["leaderboard-v2"],
        ) as evalhub:
            result = EvalHubV1Alpha1(
                client=admin_client,
                name=evalhub.name,
                namespace=model_namespace.name,
                ensure_exists=True,
            )
            assert result.instance.spec.database.type == "sqlite"
            assert "lm-evaluation-harness" in result.instance.spec.providers
            assert "leaderboard-v2" in result.instance.spec.collections

    def test_conversion_preserves_database_config(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Create an EvalHub with full database config in v1alpha1, read as v1, verify fields."""
        with EvalHubV1Alpha1(
            client=admin_client,
            name="conv-db-preserve",
            namespace=model_namespace.name,
            database={
                "type": "postgresql",
                "secret": "db-secret",  # pragma: allowlist secret
                "maxOpenConns": 50,
                "maxIdleConns": 10,
            },
            providers=["lm-evaluation-harness", "garak"],
            collections=["leaderboard-v2"],
            replicas=1,
        ) as evalhub:
            result = EvalHubV1(
                client=admin_client,
                name=evalhub.name,
                namespace=model_namespace.name,
                ensure_exists=True,
            )
            assert result.instance.spec.database.type == "postgresql"
            assert result.instance.spec.database.secret == "db-secret"  # pragma: allowlist secret
            assert result.instance.spec.database.maxOpenConns == 50
            assert result.instance.spec.database.maxIdleConns == 10
            assert set(result.instance.spec.providers) == {"lm-evaluation-harness", "garak"}

    def test_roundtrip_v1alpha1_to_v1_to_v1alpha1(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Verify roundtrip: create as v1alpha1, read as v1, update via v1, read back as v1alpha1."""
        with EvalHubV1Alpha1(
            client=admin_client,
            name="conv-roundtrip",
            namespace=model_namespace.name,
            database={"type": "sqlite"},
            providers=["lm-evaluation-harness"],
            collections=["leaderboard-v2"],
        ) as evalhub:
            v1_result = EvalHubV1(
                client=admin_client,
                name=evalhub.name,
                namespace=model_namespace.name,
                ensure_exists=True,
            )
            assert v1_result.instance.spec.database.type == "sqlite"

            roundtrip = EvalHubV1Alpha1(
                client=admin_client,
                name=evalhub.name,
                namespace=model_namespace.name,
                ensure_exists=True,
            )
            assert roundtrip.instance.spec.database.type == "sqlite"
            assert "lm-evaluation-harness" in roundtrip.instance.spec.providers
            assert "leaderboard-v2" in roundtrip.instance.spec.collections
