from __future__ import annotations

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.service_account import ServiceAccount

from tests.ai_gateway.models_as_a_service.maas_subscription.utils import (
    ModelIdentityCollisionNames,
    assert_model_identity_collision_detected_and_resolved,
    assert_model_identity_collision_preserves_runtime_ready,
)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
)
class TestModelIdentityCollision:
    """Verify MaaSModelRef model-identity collision detection and recovery."""

    @pytest.mark.tier1
    def test_colliding_model_names_flagged_then_resolved(
        self,
        admin_client: DynamicClient,
        maas_unprivileged_model_namespace: Namespace,
        maas_model_service_account: ServiceAccount,
        model_identity_collision_names: ModelIdentityCollisionNames,
    ) -> None:
        """Given two MaaSModelRefs whose LLMIS share spec.model.name, when both exist,
        then ModelIdentityUnique is False and conflict events fire; when one is removed,
        then the survivor reports unique again.
        """
        assert_model_identity_collision_detected_and_resolved(
            admin_client=admin_client,
            namespace=maas_unprivileged_model_namespace.name,
            service_account=maas_model_service_account.name,
            collision_names=model_identity_collision_names,
        )

    @pytest.mark.tier1
    def test_collision_does_not_break_runtime_ready(
        self,
        admin_client: DynamicClient,
        maas_unprivileged_model_namespace: Namespace,
        maas_model_service_account: ServiceAccount,
        model_identity_collision_names: ModelIdentityCollisionNames,
    ) -> None:
        """Given two colliding MaaSModelRefs, when ModelIdentityUnique is False,
        then RuntimeReady remains True on both refs.
        """
        assert_model_identity_collision_preserves_runtime_ready(
            admin_client=admin_client,
            namespace=maas_unprivileged_model_namespace.name,
            service_account=maas_model_service_account.name,
            collision_names=model_identity_collision_names,
        )

    @pytest.mark.tier2
    def test_delete_primary_clears_collision_on_secondary(
        self,
        admin_client: DynamicClient,
        maas_unprivileged_model_namespace: Namespace,
        maas_model_service_account: ServiceAccount,
        model_identity_collision_names: ModelIdentityCollisionNames,
    ) -> None:
        """Given a model-identity collision, when the primary MaaSModelRef is removed,
        then the secondary recovers to ModelIdentityUnique=True and emits ModelNameConflictResolved.
        """
        assert_model_identity_collision_detected_and_resolved(
            admin_client=admin_client,
            namespace=maas_unprivileged_model_namespace.name,
            service_account=maas_model_service_account.name,
            collision_names=model_identity_collision_names,
            survivor="secondary",
        )
