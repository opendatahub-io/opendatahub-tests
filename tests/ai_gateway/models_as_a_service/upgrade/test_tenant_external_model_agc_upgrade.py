"""AGC upgrade test: legacy default-tenant preservation (RHOAIENG-90809).

Narrowed scope, per the 3.6 test plan: default-tenant / legacy-shaped
objects (ExternalModels without gatewayRefs, served through the default
tenant's Gateway as in 3.5) must survive an AGC upgrade byte-identically —
same HTTPRoutes (uid / generation / labels / spec), same provider
DestinationRules (uids), no object recreation — with serving, authorization
and deletion behavior preserved. The 3.5→3.6 product upgrade path is
covered by the legacy-migration suite in this directory; runtime no-op and
coexistence behaviors are covered by the multitenancy external model suite.
"""

from __future__ import annotations

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    assert_served_by_provider,
    destination_rules_referencing,
    infer_external_chat,
    wait_for_gateway_attachment_ready,
    wait_for_no_attachment_httproute,
    wait_for_not_found,
)
from tests.ai_gateway.models_as_a_service.upgrade.utils import (
    AGCUpgradeBaseline,
    verify_agc_default_tenant_objects_unchanged,
)
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE
from utilities.resources.external_model import ExternalModel

AGC_UPGRADE_UNKNOWN_KEY = "sk-agc-upgrade-unknown-key"  # pragma: allowlist secret


@pytest.mark.pre_upgrade
@pytest.mark.usefixtures("capture_agc_upgrade_baseline")
class TestPreUpgradeAGCExternalModel:
    """Pre-upgrade: deploy and verify the legacy-shaped default-tenant model."""

    def test_model_serving_before_upgrade(
        self,
        agc_upgrade_external_model: dict[str, Any],
        agc_upgrade_api_key: Any,
        maas_host: str,
    ) -> None:
        """Given a legacy-shaped ExternalModel without gatewayRefs, when the
        default tenant serves it before the upgrade, then inference through
        the default Gateway succeeds with the model's credential injected."""
        response = infer_external_chat(
            gateway_url=f"https://{maas_host}",
            model_namespace=agc_upgrade_external_model["namespace"],
            model_client_name=agc_upgrade_external_model["client_name"],
            api_key=str(agc_upgrade_api_key),
            path_tag=agc_upgrade_external_model["path_tag"],
        )
        assert_served_by_provider(
            response=response,
            path_tag=agc_upgrade_external_model["path_tag"],
            credential=agc_upgrade_external_model["credential"],
        )


@pytest.mark.post_upgrade
@pytest.mark.order("first")
class TestPostUpgradeAGCExternalModel:
    """Post-upgrade: the legacy default-tenant objects are unchanged and served."""

    def test_model_objects_unchanged_after_upgrade(
        self,
        admin_client: DynamicClient,
        agc_upgrade_baseline: AGCUpgradeBaseline,
    ) -> None:
        """Given the pre-upgrade baseline snapshot, when AGC has been upgraded,
        then the model's HTTPRoutes, DestinationRules and published route name
        are byte-identical (no recreation, no rewrite)."""
        verify_agc_default_tenant_objects_unchanged(
            client=admin_client,
            baseline=agc_upgrade_baseline,
        )

    def test_model_serving_after_upgrade(
        self,
        admin_client: DynamicClient,
        agc_upgrade_external_model: dict[str, Any],
        agc_upgrade_api_key: Any,
        maas_host: str,
    ) -> None:
        """Given the upgraded AGC, when inference runs through the default
        Gateway with the pre-upgrade API key, then the request still succeeds
        with the same credential injected."""
        wait_for_gateway_attachment_ready(
            admin_client=admin_client,
            model_name=agc_upgrade_external_model["name"],
            model_namespace=agc_upgrade_external_model["namespace"],
            gateway_name=MAAS_GATEWAY_NAME,
        )
        response = infer_external_chat(
            gateway_url=f"https://{maas_host}",
            model_namespace=agc_upgrade_external_model["namespace"],
            model_client_name=agc_upgrade_external_model["client_name"],
            api_key=str(agc_upgrade_api_key),
            path_tag=agc_upgrade_external_model["path_tag"],
        )
        assert_served_by_provider(
            response=response,
            path_tag=agc_upgrade_external_model["path_tag"],
            credential=agc_upgrade_external_model["credential"],
        )

    def test_unknown_api_key_rejected_after_upgrade(
        self,
        agc_upgrade_external_model: dict[str, Any],
        maas_host: str,
    ) -> None:
        """Given the upgraded AGC, when inference runs with an unknown API key,
        then the gateway still rejects it (authorization preserved)."""
        response = infer_external_chat(
            gateway_url=f"https://{maas_host}",
            model_namespace=agc_upgrade_external_model["namespace"],
            model_client_name=agc_upgrade_external_model["client_name"],
            api_key=AGC_UPGRADE_UNKNOWN_KEY,
            path_tag=agc_upgrade_external_model["path_tag"],
            retries=1,
        )
        assert response.status_code in (401, 403), (
            f"unknown API key was not rejected after the AGC upgrade: {response.status_code}"
        )

    def test_model_deletion_cleanup_after_upgrade(
        self,
        admin_client: DynamicClient,
        agc_upgrade_external_model: dict[str, Any],
    ) -> None:
        """Given the upgraded AGC, when the model is deleted, then its
        attachment route and provider DestinationRules are removed (deletion
        behavior preserved). This also serves as the suite's cleanup."""
        model_name = agc_upgrade_external_model["client_name"]
        model_namespace = agc_upgrade_external_model["namespace"]
        external_model = ExternalModel(
            client=admin_client,
            name=agc_upgrade_external_model["name"],
            namespace=model_namespace,
            ensure_exists=True,
        )
        assert external_model.exists, (
            f"ExternalModel {model_namespace}/{agc_upgrade_external_model['name']} not found after upgrade"
        )
        external_model.clean_up()
        wait_for_not_found(
            client=admin_client,
            kind="externalmodel",
            name=agc_upgrade_external_model["name"],
            namespace=model_namespace,
        )
        wait_for_no_attachment_httproute(
            admin_client=admin_client,
            route_namespace=model_namespace,
            gateway_name=MAAS_GATEWAY_NAME,
            model_name=model_name,
        )
        assert not destination_rules_referencing(
            admin_client,
            agc_upgrade_external_model["name"],
            MAAS_GATEWAY_NAMESPACE,
        ), "provider DestinationRules survived the model deletion after the AGC upgrade"
