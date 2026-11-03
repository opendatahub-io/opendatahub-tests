"""E2E: one tenant's failing external provider must not degrade another tenant.

Ported from models-as-a-service ``test_tenant_external_model_failure.py``.
The attachment/shared modules cover happy-path multi-tenant external-model
serving plus per-tenant auth and quota isolation; this adds the failure-path
dimension: when inference through one tenant's gateway hits an upstream that
returns 5xx, another tenant's independent external model keeps serving
through its own gateway/ExtProc.

Each tenant gets its own single-gateway external model (the attachment
shape from test_tenant_external_model_attachment.py, so no shared-model
resolvedModelAlias dependency). The shared header-echo endpoint doubles as
the fault injector: httpbingo's ``/status/503`` returns that status, so
routing tenant A's request there makes its upstream error deterministically
while tenant B keeps using the healthy ``/anything`` echo path (the AGC
forwards whatever path follows the model name to the provider verbatim –
see ``external_chat_url``).
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import requests
import structlog

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    assert_served_by_provider,
    infer_for_served_model,
    redact_sensitive,
)

LOGGER = structlog.get_logger(name=__name__)

FAILING_UPSTREAM_PATH = os.environ.get("E2E_EXTERNAL_FAILURE_PATH", "/status/503")
FAILING_UPSTREAM_STATUS = int(os.environ.get("E2E_EXTERNAL_FAILURE_STATUS", "503"))


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestTenantProviderFailureIsolation:
    """A 5xx from one tenant's upstream must not degrade the other tenant."""

    @pytest.mark.tier2
    def test_upstream_failure_in_one_tenant_does_not_degrade_the_other(
        self,
        request_session_http: requests.Session,
        failure_env: dict[str, Any],
        failure_served: dict[str, dict[str, Any]],
    ) -> None:
        """Given two tenants each serving their own external model, when tenant
        A's upstream returns 5xx, then tenant B keeps serving healthy traffic
        through its own gateway with its own credential injected."""
        tenant_a, tenant_b = failure_env["tenant_a"], failure_env["tenant_b"]
        served_a, served_b = failure_served["tenant_a"], failure_served["tenant_b"]

        LOGGER.info("Baseline: asserting both tenant A and tenant B serve healthy traffic")
        assert_served_by_provider(
            response=infer_for_served_model(
                request_session_http=request_session_http, case=tenant_a, served_model=served_a
            ),
            path_tag=served_a["model"]["path_tag"],
            credential=served_a["model"]["credential"],
        )
        assert_served_by_provider(
            response=infer_for_served_model(
                request_session_http=request_session_http, case=tenant_b, served_model=served_b
            ),
            path_tag=served_b["model"]["path_tag"],
            credential=served_b["model"]["credential"],
        )

        LOGGER.info(f"Fault injection: routing tenant A to {FAILING_UPSTREAM_PATH} (expect {FAILING_UPSTREAM_STATUS})")
        failing = infer_for_served_model(
            request_session_http=request_session_http,
            case=tenant_a,
            served_model=served_a,
            provider_path=FAILING_UPSTREAM_PATH,
            retries=1,
        )
        assert failing.status_code == FAILING_UPSTREAM_STATUS, (
            f"expected tenant A's upstream {FAILING_UPSTREAM_STATUS} to propagate, got "
            f"{failing.status_code}: {redact_sensitive(failing.text)}"
        )

        LOGGER.info("Isolation: asserting tenant B still serves despite tenant A's 503")
        assert_served_by_provider(
            response=infer_for_served_model(
                request_session_http=request_session_http, case=tenant_b, served_model=served_b, retries=1
            ),
            path_tag=served_b["model"]["path_tag"],
            credential=served_b["model"]["credential"],
        )
