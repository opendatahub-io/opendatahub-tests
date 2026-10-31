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
forwards whatever path follows the model name to the provider verbatim —
see ``external_chat_url``).
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    admitted_user_namespace,
    assert_served_by_provider,
    external_model_stack,
    gateway_ref,
    infer_external_chat,
    namespace_name,
    redact_sensitive,
    serve_external_model_in_tenant,
)

LOGGER = structlog.get_logger(name=__name__)

# httpbin-compatible endpoints return /status/<code> as that status, so routing
# tenant A here fails its upstream deterministically. Override these if
# E2E_EXTERNAL_ENDPOINT does not serve /status/503.
FAILING_UPSTREAM_PATH = os.environ.get("E2E_EXTERNAL_FAILURE_PATH", "/status/503")
FAILING_UPSTREAM_STATUS = int(os.environ.get("E2E_EXTERNAL_FAILURE_STATUS", "503"))

# Serialized with the rest of the tenant-external-model suite (shared tenant
# gateways / IPP coexistence).
pytestmark = pytest.mark.xdist_group(name="tenant_ipp")


@pytest.fixture(scope="module")
def failure_env(
    admin_client: DynamicClient,
    tenant_external_model_tenants: dict[str, dict[str, str]],
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """Shared environment for the failure-isolation scenario.

    One user namespace admitted to both tenants' gateways; models and
    governance are created per class under one ExitStack so everything is
    cleaned up at module teardown.
    """
    with ExitStack() as stack:
        user_ns = stack.enter_context(
            cm=admitted_user_namespace(
                admin_client,
                "e2e-extmod-fail-user",
                tenant_external_model_tenants["tenant_a"]["gateway_name"],
                tenant_external_model_tenants["tenant_b"]["gateway_name"],
                teardown=teardown_resources,
            )
        )
        yield {
            "tenant_a": tenant_external_model_tenants["tenant_a"],
            "tenant_b": tenant_external_model_tenants["tenant_b"],
            "user_ns": user_ns,
            "stack": stack,
        }


def _provision_and_serve(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token: str,
    env: dict[str, Any],
    case: dict[str, str],
    teardown: bool,
) -> dict[str, Any]:
    """Provision one external model attached to this tenant's gateway and serve it."""
    model = env["stack"].enter_context(
        cm=external_model_stack(
            admin_client=admin_client,
            name=f"e2e-fail-{case['tenant_label_name']}",
            namespace=namespace_name(env["user_ns"]),
            gateway_refs=[gateway_ref(gateway_name=case["gateway_name"])],
            teardown=teardown,
        )
    )
    access = env["stack"].enter_context(
        cm=serve_external_model_in_tenant(
            admin_client=admin_client,
            request_session_http=request_session_http,
            ocp_token=ocp_token,
            tenant=case,
            model_name=model["name"],
            model_namespace=model["namespace"],
            policy_name=f"{model['name']}-access",
            subscription_name=f"{model['name']}-sub",
            teardown=teardown,
        )
    )
    return {"model": model, "api_key": access["api_key"]}


class TestTenantProviderFailureIsolation:
    """A 5xx from one tenant's upstream must not degrade the other tenant."""

    @pytest.fixture(scope="class")
    def served(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        failure_env: dict[str, Any],
        teardown_resources: bool,
    ) -> dict[str, dict[str, Any]]:
        """Each tenant gets its own independent external model, served on its gateway."""
        return {
            "a": _provision_and_serve(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                env=failure_env,
                case=failure_env["tenant_a"],
                teardown=teardown_resources,
            ),
            "b": _provision_and_serve(
                admin_client=admin_client,
                request_session_http=request_session_http,
                ocp_token=ocp_token_for_actor,
                env=failure_env,
                case=failure_env["tenant_b"],
                teardown=teardown_resources,
            ),
        }

    @pytest.mark.tier2
    def test_upstream_failure_in_one_tenant_does_not_degrade_the_other(
        self,
        failure_env: dict[str, Any],
        served: dict[str, dict[str, Any]],
    ) -> None:
        """Given two tenants each serving their own external model, when tenant
        A's upstream returns 5xx, then tenant B keeps serving healthy traffic
        through its own gateway with its own credential injected."""
        tenant_a, tenant_b = failure_env["tenant_a"], failure_env["tenant_b"]
        model_a, model_b = served["a"], served["b"]

        def serve(case: dict[str, str], served_model: dict[str, Any], *, retries: int = 6) -> requests.Response:
            return infer_external_chat(
                gateway_url=case["gateway_url"],
                model_namespace=served_model["model"]["namespace"],
                model_client_name=served_model["model"]["name"],
                api_key=served_model["api_key"],
                path_tag=served_model["model"]["path_tag"],
                retries=retries,
            )

        LOGGER.info("Baseline: asserting both tenant A and tenant B serve healthy traffic")
        assert_served_by_provider(
            response=serve(tenant_a, model_a),
            path_tag=model_a["model"]["path_tag"],
            credential=model_a["model"]["credential"],
        )
        assert_served_by_provider(
            response=serve(tenant_b, model_b),
            path_tag=model_b["model"]["path_tag"],
            credential=model_b["model"]["credential"],
        )

        LOGGER.info(f"Fault injection: routing tenant A to {FAILING_UPSTREAM_PATH} (expect {FAILING_UPSTREAM_STATUS})")
        failing = infer_external_chat(
            gateway_url=tenant_a["gateway_url"],
            model_namespace=model_a["model"]["namespace"],
            model_client_name=model_a["model"]["name"],
            api_key=model_a["api_key"],
            path_tag=model_a["model"]["path_tag"],
            provider_path=FAILING_UPSTREAM_PATH,
            retries=1,
        )
        assert failing.status_code == FAILING_UPSTREAM_STATUS, (
            f"expected tenant A's upstream {FAILING_UPSTREAM_STATUS} to propagate, got "
            f"{failing.status_code}: {redact_sensitive(failing.text)}"
        )

        LOGGER.info("Isolation: asserting tenant B still serves despite tenant A's 503")
        assert_served_by_provider(
            response=serve(tenant_b, model_b, retries=1),
            path_tag=model_b["model"]["path_tag"],
            credential=model_b["model"]["credential"],
        )
