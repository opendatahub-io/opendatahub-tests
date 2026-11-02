"""E2E: external models outside tenant namespaces, served by the default tenant.

Ported from models-as-a-service
``test/e2e/tests/test_tenant_external_model_default_tenant.py``
(RHOAIENG-101754, 3.5 → 3.6 compatibility). In 3.5 the default tenant served
ExternalModels in every namespace. With ai-gateway-controller (AGC) on the
Praxis dataplane, an ExternalModel without gatewayRefs in a namespace that has
no MaasTenantConfig and that no AITenant claims is served by the default
tenant (AITenant models-as-a-service) as an implicit attachment to its
Gateway: the same hashed HTTPRoute, DestinationRule and credential copy as an
explicit gatewayRefs attachment, with spec unchanged.

The models live in the llm namespace: the namespace 3.5 demos used, and one
the default Gateway's allowedRoutes already admits (a fresh user namespace
would need the default Gateway patched). The stack is 3.5-shaped: the provider
uses ``auth.type: simple``, which the AGC serves as ``apikey``. The provider
endpoint is a header-echo service (default httpbingo.org): under
``/anything/<tag>`` it echoes the request, so a response proves which binding
served it and which credential was injected.

Scenarios:
- The model and its provider reach Ready and serve inference through the
  default Gateway, with the credential delivered as a copy into the default
  tenant namespace.
- gatewayRefs replace the implicit attachment and remove its objects, without
  falling back to the default Gateway; removing them restores it.
- Deleting the model removes the implicit attachment's objects.

Skips unless the default tenant runs praxis-extproc (installed by the AGC): a
default tenant on IPP serves these models itself and the AGC leaves them
alone.
"""

from __future__ import annotations

import re
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.namespace import Namespace
from ocp_resources.project_project_openshift_io import Project

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    PRAXIS_IMAGE_MARKER,
    READY_TIMEOUT,
    default_tenant_model_stack,
    deployment_image,
    gateway_ref,
    gateway_status_entry,
    get_json_or_none,
    infer_chat_completion,
    list_json,
    namespace_name,
    patch_external_model_gateway_refs,
    redact_sensitive,
    serve_in_default_tenant,
    wait_for_gateway_attachment_reason,
    wait_for_not_found,
    wait_implicitly_attached,
    wait_not_implicitly_attached,
    wait_until,
)
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE
from utilities.general import generate_random_name
from utilities.infra import create_ns
from utilities.resources.external_model import ExternalModel

LOGGER = structlog.get_logger(name=__name__)

# Serialized with the tenant_ipp-flavored MaaS suites, which run legacy IPP
# tenant stacks: those must not run next to this suite's models.
pytestmark = pytest.mark.xdist_group(name="tenant_ipp")

# The namespace 3.5 demos used; the default Gateway's allowedRoutes admits it.
MODEL_NAMESPACE = "llm"

# The default tenant (MaaS subscription namespace) serves these models.

# Wait for AGC reconciliation (same env override as the shared suite timeout).

# A Gateway no AITenant manages (it need not exist): an explicit binding to it
# is refused with GatewayNotManaged.
UNMANAGED_GATEWAY_NAME = "e2e-dft-unmanaged-gateway"


@pytest.fixture(scope="module", autouse=True)
def _require_praxis_default_tenant(
    admin_client: DynamicClient,
    tenant_external_model_capabilities: dict[str, bool],
) -> None:
    """Skip unless the default tenant's payload-processing runs praxis-extproc.

    Also gates on the installed AGC + ExternalModel CRD capabilities: without
    the gatewayRefs schema these tests would patch it and wait for a
    GatewayNotManaged reason that never appears (timeout, not skip).
    """
    deployment = get_json_or_none(
        client=admin_client, kind="deployment", name="payload-processing", namespace=MAAS_GATEWAY_NAMESPACE
    )
    if PRAXIS_IMAGE_MARKER not in deployment_image(deployment):
        pytest.skip(
            "default tenant payload-processing is not praxis-extproc; its IPP serves "
            "models without gatewayRefs and the AGC leaves them alone"
        )


@pytest.fixture(scope="module")
def default_tenant_model_namespace(
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace | Project, Any, Any]:
    """The llm namespace the default Gateway's allowedRoutes already admits."""
    existing = get_json_or_none(client=admin_client, kind="namespace", name=MODEL_NAMESPACE)
    if existing is not None:
        # Left over from a previous run (fixed name): adopt it without
        # ownership so cleanup never deletes a namespace this run did not
        # create. A namespace claimed by a MaasTenantConfig cannot receive
        # implicit default-tenant attachment (its models belong to that
        # tenant's gateway), so adopting it would only time out.
        claimed = list_json(
            client=admin_client, kind="maastenantconfig", namespace=MODEL_NAMESPACE
        )
        if claimed:
            pytest.skip(
                f"namespace '{MODEL_NAMESPACE}' is claimed by a MaasTenantConfig "
                f"({', '.join(item['metadata']['name'] for item in claimed)}); "
                "the default Gateway will not implicitly attach its models"
            )
        yield Namespace(client=admin_client, name=MODEL_NAMESPACE)
        return
    with create_ns(
        admin_client=admin_client,
        name=MODEL_NAMESPACE,
        teardown=teardown_resources,
    ) as namespace:
        yield namespace


# ─── 3.5-shaped stack ───────────────────────────────────────────────────────


# ─── Attachment state ───────────────────────────────────────────────────────


# ─── Serving through the default tenant ────────────────────────────────────


# ─── Serving through the default Gateway ────────────────────────────────────


class TestDefaultTenantServesModelWithoutGatewayRefs:
    """A 3.5-shaped model in a namespace of no tenant is served through the
    default tenant's Gateway."""

    @pytest.fixture(scope="class")
    def implicit_model(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        base_url: str,
        default_tenant_model_namespace: Namespace,
    ) -> Generator[dict[str, Any], Any, Any]:
        """A 3.5-shaped model implicitly attached to the default Gateway and served by the default tenant."""
        with ExitStack() as stack:
            model = stack.enter_context(
                cm=default_tenant_model_stack(
                    admin_client,
                    prefix="e2e-dft",
                    namespace=namespace_name(default_tenant_model_namespace),
                )
            )
            attachment = wait_implicitly_attached(admin_client=admin_client, model=model)
            api_key = stack.enter_context(
                cm=serve_in_default_tenant(admin_client, request_session_http, ocp_token_for_actor, base_url, model)
            )
            yield {"model": model, "api_key": api_key, **attachment}

    @pytest.mark.tier2
    def test_attached_to_default_gateway(
        self,
        admin_client: DynamicClient,
        implicit_model: dict[str, Any],
    ) -> None:
        """Given a model without gatewayRefs in a tenant-less namespace, when the AGC reconciles it,
        then the single result is for the default Gateway, names an attachment route on it,
        and the spec still names no Gateway.
        """
        model = implicit_model["model"]
        route = implicit_model["route"]
        route_name = route["metadata"]["name"]
        assert re.fullmatch(rf"external-model-{re.escape(model['name'])}-[0-9a-f]{{12}}", route_name), (
            f"{route_name} is not an attachment route name (external-model-<model>-<hash>)"
        )
        parents = [
            (parent.get("namespace"), parent.get("name"))
            for parent in (route.get("spec") or {}).get("parentRefs") or []
        ]
        assert parents == [(MAAS_GATEWAY_NAMESPACE, MAAS_GATEWAY_NAME)], f"unexpected route parentRefs: {parents}"
        external_model = get_json_or_none(admin_client, "externalmodel", model["name"], model["namespace"]) or {}
        spec = external_model.get("spec") or {}
        assert not spec.get("gatewayRefs"), f"the implicit attachment must not mutate spec: {spec}"
        status = external_model.get("status") or {}
        results = [(entry.get("namespace"), entry.get("name")) for entry in status.get("gateways") or []]
        assert results == [(MAAS_GATEWAY_NAMESPACE, MAAS_GATEWAY_NAME)], f"unexpected status.gateways: {results}"
        assert status.get("httpRouteName") == route_name, (
            f"status.httpRouteName {status.get('httpRouteName')!r} must name the single attachment "
            f"route {route_name} (maas-controller resolves the model's route from it)"
        )

    @pytest.mark.tier2
    def test_provider_reports_ready(
        self,
        admin_client: DynamicClient,
        implicit_model: dict[str, Any],
    ) -> None:
        """Given a provider in a namespace with no MaasTenantConfig, when its status is read,
        then it reports its own validated Ready configuration instead of Pending.
        """
        model = implicit_model["model"]

        def _check() -> bool:
            provider = get_json_or_none(admin_client, "externalprovider", model["provider"], model["namespace"]) or {}
            status = provider.get("status") or {}
            ready: dict[str, Any] = next(
                (condition for condition in status.get("conditions") or [] if condition.get("type") == "Ready"),
                {},
            )
            assert status.get("phase") == "Ready" and ready.get("status") == "True", (
                f"provider {model['provider']} is not Ready: {redact_sensitive(status)}"
            )
            return True

        wait_until(check=_check, timeout=READY_TIMEOUT, what=f"provider {model['provider']} Ready")

    @pytest.mark.tier2
    def test_served_through_default_gateway(
        self,
        request_session_http: requests.Session,
        default_gateway_url: str,
        implicit_model: dict[str, Any],
    ) -> None:
        """Given the implicitly attached model, when inference runs through the default Gateway,
        then the provider serves it with the model's credential injected (``simple`` auth served as
        ``apikey``) and without the client's MaaS API key.
        """
        model = implicit_model["model"]
        response = infer_chat_completion(
            request_session_http=request_session_http,
            gateway_url=default_gateway_url,
            model=model,
            api_key=implicit_model["api_key"],
        )
        assert response.status_code == 200, (
            f"expected 200 through {MAAS_GATEWAY_NAME}, got {response.status_code}: {redact_sensitive(response.text)}"
        )
        assert model["path_tag"] in response.text, (
            f"response did not come from the model's provider: {redact_sensitive(response.text)}"
        )
        assert model["credential"] in response.text, "the model's credential did not reach the provider"
        assert implicit_model["api_key"] not in response.text, "the client's MaaS API key was forwarded to the provider"

    @pytest.mark.tier3
    def test_unknown_key_rejected(
        self,
        request_session_http: requests.Session,
        default_gateway_url: str,
        implicit_model: dict[str, Any],
    ) -> None:
        """Given an unknown API key, when inference runs through the default Gateway,
        then the request is rejected with 401/403.
        """
        response = infer_chat_completion(
            request_session_http=request_session_http,
            gateway_url=default_gateway_url,
            model=implicit_model["model"],
            api_key=f"sk-extmod-invalid-{generate_random_name()}",
            retries=1,
        )
        assert response.status_code in (401, 403), (
            f"unknown API key must be rejected, got {response.status_code}: {redact_sensitive(response.text)}"
        )


# ─── Implicit attachment lifecycle ──────────────────────────────────────────


class TestImplicitAttachmentLifecycle:
    """gatewayRefs replace the implicit attachment and removing them restores
    it; deleting the model removes its objects."""

    @pytest.mark.tier3
    def test_gateway_refs_replace_implicit_attachment(
        self,
        admin_client: DynamicClient,
        default_tenant_model_namespace: Namespace,
    ) -> None:
        """Given an implicitly attached model, when gatewayRefs bind it to an unmanaged Gateway,
        then the binding is refused with GatewayNotManaged and the implicit attachment is removed
        without falling back to the default Gateway; removing gatewayRefs restores it.
        """
        with default_tenant_model_stack(
            admin_client,
            prefix="e2e-dft-refs",
            namespace=namespace_name(default_tenant_model_namespace),
        ) as model:
            route_name = wait_implicitly_attached(admin_client=admin_client, model=model)["route"]["metadata"]["name"]

            patch_external_model_gateway_refs(
                admin_client=admin_client,
                name=model["name"],
                namespace=model["namespace"],
                refs=[gateway_ref(UNMANAGED_GATEWAY_NAME)],
            )
            wait_for_gateway_attachment_reason(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=UNMANAGED_GATEWAY_NAME,
                reason="GatewayNotManaged",
            )
            wait_not_implicitly_attached(admin_client=admin_client, model=model, route_name=route_name)

            patch_external_model_gateway_refs(
                admin_client=admin_client, name=model["name"], namespace=model["namespace"], refs=None
            )
            wait_implicitly_attached(admin_client=admin_client, model=model)
            assert (
                gateway_status_entry(admin_client, model["name"], model["namespace"], UNMANAGED_GATEWAY_NAME) is None
            ), f"removing gatewayRefs must drop the {UNMANAGED_GATEWAY_NAME} result"

    @pytest.mark.tier3
    def test_model_delete_removes_implicit_attachment(
        self,
        admin_client: DynamicClient,
        default_tenant_model_namespace: Namespace,
    ) -> None:
        """Given an implicitly attached model, when the model and its MaaSModelRef are deleted,
        then the implicit attachment and every object it published are removed.
        """
        with default_tenant_model_stack(
            admin_client,
            prefix="e2e-dft-del",
            namespace=namespace_name(default_tenant_model_namespace),
        ) as model:
            route_name = wait_implicitly_attached(admin_client=admin_client, model=model)["route"]["metadata"]["name"]

            for resource in (
                MaaSModelRef(
                    client=admin_client,
                    name=model["name"],
                    namespace=model["namespace"],
                    ensure_exists=True,
                ),
                ExternalModel(
                    client=admin_client,
                    name=model["name"],
                    namespace=model["namespace"],
                    ensure_exists=True,
                ),
            ):
                resource.delete()
            wait_for_not_found(
                client=admin_client,
                kind="maasmodelref",
                name=model["name"],
                namespace=model["namespace"],
                timeout=READY_TIMEOUT,
            )
            wait_for_not_found(
                client=admin_client,
                kind="externalmodel",
                name=model["name"],
                namespace=model["namespace"],
                timeout=READY_TIMEOUT,
            )
            wait_not_implicitly_attached(admin_client=admin_client, model=model, route_name=route_name)
