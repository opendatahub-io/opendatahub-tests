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

import os
import re
import time
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace
from ocp_resources.project_project_openshift_io import Project
from ocp_resources.secret import Secret

from tests.ai_gateway.models_as_a_service.maas_subscription.utils import (
    MAAS_SUBSCRIPTION_NAMESPACE,
)
from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    CREDENTIAL_DATA_KEY,
    EXTERNAL_API_FORMAT,
    EXTERNAL_ENDPOINT,
    EXTERNAL_TARGET_MODEL,
    PRAXIS_IMAGE_MARKER,
    deployment_image,
    destination_rules_referencing,
    gateway_ref,
    gateway_status_entry,
    get_json_or_none,
    namespace_name,
    patch_external_model_gateway_refs,
    redact_sensitive,
    wait_for_attachment_httproute,
    wait_for_credential_copy,
    wait_for_gateway_attachment_absent,
    wait_for_gateway_attachment_ready,
    wait_for_gateway_attachment_reason,
    wait_for_no_credential_copy,
    wait_for_no_destination_rules_referencing,
    wait_for_not_found,
    wait_until,
)
from tests.ai_gateway.models_as_a_service.multitenancy.utils import (
    tenant_gateway_hostname,
)
from tests.ai_gateway.models_as_a_service.utils import create_api_key
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE
from utilities.general import generate_random_name
from utilities.infra import create_ns
from utilities.resources.external_model import ExternalModel
from utilities.resources.external_provider import ExternalProvider

LOGGER = structlog.get_logger(name=__name__)

# Serialized with the tenant_ipp-flavored MaaS suites, which run legacy IPP
# tenant stacks: those must not run next to this suite's models.
pytestmark = pytest.mark.xdist_group(name="tenant_ipp")

# The namespace 3.5 demos used; the default Gateway's allowedRoutes admits it.
MODEL_NAMESPACE = "llm"

# The default tenant (MaaS subscription namespace) serves these models.
DEFAULT_TENANT_NAMESPACE = MAAS_SUBSCRIPTION_NAMESPACE

# Wait for AGC reconciliation (same env override as the shared suite timeout).
READY_TIMEOUT = int(os.environ.get("E2E_TENANT_EXTERNAL_MODEL_READY_TIMEOUT", "300"))

# A Gateway no AITenant manages (it need not exist): an explicit binding to it
# is refused with GatewayNotManaged.
UNMANAGED_GATEWAY_NAME = "e2e-dft-unmanaged-gateway"

INFERENCE_RETRIES = 6
INFERENCE_RETRY_DELAY = 5
INFERENCE_REQUEST_TIMEOUT = 60


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
    with create_ns(
        admin_client=admin_client,
        name=MODEL_NAMESPACE,
        teardown=teardown_resources,
    ) as namespace:
        yield namespace


@pytest.fixture(scope="class")
def default_tenant_gateway_url(maas_host: str) -> str:
    """External URL of the default (models-as-a-service) Gateway."""
    return f"https://{tenant_gateway_hostname(gateway_name=MAAS_GATEWAY_NAME, maas_host=maas_host)}"


# ─── 3.5-shaped stack ───────────────────────────────────────────────────────


@contextmanager
def default_tenant_model_stack(
    admin_client: DynamicClient,
    *,
    prefix: str,
    namespace: str,
) -> Generator[dict[str, Any], Any, Any]:
    """A 3.5-shaped stack in the model namespace: credential Secret, provider with
    ``auth.type: simple``, ExternalModel without gatewayRefs and its MaaSModelRef."""
    suffix = generate_random_name()
    name = f"{prefix}-{suffix}"
    model: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "provider": f"{name}-provider",
        "secret_name": f"{name}-cred",
        "credential": f"sk-extmod-{suffix}",  # pragma: allowlist secret
        "path_tag": f"{name}-{generate_random_name()}",
    }
    with ExitStack() as stack:
        stack.enter_context(
            cm=Secret(
                client=admin_client,
                name=model["secret_name"],
                namespace=namespace,
                type="Opaque",
                string_data={CREDENTIAL_DATA_KEY: model["credential"]},
                wait_for_resource=True,
            )
        )
        stack.enter_context(
            cm=ExternalProvider(
                client=admin_client,
                name=model["provider"],
                namespace=namespace,
                provider="openai",
                endpoint=EXTERNAL_ENDPOINT,
                # 3.5 shape: simple auth, which the AGC serves as apikey.
                auth={"type": "simple", "secretRef": {"name": model["secret_name"]}},
                wait_for_resource=True,
            )
        )
        stack.enter_context(
            cm=ExternalModel(
                client=admin_client,
                name=name,
                namespace=namespace,
                external_provider_refs=[
                    {
                        "ref": {"name": model["provider"]},
                        "targetModel": EXTERNAL_TARGET_MODEL,
                        "apiFormat": EXTERNAL_API_FORMAT,
                        "path": f"/anything/{model['path_tag']}",
                    }
                ],
                wait_for_resource=True,
            )
        )
        stack.enter_context(
            cm=MaaSModelRef(
                client=admin_client,
                name=name,
                namespace=namespace,
                model_ref={
                    "name": name,
                    "namespace": namespace,
                    "kind": "ExternalModel",
                    "apiGroup": "inference.opendatahub.io",
                },
                wait_for_resource=True,
            )
        )
        yield model


# ─── Attachment state ───────────────────────────────────────────────────────


def _wait_implicitly_attached(
    admin_client: DynamicClient,
    model: dict[str, Any],
) -> dict[str, Any]:
    """Wait for the implicit attachment's result, accepted route, credential
    copy and DestinationRule; returns the result and route."""

    def _destination_rules() -> list[dict[str, Any]]:
        # Source scoped these by the AGC's external-provider/external-gateway
        # labels; content matching on model + default gateway is equivalent.
        return destination_rules_referencing(  # noqa: FCN001 - namespace feeds *args
            admin_client, model["name"], MAAS_GATEWAY_NAMESPACE, also_matching=(MAAS_GATEWAY_NAME,)
        )

    def _published() -> bool:
        assert _destination_rules(), f"no DestinationRule for {model['provider']} in {MAAS_GATEWAY_NAMESPACE}"
        return True

    wait_for_gateway_attachment_ready(
        admin_client=admin_client,
        model_name=model["name"],
        model_namespace=model["namespace"],
        gateway_name=MAAS_GATEWAY_NAME,
    )
    result = gateway_status_entry(
        admin_client=admin_client,
        model_name=model["name"],
        model_namespace=model["namespace"],
        gateway_name=MAAS_GATEWAY_NAME,
    )
    route_name = ((result or {}).get("httpRouteRef") or {}).get("name")
    assert route_name, f"Ready result names no route: {redact_sensitive(result)}"
    # Published only after the Gateway accepts the route.
    route = wait_for_attachment_httproute(
        admin_client=admin_client,
        route_namespace=model["namespace"],
        gateway_name=MAAS_GATEWAY_NAME,
        model_name=model["name"],
    )[0]
    wait_for_credential_copy(
        admin_client=admin_client,
        tenant_namespace=DEFAULT_TENANT_NAMESPACE,
        source_secret_name=model["secret_name"],
        source_namespace=model["namespace"],
    )
    wait_until(check=_published, timeout=READY_TIMEOUT, what=f"DestinationRule for {model['name']}")
    return {"result": result, "route": route}


def _wait_not_implicitly_attached(
    admin_client: DynamicClient,
    model: dict[str, Any],
    route_name: str,
) -> None:
    """Wait until the implicit attachment and every object it published are gone."""
    wait_for_gateway_attachment_absent(
        admin_client=admin_client,
        model_name=model["name"],
        model_namespace=model["namespace"],
        gateway_name=MAAS_GATEWAY_NAME,
    )
    wait_for_not_found(
        client=admin_client, kind="httproute", name=route_name, namespace=model["namespace"], timeout=READY_TIMEOUT
    )
    wait_for_no_credential_copy(
        admin_client=admin_client,
        tenant_namespace=DEFAULT_TENANT_NAMESPACE,
        source_secret_name=model["secret_name"],
        source_namespace=model["namespace"],
        timeout=READY_TIMEOUT,
    )
    wait_for_no_destination_rules_referencing(  # noqa: FCN001 - namespace feeds *args
        admin_client, model["name"], MAAS_GATEWAY_NAMESPACE, also_matching=(MAAS_GATEWAY_NAME,), timeout=READY_TIMEOUT
    )


# ─── Serving through the default tenant ────────────────────────────────────


@contextmanager
def serve_in_default_tenant(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token: str,
    base_url: str,
    model: dict[str, Any],
) -> Generator[str, Any, Any]:
    """MaaSAuthPolicy + MaaSSubscription in the default tenant; yields an API key."""
    policy_name, subscription_name = f"{model['name']}-access", f"{model['name']}-sub"
    with ExitStack() as stack:
        policy = stack.enter_context(
            cm=MaaSAuthPolicy(
                client=admin_client,
                name=policy_name,
                namespace=DEFAULT_TENANT_NAMESPACE,
                model_refs=[{"name": model["name"], "namespace": model["namespace"]}],
                subjects={"groups": [{"name": "system:authenticated"}]},
                wait_for_resource=True,
            )
        )
        policy.wait_for_condition(condition="Ready", status="True", timeout=READY_TIMEOUT)
        subscription = stack.enter_context(
            cm=MaaSSubscription(
                client=admin_client,
                name=subscription_name,
                namespace=DEFAULT_TENANT_NAMESPACE,
                owner={"groups": [{"name": "system:authenticated"}]},
                model_refs=[
                    {
                        "name": model["name"],
                        "namespace": model["namespace"],
                        "tokenRateLimits": [{"limit": 100, "window": "1m"}],
                    }
                ],
                priority=0,
                wait_for_resource=True,
            )
        )
        subscription.wait_for_condition(condition="Ready", status="True", timeout=READY_TIMEOUT)
        # The MaaSModelRef is Ready only once the policy and subscription govern it.
        model_ref = MaaSModelRef(
            client=admin_client,
            name=model["name"],
            namespace=model["namespace"],
            ensure_exists=True,
        )
        model_ref.wait_for_condition(condition="Ready", status="True", timeout=READY_TIMEOUT)
        response, body = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token,
            request_session_http=request_session_http,
            api_key_name=f"e2e-dft-{generate_random_name()}",
            subscription=subscription_name,
        )
        assert response.status_code in (200, 201), (
            f"failed to mint API key on default maas-api {base_url}: "
            f"{response.status_code} {redact_sensitive(response.text)}"
        )
        api_key = body.get("key")
        assert api_key, f"API key missing in response: {redact_sensitive(body)}"
        yield api_key


def _infer(
    request_session_http: requests.Session,
    gateway_url: str,
    model: dict[str, Any],
    api_key: str,
    *,
    retries: int = INFERENCE_RETRIES,
) -> requests.Response:
    """POST a chat completion to the model's echo path through a Gateway."""
    url = f"{gateway_url}/{model['namespace']}/{model['name']}/anything/{model['path_tag']}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model["name"],
        "messages": [{"role": "user", "content": "default tenant e2e"}],
        "max_tokens": 8,
    }
    for attempt in range(retries + 1):
        response = request_session_http.post(url=url, headers=headers, json=payload, timeout=INFERENCE_REQUEST_TIMEOUT)
        transient = response.status_code >= 500 or (
            response.status_code == 429 and "retry" in (response.text or "").lower()
        )
        if not transient or attempt == retries:
            return response
        time.sleep(INFERENCE_RETRY_DELAY)
    raise AssertionError(f"inference to {url} kept failing transiently")  # pragma: no cover


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
            attachment = _wait_implicitly_attached(admin_client=admin_client, model=model)
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
        default_tenant_gateway_url: str,
        implicit_model: dict[str, Any],
    ) -> None:
        """Given the implicitly attached model, when inference runs through the default Gateway,
        then the provider serves it with the model's credential injected (``simple`` auth served as
        ``apikey``) and without the client's MaaS API key.
        """
        model = implicit_model["model"]
        response = _infer(
            request_session_http=request_session_http,
            gateway_url=default_tenant_gateway_url,
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
        default_tenant_gateway_url: str,
        implicit_model: dict[str, Any],
    ) -> None:
        """Given an unknown API key, when inference runs through the default Gateway,
        then the request is rejected with 401/403.
        """
        response = _infer(
            request_session_http=request_session_http,
            gateway_url=default_tenant_gateway_url,
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
            route_name = _wait_implicitly_attached(admin_client=admin_client, model=model)["route"]["metadata"]["name"]

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
            _wait_not_implicitly_attached(admin_client=admin_client, model=model, route_name=route_name)

            patch_external_model_gateway_refs(
                admin_client=admin_client, name=model["name"], namespace=model["namespace"], refs=None
            )
            _wait_implicitly_attached(admin_client=admin_client, model=model)
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
            route_name = _wait_implicitly_attached(admin_client=admin_client, model=model)["route"]["metadata"]["name"]

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
            _wait_not_implicitly_attached(admin_client=admin_client, model=model, route_name=route_name)
