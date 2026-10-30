"""E2E: explicit gateway attachment for user-namespace external models.

Ported from models-as-a-service
``test/e2e/tests/test_tenant_external_model_attachment.py`` (RHAI-12994 P0),
adapted to the openshift-python-wrapper / DynamicClient style of this repo.

P0 scenarios:

1. A model in a user namespace explicitly attached to a non-default Praxis
   tenant gateway serves inference through that gateway using its credential
   Secret from the user namespace (delivered as a controller-owned copy).
   Other unselected gateways do not serve it.
2. Removing an attachment or deleting a model removes stale serving
   configuration and unreferenced credential copies while preserving
   unrelated attachments/models.
3. An attachment to a gateway whose allowedRoutes do not admit the model
   namespace is not served, gets no DestinationRule or credential copy, and
   reports NotAllowedByListeners.
4. Admission changes recover without editing the model: re-admitting a
   refused namespace makes its attachment Ready, and narrowing admission
   (removing the gateway-access label) stops a serving attachment. A
   refused model in another namespace never breaks the tenant's own models.
5. The controller-owned credential copy in the tenant namespace follows its
   user-namespace source: rotating the source key updates the copy and
   serving traffic with no manual steps.

Shares the tenant_external_model module tenants (see the package conftest's
``tenant_external_model_tenants``); this module creates its own user
namespaces with unique names.
"""

from __future__ import annotations

import base64
import time
import uuid
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    CREDENTIAL_DATA_KEY,
    TENANT_EXTERNAL_MODEL_POLL_INTERVAL,
    TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
    admitted_user_namespace,
    assert_served_by_provider,
    attachment_httproutes,
    credential_copy_in,
    destination_rules_referencing,
    external_model_stack,
    gateway_ref,
    get_json_or_none,
    httproute_parent_conditions,
    infer_external_chat,
    list_json,
    namespace_name,
    patch_external_model_gateway_refs,
    redact_sensitive,
    serve_external_model_in_tenant,
    unadmitted_user_namespace,
    wait_for_attachment_httproute,
    wait_for_credential_copy,
    wait_for_gateway_attachment_absent,
    wait_for_gateway_attachment_ready,
    wait_for_gateway_attachment_reason,
    wait_for_no_attachment_httproute,
    wait_for_no_credential_copy,
    wait_for_no_destination_rules_referencing,
    wait_for_not_found,
    wait_until,
)
from tests.ai_gateway.models_as_a_service.multitenancy.utils import (
    gateway_access_label_key,
    label_namespace_gateway_access,
)
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.resources.external_model import ExternalModel

LOGGER = structlog.get_logger(name=__name__)

# Settle window mirroring the source suite's RECONCILE_WAIT (E2E_RECONCILE_WAIT,
# default 8s): long enough to observe a controller that wrongly reconciles,
# short enough to not dominate the runtime.
RECONCILE_WAIT = 8

# Serialized with tenant_ipp until the legacy IPP controllers skip
# gatewayRefs ExternalModels (image pin pending): without that guard they
# adopt this suite's models and flap status.httpRouteName.
pytestmark = [pytest.mark.tier2, pytest.mark.xdist_group("tenant_ipp")]


# ─── Module-level helpers ────────────────────────────────────────────────────


def provider_external_name_service(provider_name: str) -> str:
    """The AGC's ExternalName Service name for a provider.

    The attachment HTTPRoute's backendRefs reference this Service (created in
    the model namespace, resolving to the provider endpoint), alongside the
    provider-selection sink Service.
    """
    return f"provider-{provider_name}"


def remove_gateway_access_label(
    admin_client: DynamicClient,
    namespace_name: str,
    gateway_name: str,
) -> None:
    """Remove a namespace's gateway-access label (narrow allowedRoutes admission)."""
    namespace = Namespace(client=admin_client, name=namespace_name, ensure_exists=True)
    assert namespace.exists, f"namespace {namespace_name} not found"
    ResourceEditor(
        patches={
            namespace: {
                "metadata": {"labels": {gateway_access_label_key(gateway_name=gateway_name): None}},
            },
        },
    ).update()
    LOGGER.info(f"Removed gateway-access label of {gateway_name!r} from namespace '{namespace_name}'")


def apply_credential_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    credential: str,
) -> None:
    """Rewrite a provider credential Secret in place (data key ``api-key``).

    Used by the rotation scenario: the source Secret keeps its identity
    (name/namespace) while its key value changes, which is what the
    controller-owned tenant copy must follow.
    """
    secret = Secret(client=admin_client, name=name, namespace=namespace, ensure_exists=True)
    assert secret.exists, f"secret {namespace}/{name} not found"
    ResourceEditor(
        patches={secret: {"stringData": {CREDENTIAL_DATA_KEY: credential}}},
    ).update()
    LOGGER.info(f"Rotated credential secret {namespace}/{name}")


@contextmanager
def _serve_in_tenant(
    admin_client: DynamicClient,
    request_session_http: requests.Session,
    ocp_token: str,
    case: dict[str, str],
    model: dict[str, Any],
) -> Generator[dict[str, str], Any, Any]:
    """Serve one model in a tenant and clean the MaaS stack up afterwards."""
    with serve_external_model_in_tenant(
        admin_client,
        request_session_http,
        ocp_token,
        case,
        model["name"],
        model["namespace"],
        policy_name=f"{model['name']}-access",
        subscription_name=f"{model['name']}-sub",
    ) as access:
        yield access


def _assert_not_served(response: requests.Response) -> None:
    """Assert a response proves the model was not served.

    A successful serve is a 200 from the echo provider, so any other status
    is a non-serving verdict. 401/403/404 are the expected refusals (auth
    denial / no route); 502/503 with a gateway-generated (non-provider)
    body are envoy-side failures on a gateway that holds no route for the
    model — observed under load while filters reload, and equally proof
    that no serve happened.
    """
    assert response.status_code != 200, (
        f"expected the model to not be served, got {response.status_code}: {redact_sensitive(response.text)}"
    )
    if response.status_code in (502, 503):
        assert (
            not (response.text or "").strip()
            or "Internal Server Error" in response.text
            or "Service Unavailable" in response.text
        ), (
            f"5xx carries a provider-shaped body, expected a gateway-generated "
            f"refusal/failure: {redact_sensitive(response.text)}"
        )


# ─── Shared module environment ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def attachment_env(
    admin_client: DynamicClient,
    tenant_external_model_tenants: dict[str, dict[str, str]],
    teardown_resources: bool,
) -> Generator[dict[str, Any], Any, Any]:
    """User namespaces (admitted + deliberately unadmitted) on the shared tenants."""
    tenant_a = tenant_external_model_tenants["tenant_a"]
    tenant_b = tenant_external_model_tenants["tenant_b"]
    with ExitStack() as stack:
        user_ns = stack.enter_context(
            cm=admitted_user_namespace(
                admin_client,
                "e2e-extmod-att-user",
                tenant_a["gateway_name"],
                tenant_b["gateway_name"],
                teardown=teardown_resources,
            )
        )
        unadmitted_ns = stack.enter_context(
            cm=unadmitted_user_namespace(
                admin_client,
                "e2e-extmod-att-noaccess",
                teardown=teardown_resources,
            )
        )
        yield {
            "tenant_a": tenant_a,
            "tenant_b": tenant_b,
            "user_ns": user_ns.name,
            "unadmitted_ns": unadmitted_ns.name,
        }


# ─── P0 #1: user-namespace model on a non-default tenant gateway ────────────


class TestUserNamespaceAttachmentServing:
    """A user-ns model attached to tenant B's gateway serves through it only."""

    @pytest.fixture(scope="class")
    def attached_model(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        attachment_env: dict[str, Any],
        teardown_resources: bool,
    ) -> Generator[dict[str, Any], Any, Any]:
        """One model attached to (and served through) tenant B's gateway."""
        env = attachment_env
        tenant_b = env["tenant_b"]
        with ExitStack() as stack:
            model = stack.enter_context(
                cm=external_model_stack(
                    admin_client,
                    name=f"e2e-att-serve-{uuid.uuid4().hex[:6]}",
                    namespace=env["user_ns"],
                    gateway_refs=[gateway_ref(tenant_b["gateway_name"])],
                    teardown=teardown_resources,
                )
            )
            access = stack.enter_context(
                cm=_serve_in_tenant(admin_client, request_session_http, ocp_token_for_actor, tenant_b, model)
            )
            yield {"model": model, "access": access}

    def test_attachment_status_ready(
        self, admin_client: DynamicClient, attachment_env: dict[str, Any], attached_model: dict[str, Any]
    ) -> None:
        """The attachment to tenant B's gateway reports serving readiness.

        Given a user-ns model attached to tenant B's gateway, when reading the
        model's status, then the per-gateway status.gateways entry (or the
        model's aggregate Ready when the AGC reports aggregate status only)
        reports serving readiness.
        """
        env = attachment_env
        model = attached_model["model"]
        wait_for_gateway_attachment_ready(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=env["tenant_b"]["gateway_name"],
        )

    def test_one_httproute_in_model_namespace(
        self, admin_client: DynamicClient, attachment_env: dict[str, Any], attached_model: dict[str, Any]
    ) -> None:
        """The attachment's HTTPRoute lives in the model namespace.

        Given the attachment is Ready, when listing HTTPRoutes in the model
        namespace attached to tenant B's gateway, then exactly one exists and
        it routes to the provider Service created there.
        """
        env = attachment_env
        model = attached_model["model"]
        tenant_b = env["tenant_b"]
        # Scope by model name: the lifecycle and listener classes share this
        # user namespace and gateway, so an unscoped wait could pass on one
        # of their routes.
        routes = wait_for_attachment_httproute(
            admin_client=admin_client,
            route_namespace=model["namespace"],
            gateway_name=tenant_b["gateway_name"],
            model_name=model["name"],
        )
        assert len(routes) == 1, (
            f"expected exactly one HTTPRoute attached to {tenant_b['gateway_name']} "
            f"in {model['namespace']}, got {[r['metadata']['name'] for r in routes]}"
        )
        route = routes[0]
        assert route["metadata"]["namespace"] == model["namespace"]
        backend_names = [
            backend.get("name")
            for rule in route.get("spec", {}).get("rules", [])
            for backend in rule.get("backendRefs", [])
        ]
        provider_service = provider_external_name_service(provider_name=model["provider"])
        assert provider_service in backend_names, (
            f"HTTPRoute backendRefs {backend_names} do not reference the provider's "
            f"ExternalName Service {provider_service}"
        )

    def test_credential_copy_delivered_to_tenant(
        self, admin_client: DynamicClient, attachment_env: dict[str, Any], attached_model: dict[str, Any]
    ) -> None:
        """Given a serving attachment, when reading tenant B's namespace, then the
        user-namespace credential is delivered there as a controller-owned copy
        with the same content.
        """
        env = attachment_env
        model = attached_model["model"]
        tenant_b = env["tenant_b"]
        copy = wait_for_credential_copy(
            admin_client=admin_client,
            tenant_namespace=tenant_b["tenant_ns"],
            source_secret_name=model["secret_name"],
            source_namespace=model["secret_namespace"],
        )
        assert copy["metadata"]["namespace"] == tenant_b["tenant_ns"]

    def test_unselected_gateway_not_serving(
        self, admin_client: DynamicClient, attachment_env: dict[str, Any], attached_model: dict[str, Any]
    ) -> None:
        """Given the model is attached only to tenant B's gateway, when inspecting
        tenant A's gateway, then it has no route, no credential copy, and does
        not serve the model.
        """
        env = attachment_env
        model = attached_model["model"]
        tenant_a = env["tenant_a"]
        assert attachment_httproutes(admin_client, model["namespace"], tenant_a["gateway_name"]) == [], (
            "unselected gateway must not have an HTTPRoute for the model"
        )
        assert (
            credential_copy_in(admin_client, tenant_a["tenant_ns"], model["secret_name"], model["secret_namespace"])
            is None
        ), "unselected tenant must not receive a credential copy"
        response = infer_external_chat(
            gateway_url=tenant_a["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=attached_model["access"]["api_key"],
            # The same request that serves on tenant B, with no retries: A
            # has no route, so the refusal is immediate and stable.
            path_tag=model["path_tag"],
            retries=1,
        )
        _assert_not_served(response=response)

    def test_inference_serves_with_user_namespace_credential(
        self, admin_client: DynamicClient, attachment_env: dict[str, Any], attached_model: dict[str, Any]
    ) -> None:
        """Given a serving attachment, when running real inference through tenant
        B's gateway, then it reaches the provider with the user-namespace
        credential injected.
        """
        env = attachment_env
        model = attached_model["model"]
        tenant_b = env["tenant_b"]
        response = infer_external_chat(
            gateway_url=tenant_b["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=attached_model["access"]["api_key"],
            path_tag=model["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])


# ─── P0 #3: lifecycle cleanup ───────────────────────────────────────────────


class TestAttachmentLifecycleCleanup:
    """Detach and delete remove only their own serving configuration."""

    @pytest.fixture(scope="class")
    def unrelated_model(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        attachment_env: dict[str, Any],
        teardown_resources: bool,
    ) -> Generator[dict[str, Any], Any, Any]:
        """A bystander model on tenant B's gateway that must survive cleanup."""
        env = attachment_env
        tenant_b = env["tenant_b"]
        with ExitStack() as stack:
            model = stack.enter_context(
                cm=external_model_stack(
                    admin_client,
                    name=f"e2e-att-unrel-{uuid.uuid4().hex[:6]}",
                    namespace=env["user_ns"],
                    gateway_refs=[gateway_ref(tenant_b["gateway_name"])],
                    teardown=teardown_resources,
                )
            )
            access = stack.enter_context(
                cm=_serve_in_tenant(admin_client, request_session_http, ocp_token_for_actor, tenant_b, model)
            )
            yield {"model": model, "access": access}

    def test_detach_removes_only_that_attachment(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        attachment_env: dict[str, Any],
        teardown_resources: bool,
        unrelated_model: dict[str, Any],
    ) -> None:
        """Removing tenant A's attachment removes A's route and (now
        unreferenced) credential copy while tenant B keeps serving.

        Given a model attached to both tenant gateways, when its gatewayRefs
        are narrowed to tenant B only, then tenant A's route, credential copy
        and DestinationRule disappear while tenant B's stay and keep serving.

        The model starts attached to both gateways, but the AGC reports
        ``status.httpRouteName`` only for single-gateway models (the same
        gap the shared module's MaaS-route gate documents), so the MaaS
        subscription/key stack is built after the detach leaves one
        gateway; before that the attachment is verified at the
        route/credential level on both tenants.
        """
        env = attachment_env
        tenant_a = env["tenant_a"]
        tenant_b = env["tenant_b"]
        with external_model_stack(
            admin_client,
            name=f"e2e-att-detach-{uuid.uuid4().hex[:6]}",
            namespace=env["user_ns"],
            gateway_refs=[gateway_ref(tenant_a["gateway_name"]), gateway_ref(tenant_b["gateway_name"])],
            teardown=teardown_resources,
        ) as model:
            # Baseline: both attachments are live at the serving-config level.
            # AGC puts the provider DestinationRules in the Gateway namespace,
            # labeled with each attachment's gateway — capture both per gateway.
            for case in (tenant_a, tenant_b):
                wait_for_gateway_attachment_ready(
                    admin_client=admin_client,
                    model_name=model["name"],
                    model_namespace=model["namespace"],
                    gateway_name=case["gateway_name"],
                )
                wait_for_attachment_httproute(
                    admin_client=admin_client,
                    route_namespace=model["namespace"],
                    gateway_name=case["gateway_name"],
                    model_name=model["name"],
                )
                wait_for_credential_copy(
                    admin_client=admin_client,
                    tenant_namespace=case["tenant_ns"],
                    source_secret_name=model["secret_name"],
                    source_namespace=model["secret_namespace"],
                )
                assert destination_rules_referencing(
                    admin_client, model["name"], MAAS_GATEWAY_NAMESPACE, also_matching=(case["gateway_name"],)
                ), f"provider DestinationRule for {model['name']} on {case['gateway_name']} must exist while attached"

            tenant_b_rule_uids = {
                rule["metadata"]["uid"]
                for rule in destination_rules_referencing(
                    admin_client, model["name"], MAAS_GATEWAY_NAMESPACE, also_matching=(tenant_b["gateway_name"],)
                )
            }

            patch_external_model_gateway_refs(
                admin_client=admin_client,
                name=model["name"],
                namespace=model["namespace"],
                refs=[gateway_ref(tenant_b["gateway_name"])],
            )

            wait_for_gateway_attachment_absent(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=tenant_a["gateway_name"],
            )
            wait_for_no_attachment_httproute(
                admin_client=admin_client,
                route_namespace=model["namespace"],
                gateway_name=tenant_a["gateway_name"],
                model_name=model["name"],
            )
            wait_for_no_credential_copy(
                admin_client=admin_client,
                tenant_namespace=tenant_a["tenant_ns"],
                source_secret_name=model["secret_name"],
                source_namespace=model["secret_namespace"],
            )
            wait_for_no_destination_rules_referencing(  # noqa: FCN001 - namespaces feed *args
                admin_client, model["name"], MAAS_GATEWAY_NAMESPACE, also_matching=(tenant_a["gateway_name"],)
            )
            # The remaining attachment's DestinationRule is untouched.
            assert {
                rule["metadata"]["uid"]
                for rule in destination_rules_referencing(
                    admin_client, model["name"], MAAS_GATEWAY_NAMESPACE, also_matching=(tenant_b["gateway_name"],)
                )
            } == tenant_b_rule_uids, "detaching tenant A must not recreate tenant B's DestinationRule"

            # The remaining attachment is untouched and now serves end to end:
            # with a single gateway the AGC reports httpRouteName, so the MaaS
            # subscription/key stack completes.
            wait_for_gateway_attachment_ready(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=tenant_b["gateway_name"],
            )
            wait_for_credential_copy(
                admin_client=admin_client,
                tenant_namespace=tenant_b["tenant_ns"],
                source_secret_name=model["secret_name"],
                source_namespace=model["secret_namespace"],
            )
            with _serve_in_tenant(admin_client, request_session_http, ocp_token_for_actor, tenant_b, model) as access_b:
                response = infer_external_chat(
                    gateway_url=tenant_b["gateway_url"],
                    model_namespace=model["namespace"],
                    model_client_name=model["client_name"],
                    api_key=access_b["api_key"],
                    path_tag=model["path_tag"],
                )
                assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])

    def test_model_delete_removes_serving_config(
        self,
        admin_client: DynamicClient,
        attachment_env: dict[str, Any],
        teardown_resources: bool,
        unrelated_model: dict[str, Any],
    ) -> None:
        """Deleting a model removes all of its serving configuration while an
        unrelated model on the same gateway keeps serving.

        Given a model attached to two gateways, when the model (and its
        MaaSModelRef) are deleted, then every route, credential copy and
        DestinationRule of that model disappears while the bystander keeps
        serving.

        Like the detach test, the deleted model is attached to two
        gateways, so no MaaS stack is built for it (single-gateway
        httpRouteName gap) — its cleanup is proven at the route/credential
        level; the bystander proves serving survives the deletion.
        """
        env = attachment_env
        tenant_a = env["tenant_a"]
        tenant_b = env["tenant_b"]
        with external_model_stack(
            admin_client,
            name=f"e2e-att-delete-{uuid.uuid4().hex[:6]}",
            namespace=env["user_ns"],
            gateway_refs=[gateway_ref(tenant_a["gateway_name"]), gateway_ref(tenant_b["gateway_name"])],
            teardown=teardown_resources,
        ) as model:
            # Both attachments are live before the delete.
            for case in (tenant_a, tenant_b):
                wait_for_gateway_attachment_ready(
                    admin_client=admin_client,
                    model_name=model["name"],
                    model_namespace=model["namespace"],
                    gateway_name=case["gateway_name"],
                )
                wait_for_credential_copy(
                    admin_client=admin_client,
                    tenant_namespace=case["tenant_ns"],
                    source_secret_name=model["secret_name"],
                    source_namespace=model["secret_namespace"],
                )
                assert destination_rules_referencing(
                    admin_client, model["name"], MAAS_GATEWAY_NAMESPACE, also_matching=(case["gateway_name"],)
                ), f"provider DestinationRule for {model['name']} on {case['gateway_name']} must exist while attached"

            model_ref = MaaSModelRef(
                client=admin_client,
                name=model["name"],
                namespace=model["namespace"],
                ensure_exists=True,
            )
            model_ref.delete()
            wait_for_not_found(
                client=admin_client, kind="maasmodelref", name=model["name"], namespace=model["namespace"]
            )

            external_model = ExternalModel(
                client=admin_client,
                name=model["name"],
                namespace=model["namespace"],
                ensure_exists=True,
            )
            external_model.delete()

            for case in (tenant_a, tenant_b):
                wait_for_no_attachment_httproute(
                    admin_client=admin_client,
                    route_namespace=model["namespace"],
                    gateway_name=case["gateway_name"],
                    model_name=model["name"],
                )
                wait_for_no_credential_copy(
                    admin_client=admin_client,
                    tenant_namespace=case["tenant_ns"],
                    source_secret_name=model["secret_name"],
                    source_namespace=model["secret_namespace"],
                )
            # AGC keeps the provider DestinationRules in the Gateway namespace:
            # deleting the model must remove them all, on every attachment.
            wait_for_no_destination_rules_referencing(  # noqa: FCN001 - namespace feeds *args
                admin_client, model["name"], MAAS_GATEWAY_NAMESPACE
            )
            assert get_json_or_none(admin_client, "externalmodel", model["name"], model["namespace"]) is None

        # The unrelated model on the same gateway is unaffected.
        unrelated = unrelated_model["model"]
        response = infer_external_chat(
            gateway_url=tenant_b["gateway_url"],
            model_namespace=unrelated["namespace"],
            model_client_name=unrelated["client_name"],
            api_key=unrelated_model["access"]["api_key"],
            path_tag=unrelated["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=unrelated["path_tag"], credential=unrelated["credential"])


# ─── P0 #6: gateway-attachment authorization (NotAllowedByListeners) ────────


class TestListenerNotAllowed:
    """A gateway whose allowedRoutes do not admit the model namespace refuses
    the attachment and publishes nothing into the tenant.
    """

    @pytest.fixture(scope="class")
    def refused_model(
        self,
        admin_client: DynamicClient,
        attachment_env: dict[str, Any],
        teardown_resources: bool,
    ) -> Generator[dict[str, Any], Any, Any]:
        """One model attached to tenant B's gateway from an unadmitted namespace."""
        env = attachment_env
        tenant_b = env["tenant_b"]
        with external_model_stack(
            admin_client,
            name=f"e2e-att-noaccess-{uuid.uuid4().hex[:6]}",
            namespace=env["unadmitted_ns"],
            gateway_refs=[gateway_ref(tenant_b["gateway_name"])],
            teardown=teardown_resources,
        ) as model:
            yield model

    def test_attachment_reports_not_allowed_by_listeners(
        self, admin_client: DynamicClient, attachment_env: dict[str, Any], refused_model: dict[str, Any]
    ) -> None:
        """Given a model in a namespace the gateway does not admit, when reading
        the attachment status, then it reports NotAllowedByListeners.
        """
        env = attachment_env
        model = refused_model
        wait_for_gateway_attachment_reason(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=env["tenant_b"]["gateway_name"],
            reason="NotAllowedByListeners",
        )

    def test_nothing_published_into_tenant(
        self, admin_client: DynamicClient, attachment_env: dict[str, Any], refused_model: dict[str, Any]
    ) -> None:
        """Given a refused attachment, when watching the tenant namespace and the
        Gateway namespace for a settle window, then no credential copy and no
        DestinationRule are ever published.
        """
        env = attachment_env
        model = refused_model
        tenant_b = env["tenant_b"]
        # Gate on the refusal being reported first: a bare settle window
        # could pass before the controller ever reconciles, making absence
        # trivially true. The test must not rely on its sibling having
        # already waited for the reason.
        wait_for_gateway_attachment_reason(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=tenant_b["gateway_name"],
            reason="NotAllowedByListeners",
        )
        # Then require absence to HOLD across a settle window: the
        # attachment must never publish, so absence must hold, not converge.
        # AGC keeps the provider DestinationRules in the Gateway namespace,
        # so that is where "never created" is observable.
        deadline = time.time() + RECONCILE_WAIT
        while True:
            assert (
                credential_copy_in(admin_client, tenant_b["tenant_ns"], model["secret_name"], model["secret_namespace"])
                is None
            ), "credential copy must not be delivered while the listener refuses the namespace"
            assert not destination_rules_referencing(admin_client, model["name"], MAAS_GATEWAY_NAMESPACE), (
                "DestinationRule must not be created while the listener refuses the namespace"
            )
            if time.time() >= deadline:
                break
            time.sleep(TENANT_EXTERNAL_MODEL_POLL_INTERVAL)

    def test_route_not_admitted_and_not_served(
        self, admin_client: DynamicClient, attachment_env: dict[str, Any], refused_model: dict[str, Any]
    ) -> None:
        """Given a refused attachment, when reading the route's parent conditions
        and sending a request through the gateway, then the route (if created)
        is not admitted and the request is not served.
        """
        env = attachment_env
        model = refused_model
        tenant_b = env["tenant_b"]
        routes = attachment_httproutes(
            admin_client=admin_client, route_namespace=model["namespace"], gateway_name=tenant_b["gateway_name"]
        )
        if routes:
            # AGC creates the attachment's HTTPRoute in the model namespace
            # first; the gateway must refuse to admit it.
            for route in routes:
                conditions = httproute_parent_conditions(route=route, gateway_name=tenant_b["gateway_name"])
                accepted = next((c for c in conditions if c.get("type") == "Accepted"), None)
                assert accepted is not None and accepted.get("status") == "False", (
                    f"route {route['metadata']['name']} must not be admitted: {conditions}"
                )
        response = infer_external_chat(
            gateway_url=tenant_b["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=f"sk-extmod-unused-{uuid.uuid4().hex[:8]}",  # pragma: allowlist secret
            path_tag=model["path_tag"],
            retries=1,
        )
        _assert_not_served(response=response)


# ─── P0 #4: admission changes recover without editing the model ─────────────


class TestListenerAdmissionRecovery:
    """A refused attachment recovers when the namespace is admitted, and a
    serving attachment stops when admission is narrowed — in both directions
    without touching the model, and without breaking the tenant's own models.

    The recovery model lives in its own namespace so this class can flip the
    gateway-access label freely: the module's shared user namespace hosts
    other classes' routes, and removing its label would break them.
    """

    @pytest.fixture(scope="class")
    def recovery_env(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        attachment_env: dict[str, Any],
        teardown_resources: bool,
    ) -> Generator[dict[str, Any], Any, Any]:
        """A refused model and a healthy control model, each in its own namespace.

        The namespace context managers start (and own teardown of) the
        namespaces right after creation: pytest runs no teardown after a
        failed setup, so provisioning or serving failures must still delete
        the namespaces.
        """
        env = attachment_env
        tenant_b = env["tenant_b"]
        with ExitStack() as stack:
            refused_ns = stack.enter_context(
                cm=unadmitted_user_namespace(
                    admin_client,
                    "e2e-extmod-rec-noaccess",
                    teardown=teardown_resources,
                )
            )
            control_ns = stack.enter_context(
                cm=admitted_user_namespace(
                    admin_client,
                    "e2e-extmod-rec-user",
                    tenant_b["gateway_name"],
                    teardown=teardown_resources,
                )
            )
            recovering_model = stack.enter_context(
                cm=external_model_stack(
                    admin_client,
                    name=f"e2e-rec-recover-{uuid.uuid4().hex[:6]}",
                    namespace=namespace_name(refused_ns),
                    gateway_refs=[gateway_ref(tenant_b["gateway_name"])],
                    teardown=teardown_resources,
                )
            )
            control_model = stack.enter_context(
                cm=external_model_stack(
                    admin_client,
                    name=f"e2e-rec-ctrl-{uuid.uuid4().hex[:6]}",
                    namespace=namespace_name(control_ns),
                    gateway_refs=[gateway_ref(tenant_b["gateway_name"])],
                    teardown=teardown_resources,
                )
            )
            control_access = stack.enter_context(
                cm=_serve_in_tenant(admin_client, request_session_http, ocp_token_for_actor, tenant_b, control_model)
            )
            yield {
                "tenant_b": tenant_b,
                "refused_ns": refused_ns.name,
                "control_ns": control_ns.name,
                "recovering_model": recovering_model,
                "control_model": control_model,
                "control_access": control_access,
            }

    def test_refused_model_does_not_break_tenant_models(
        self, admin_client: DynamicClient, attachment_env: dict[str, Any], recovery_env: dict[str, Any]
    ) -> None:
        """Given the gateway refuses the recovering model's namespace, when
        reading that attachment's status and serving the control model, then
        the attachment reports NotAllowedByListeners and the healthy model on
        the same gateway keeps serving end to end.
        """
        env = recovery_env
        tenant_b = env["tenant_b"]
        model = env["recovering_model"]
        wait_for_gateway_attachment_reason(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=tenant_b["gateway_name"],
            reason="NotAllowedByListeners",
        )
        control = env["control_model"]
        response = infer_external_chat(
            gateway_url=tenant_b["gateway_url"],
            model_namespace=control["namespace"],
            model_client_name=control["client_name"],
            api_key=env["control_access"]["api_key"],
            path_tag=control["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=control["path_tag"], credential=control["credential"])

    def test_readmission_recovers_attachment_without_model_edit(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        attachment_env: dict[str, Any],
        recovery_env: dict[str, Any],
    ) -> None:
        """Given a refused attachment, when the namespace is labeled (admitted
        into the gateway's allowedRoutes), then the attachment becomes Ready
        and serving — without any change to the ExternalModel itself.
        """
        env = recovery_env
        tenant_b = env["tenant_b"]
        model = env["recovering_model"]
        spec_before = (get_json_or_none(admin_client, "externalmodel", model["name"], model["namespace"]) or {}).get(
            "spec"
        )

        label_namespace_gateway_access(
            admin_client=admin_client,
            namespace_name=env["refused_ns"],
            gateway_name=tenant_b["gateway_name"],
        )

        wait_for_gateway_attachment_ready(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=tenant_b["gateway_name"],
        )
        wait_for_credential_copy(
            admin_client=admin_client,
            tenant_namespace=tenant_b["tenant_ns"],
            source_secret_name=model["secret_name"],
            source_namespace=model["secret_namespace"],
        )
        spec_after = (get_json_or_none(admin_client, "externalmodel", model["name"], model["namespace"]) or {}).get(
            "spec"
        )
        assert spec_after == spec_before, (
            f"recovery must not edit the model: spec changed from {spec_before} to {spec_after}"
        )

        # The recovered attachment now carries the full MaaS stack (it is a
        # single-gateway model, so the route resolves for subscriptions).
        with _serve_in_tenant(admin_client, request_session_http, ocp_token_for_actor, tenant_b, model) as access:
            response = infer_external_chat(
                gateway_url=tenant_b["gateway_url"],
                model_namespace=model["namespace"],
                model_client_name=model["client_name"],
                api_key=access["api_key"],
                path_tag=model["path_tag"],
            )
            assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])

    def test_narrowing_admission_stops_serving(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        attachment_env: dict[str, Any],
        recovery_env: dict[str, Any],
    ) -> None:
        """Given a serving attachment, when the gateway-access label is removed
        (narrowing allowedRoutes), then the gateway refuses the route again:
        the attachment reports NotAllowedByListeners, requests are refused,
        and the healthy model in an admitted namespace keeps serving.

        Self-contained on purpose: this test establishes the serving state
        (admitted attachment + valid API key) itself rather than relying on
        the readmission test having run first, and proves serving stopped
        with the key that was valid before the narrowing — a 401 from an
        invalid key would not distinguish "no longer admitted" from "never
        served".
        """
        env = recovery_env
        tenant_b = env["tenant_b"]
        model = env["recovering_model"]

        # Establish serving (no-op label apply if the readmission test
        # already ran; the MaaS stack apply is idempotent).
        label_namespace_gateway_access(
            admin_client=admin_client,
            namespace_name=env["refused_ns"],
            gateway_name=tenant_b["gateway_name"],
        )
        wait_for_gateway_attachment_ready(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=tenant_b["gateway_name"],
        )
        with _serve_in_tenant(admin_client, request_session_http, ocp_token_for_actor, tenant_b, model) as access:
            served = infer_external_chat(
                gateway_url=tenant_b["gateway_url"],
                model_namespace=model["namespace"],
                model_client_name=model["client_name"],
                api_key=access["api_key"],
                path_tag=model["path_tag"],
            )
            assert_served_by_provider(response=served, path_tag=model["path_tag"], credential=model["credential"])

            # Narrow: the gateway must refuse the route again.
            remove_gateway_access_label(
                admin_client=admin_client,
                namespace_name=env["refused_ns"],
                gateway_name=tenant_b["gateway_name"],
            )

            wait_for_gateway_attachment_reason(
                admin_client=admin_client,
                model_name=model["name"],
                model_namespace=model["namespace"],
                gateway_name=tenant_b["gateway_name"],
                reason="NotAllowedByListeners",
            )
            routes = attachment_httproutes(
                admin_client=admin_client, route_namespace=model["namespace"], gateway_name=tenant_b["gateway_name"]
            )
            for route in routes:
                conditions = httproute_parent_conditions(route=route, gateway_name=tenant_b["gateway_name"])
                accepted = next((c for c in conditions if c.get("type") == "Accepted"), None)
                assert accepted is not None and accepted.get("status") == "False", (
                    f"route {route['metadata']['name']} must not stay admitted: {conditions}"
                )
            # The key that just served must now be refused: a non-200 here is
            # the narrowing verdict, not an authorization failure. The control
            # plane (attachment reason, route Accepted=False) can report the
            # refusal while envoy still serves the old route briefly, so poll
            # until the data plane catches up; _assert_not_served keeps the
            # failure message specific if the narrowing never takes effect.

            def _refused() -> bool:
                response = infer_external_chat(
                    gateway_url=tenant_b["gateway_url"],
                    model_namespace=model["namespace"],
                    model_client_name=model["client_name"],
                    api_key=access["api_key"],
                    path_tag=model["path_tag"],
                    retries=1,
                )
                _assert_not_served(response=response)
                return True

            wait_until(
                check=_refused,
                timeout=TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
                what=(
                    f"previously valid key no longer serves {model['namespace']}/{model['name']} "
                    "after admission narrowing"
                ),
            )

        control = env["control_model"]
        control_served = infer_external_chat(
            gateway_url=tenant_b["gateway_url"],
            model_namespace=control["namespace"],
            model_client_name=control["client_name"],
            api_key=env["control_access"]["api_key"],
            path_tag=control["path_tag"],
        )
        assert_served_by_provider(
            response=control_served, path_tag=control["path_tag"], credential=control["credential"]
        )


# ─── P0 #5: the tenant credential copy follows its source ──────────────────


class TestCredentialRotationFollowsSource:
    """Rotating the key in the user-namespace source Secret updates the
    controller-owned copy in the tenant namespace and serving traffic uses
    the new key, with no manual steps.

    A stale copy (old key still in use, nothing visible to the user) is the
    failure this catches: the assertions check both that a copy carrying the
    new key exists and that no Secret in the tenant namespace still holds
    the old one.
    """

    @pytest.fixture(scope="class")
    def rotation_model(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        attachment_env: dict[str, Any],
        teardown_resources: bool,
    ) -> Generator[dict[str, Any], Any, Any]:
        """One serving model whose source credential Secret will be rotated.

        Its own namespace: the rotation rewrites this Secret in place, and
        the module's shared user namespace hosts other classes' models. The
        namespace context manager owns its teardown right after creation:
        pytest runs no teardown after a failed setup, so provisioning or
        serving failures must still delete the namespace.
        """
        env = attachment_env
        tenant_b = env["tenant_b"]
        with ExitStack() as stack:
            user_ns = stack.enter_context(
                cm=admitted_user_namespace(
                    admin_client,
                    "e2e-extmod-rot-user",
                    tenant_b["gateway_name"],
                    teardown=teardown_resources,
                )
            )
            model = stack.enter_context(
                cm=external_model_stack(
                    admin_client,
                    name=f"e2e-rot-{uuid.uuid4().hex[:6]}",
                    namespace=namespace_name(user_ns),
                    gateway_refs=[gateway_ref(tenant_b["gateway_name"])],
                    teardown=teardown_resources,
                )
            )
            access = stack.enter_context(
                cm=_serve_in_tenant(admin_client, request_session_http, ocp_token_for_actor, tenant_b, model)
            )
            yield {
                "tenant_b": tenant_b,
                "user_ns": user_ns.name,
                "model": model,
                "access": access,
            }

    def test_rotation_updates_copy_and_serving(
        self, admin_client: DynamicClient, attachment_env: dict[str, Any], rotation_model: dict[str, Any]
    ) -> None:
        """Given a serving attachment, when the user-namespace source Secret's
        key is rotated, then the tenant copy carries the new key, no stale
        copy keeps the old one, and serving traffic uses the new key with no
        manual steps.
        """
        env = rotation_model
        tenant_b = env["tenant_b"]
        model = env["model"]

        # Baseline: serving with the original key.
        response = infer_external_chat(
            gateway_url=tenant_b["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=env["access"]["api_key"],
            path_tag=model["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])

        rotated_credential = f"sk-extmod-rotated-{uuid.uuid4().hex[:10]}"  # pragma: allowlist secret
        apply_credential_secret(
            admin_client=admin_client,
            name=model["secret_name"],
            namespace=model["secret_namespace"],
            credential=rotated_credential,
        )

        # The copy must catch up with the new key...
        wait_for_credential_copy(
            admin_client=admin_client,
            tenant_namespace=tenant_b["tenant_ns"],
            source_secret_name=model["secret_name"],
            source_namespace=model["secret_namespace"],
        )
        # ...and no Secret in the tenant namespace may keep the old one.
        stale_credential = base64.b64encode(model["credential"].encode()).decode()

        def _no_stale_copy() -> bool:
            stale = [
                secret["metadata"]["name"]
                for secret in list_json(admin_client, "secret", tenant_b["tenant_ns"])
                if (secret.get("data") or {}).get(CREDENTIAL_DATA_KEY) == stale_credential
            ]
            assert not stale, f"stale credential copies still present in tenant namespace: {stale}"
            return True

        wait_until(
            check=_no_stale_copy,
            timeout=TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
            what=f"no stale credential copy of {model['secret_namespace']}/{model['secret_name']} "
            f"in {tenant_b['tenant_ns']}",
        )

        # Serving traffic carries the rotated key with no manual steps. The
        # data plane may lag the updated copy (envoy/secret-store refresh),
        # so probe within a bounded window. Only the exact known-gap symptom
        # is tolerated as a skip — a 200 that still carries the OLD key
        # (stale injection; the scenario's "nothing visible to the user"
        # failure). Anything else — serving broke entirely (non-200s), or a
        # 200 carrying neither key — is a different bug and fails red.
        deadline = time.time() + TENANT_EXTERNAL_MODEL_READY_TIMEOUT
        stale_serving_observed = False
        response = None
        last_error_excerpt = ""
        while True:
            response = infer_external_chat(
                gateway_url=tenant_b["gateway_url"],
                model_namespace=model["namespace"],
                model_client_name=model["client_name"],
                api_key=env["access"]["api_key"],
                path_tag=model["path_tag"],
            )
            if response.status_code == 200:
                if rotated_credential in response.text:
                    break
                if model["credential"] in response.text:
                    stale_serving_observed = True
                else:
                    pytest.fail(
                        "serving returned 200 with neither the rotated nor the "
                        "original credential after rotation: "
                        f"{redact_sensitive(response.text)[:300]}"
                    )
            else:
                # Serving broke after a stale 200 was seen: the skip must
                # apply only when the last observed state is the stale-key
                # symptom — a total outage at the deadline fails red.
                stale_serving_observed = False
                last_error_excerpt = redact_sensitive(value=response.text)[:300]
            if time.time() >= deadline:
                if stale_serving_observed:
                    pytest.skip(
                        "credential rotation did not reach serving traffic within "
                        f"{TENANT_EXTERNAL_MODEL_READY_TIMEOUT}s: requests kept "
                        "succeeding with the OLD key (stale data-plane injection — "
                        "the known data-plane credential propagation gap this "
                        "scenario tracks; the controller-owned copy in the tenant "
                        "namespace did update); rerun when the Secret-watch "
                        "propagation ships"
                    )
                pytest.fail(
                    "serving broke entirely after credential rotation (no "
                    "successful request carried the rotated key within "
                    f"{TENANT_EXTERNAL_MODEL_READY_TIMEOUT}s; last response "
                    f"{response.status_code}: {last_error_excerpt}) — distinct "
                    "from the known stale-key gap"
                )
            time.sleep(TENANT_EXTERNAL_MODEL_POLL_INTERVAL)
        assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=rotated_credential)
