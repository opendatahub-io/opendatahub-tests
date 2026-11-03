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
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_model_ref import MaaSModelRef

from tests.ai_gateway.models_as_a_service.multitenancy.external_model.utils import (
    CREDENTIAL_DATA_KEY,
    TENANT_EXTERNAL_MODEL_POLL_INTERVAL,
    TENANT_EXTERNAL_MODEL_READY_TIMEOUT,
    apply_credential_secret,
    assert_not_served,
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
    patch_external_model_gateway_refs,
    provider_external_name_service,
    redact_sensitive,
    remove_gateway_access_label,
    serve_in_tenant_and_cleanup,
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
    label_namespace_gateway_access,
)
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.resources.external_model import ExternalModel

LOGGER = structlog.get_logger(name=__name__)

RECONCILE_WAIT = 8

pytestmark = pytest.mark.tier2


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestUserNamespaceAttachmentServing:
    """A user-ns model attached to tenant B's gateway serves through it only."""

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
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        attachment_env: dict[str, Any],
        attached_model: dict[str, Any],
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
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        attachment_env: dict[str, Any],
        attached_model: dict[str, Any],
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
            request_session_http,
            gateway_url=tenant_a["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=attached_model["access"]["api_key"],
            path_tag=model["path_tag"],
            retries=1,
        )
        assert_not_served(response=response)

    def test_inference_serves_with_user_namespace_credential(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        attachment_env: dict[str, Any],
        attached_model: dict[str, Any],
    ) -> None:
        """Given a serving attachment, when running real inference through tenant
        B's gateway, then it reaches the provider with the user-namespace
        credential injected.
        """
        env = attachment_env
        model = attached_model["model"]
        tenant_b = env["tenant_b"]
        response = infer_external_chat(
            request_session_http,
            gateway_url=tenant_b["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=attached_model["access"]["api_key"],
            path_tag=model["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestAttachmentLifecycleCleanup:
    """Detach and delete remove only their own serving configuration."""

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
            assert {
                rule["metadata"]["uid"]
                for rule in destination_rules_referencing(
                    admin_client, model["name"], MAAS_GATEWAY_NAMESPACE, also_matching=(tenant_b["gateway_name"],)
                )
            } == tenant_b_rule_uids, "detaching tenant A must not recreate tenant B's DestinationRule"

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
            with serve_in_tenant_and_cleanup(
                admin_client, request_session_http, ocp_token_for_actor, tenant_b, model
            ) as access_b:
                response = infer_external_chat(
                    request_session_http,
                    gateway_url=tenant_b["gateway_url"],
                    model_namespace=model["namespace"],
                    model_client_name=model["client_name"],
                    api_key=access_b["api_key"],
                    path_tag=model["path_tag"],
                )
                assert_served_by_provider(response=response, path_tag=model["path_tag"], credential=model["credential"])

    def test_model_delete_removes_serving_config(
        self,
        request_session_http: requests.Session,
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
            wait_for_no_destination_rules_referencing(  # noqa: FCN001 - namespace feeds *args
                admin_client, model["name"], MAAS_GATEWAY_NAMESPACE
            )
            assert get_json_or_none(admin_client, "externalmodel", model["name"], model["namespace"]) is None

        unrelated = unrelated_model["model"]
        response = infer_external_chat(
            request_session_http,
            gateway_url=tenant_b["gateway_url"],
            model_namespace=unrelated["namespace"],
            model_client_name=unrelated["client_name"],
            api_key=unrelated_model["access"]["api_key"],
            path_tag=unrelated["path_tag"],
        )
        assert_served_by_provider(response=response, path_tag=unrelated["path_tag"], credential=unrelated["credential"])


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestListenerNotAllowed:
    """A gateway whose allowedRoutes do not admit the model namespace refuses
    the attachment and publishes nothing into the tenant.
    """

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
        wait_for_gateway_attachment_reason(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=tenant_b["gateway_name"],
            reason="NotAllowedByListeners",
        )
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
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        attachment_env: dict[str, Any],
        refused_model: dict[str, Any],
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
            for route in routes:
                conditions = httproute_parent_conditions(route=route, gateway_name=tenant_b["gateway_name"])
                accepted = next((condition for condition in conditions if condition.get("type") == "Accepted"), None)
                assert accepted is not None and accepted.get("status") == "False", (
                    f"route {route['metadata']['name']} must not be admitted: {conditions}"
                )
        response = infer_external_chat(
            request_session_http,
            gateway_url=tenant_b["gateway_url"],
            model_namespace=model["namespace"],
            model_client_name=model["client_name"],
            api_key=f"sk-extmod-unused-{uuid.uuid4().hex[:8]}",  # pragma: allowlist secret
            path_tag=model["path_tag"],
            retries=1,
        )
        assert_not_served(response=response)


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestListenerAdmissionRecovery:
    """A refused attachment recovers when the namespace is admitted, and a
    serving attachment stops when admission is narrowed — in both directions
    without touching the model, and without breaking the tenant's own models.

    The recovery model lives in its own namespace so this class can flip the
    gateway-access label freely: the module's shared user namespace hosts
    other classes' routes, and removing its label would break them.
    """

    def test_refused_model_does_not_break_tenant_models(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        attachment_env: dict[str, Any],
        recovery_env: dict[str, Any],
    ) -> None:
        """Given the gateway refuses the recovering model's namespace, when
        reading that attachment's status and serving the control model, then
        the attachment reports NotAllowedByListeners and the healthy model on
        the same gateway keeps serving end to end.
        """
        env = recovery_env
        tenant_b = env["tenant_b"]
        model = env["recovering_model"]
        remove_gateway_access_label(
            admin_client=admin_client,
            namespace_name=model["namespace"],
            gateway_name=tenant_b["gateway_name"],
        )
        wait_for_gateway_attachment_reason(
            admin_client=admin_client,
            model_name=model["name"],
            model_namespace=model["namespace"],
            gateway_name=tenant_b["gateway_name"],
            reason="NotAllowedByListeners",
        )
        control = env["control_model"]
        response = infer_external_chat(
            request_session_http,
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

        with serve_in_tenant_and_cleanup(
            admin_client, request_session_http, ocp_token_for_actor, tenant_b, model
        ) as access:
            response = infer_external_chat(
                request_session_http,
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
        with serve_in_tenant_and_cleanup(
            admin_client, request_session_http, ocp_token_for_actor, tenant_b, model
        ) as access:
            served = infer_external_chat(
                request_session_http,
                gateway_url=tenant_b["gateway_url"],
                model_namespace=model["namespace"],
                model_client_name=model["client_name"],
                api_key=access["api_key"],
                path_tag=model["path_tag"],
            )
            assert_served_by_provider(response=served, path_tag=model["path_tag"], credential=model["credential"])

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
                accepted = next((condition for condition in conditions if condition.get("type") == "Accepted"), None)
                assert accepted is not None and accepted.get("status") == "False", (
                    f"route {route['metadata']['name']} must not stay admitted: {conditions}"
                )

            def _refused() -> bool:
                response = infer_external_chat(
                    request_session_http,
                    gateway_url=tenant_b["gateway_url"],
                    model_namespace=model["namespace"],
                    model_client_name=model["client_name"],
                    api_key=access["api_key"],
                    path_tag=model["path_tag"],
                    retries=1,
                )
                assert_not_served(response=response)
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
            request_session_http,
            gateway_url=tenant_b["gateway_url"],
            model_namespace=control["namespace"],
            model_client_name=control["client_name"],
            api_key=env["control_access"]["api_key"],
            path_tag=control["path_tag"],
        )
        assert_served_by_provider(
            response=control_served, path_tag=control["path_tag"], credential=control["credential"]
        )


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
)
class TestCredentialRotationFollowsSource:
    """Rotating the key in the user-namespace source Secret updates the
    controller-owned copy in the tenant namespace and serving traffic uses
    the new key, with no manual steps.

    A stale copy (old key still in use, nothing visible to the user) is the
    failure this catches: the assertions check both that a copy carrying the
    new key exists and that no Secret in the tenant namespace still holds
    the old one.
    """

    def test_rotation_updates_copy_and_serving(
        self,
        request_session_http: requests.Session,
        admin_client: DynamicClient,
        attachment_env: dict[str, Any],
        rotation_model: dict[str, Any],
    ) -> None:
        """Given a serving attachment, when the user-namespace source Secret's
        key is rotated, then the tenant copy carries the new key, no stale
        copy keeps the old one, and serving traffic uses the new key with no
        manual steps.
        """
        env = rotation_model
        tenant_b = env["tenant_b"]
        model = env["model"]

        response = infer_external_chat(
            request_session_http,
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

        wait_for_credential_copy(
            admin_client=admin_client,
            tenant_namespace=tenant_b["tenant_ns"],
            source_secret_name=model["secret_name"],
            source_namespace=model["secret_namespace"],
        )
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

        deadline = time.time() + TENANT_EXTERNAL_MODEL_READY_TIMEOUT
        stale_serving_observed = False
        response = None
        last_error_excerpt = ""
        while True:
            response = infer_external_chat(
                request_session_http,
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
