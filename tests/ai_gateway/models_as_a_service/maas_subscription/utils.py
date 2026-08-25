from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.event import Event
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.resource import ResourceEditor
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from utilities.constants import (
    MAAS_GATEWAY_NAME,
    MAAS_GATEWAY_NAMESPACE,
    ApiGroups,
    ModelStorage,
)
from utilities.image_constants import SharedImages
from utilities.llmd_utils import create_llmisvc
from utilities.resources.auth import Auth
from utilities.resources.llm_inference_service import LLMInferenceService

LOGGER = structlog.get_logger(name=__name__)
MAAS_SUBSCRIPTION_NAMESPACE = "models-as-a-service"
MODEL_IDENTITY_UNIQUE_CONDITION: str = "ModelIdentityUnique"
RUNTIME_READY_CONDITION: str = "RuntimeReady"
MODEL_NAME_CONFLICT_EVENT_REASON: str = "ModelNameConflict"
MODEL_NAME_CONFLICT_RESOLVED_EVENT_REASON: str = "ModelNameConflictResolved"
IDENTITY_COLLISION_LLMIS_STORAGE_URI: str = ModelStorage.S3.OPT_125M
IDENTITY_COLLISION_LLMIS_IMAGE: str = SharedImages.VLLM_CPU


class ModelIdentityCollisionNames(TypedDict):
    """Resource names for a model-identity collision test run."""

    shared_model_name: str
    primary_llmis_name: str
    secondary_llmis_name: str


def build_model_identity_collision_names(suffix: str) -> ModelIdentityCollisionNames:
    """Build unique LLMIS and shared model names for identity-collision tests."""
    return {
        "shared_model_name": f"test/e2e-identity-conflict-{suffix}",
        "primary_llmis_name": f"e2e-conflict-a-{suffix}",
        "secondary_llmis_name": f"e2e-conflict-b-{suffix}",
    }


@contextmanager
def patch_llmisvc_with_maas_router_and_tiers(
    llm_service: LLMInferenceService,
    tiers: Sequence[str],
    enable_auth: bool = True,
) -> Generator[None]:
    """
    Patch an LLMInferenceService to use MaaS router (gateway refs + route {})
    and set MaaS tier annotation.

    This is intended for MaaS subscription tests where you want distinct
    tiered models (e.g. free vs premium)

    Examples:
      - tiers=[]              -> open model
      - tiers=["premium"]     -> premium-only
    """
    router_spec = {
        "gateway": {"refs": [{"name": MAAS_GATEWAY_NAME, "namespace": MAAS_GATEWAY_NAMESPACE}]},
        "route": {},
    }

    tiers_val = list(tiers)
    patch_body = {
        "metadata": {
            "annotations": {
                f"alpha.{ApiGroups.MAAS_IO}/tiers": json.dumps(tiers_val),
                "security.opendatahub.io/enable-auth": "true" if enable_auth else "false",
            }
        },
        "spec": {"router": router_spec},
    }

    with ResourceEditor(patches={llm_service: patch_body}):
        yield


def model_id_from_chat_completions_url(model_url: str) -> str:
    path = urlparse(model_url).path.strip("/")
    parts = path.split("/")

    if len(parts) >= 2 and parts[0] == "llm":
        model_id = parts[1]
        if model_id:
            return model_id

    raise AssertionError(f"Cannot extract model id from url: {model_url!r} (path={path!r})")


def chat_payload_for_url(model_url: str, *, prompt: str = "Hello", max_tokens: int = 8) -> dict:
    model_id = model_id_from_chat_completions_url(model_url=model_url)
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }


def poll_expected_status(
    request_session_http: requests.Session,
    model_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    expected_statuses: set[int],
    wait_timeout: int = 240,
    sleep: int = 5,
    request_timeout: int = 60,
) -> requests.Response:
    """
    Poll model endpoint until we see one of `expected_statuses` or timeout.

    Returns the response that matched expected status.
    """
    last_response: requests.Response | None = None
    observed_responses: list[tuple[int | None, str]] = []

    for response in TimeoutSampler(
        wait_timeout=wait_timeout,
        sleep=sleep,
        func=request_session_http.post,
        url=model_url,
        headers=headers,
        json=payload,
        timeout=request_timeout,
    ):
        last_response = response
        status_code = getattr(response, "status_code", None)
        response_text = (getattr(response, "text", "") or "")[:200]

        observed_responses.append((status_code, response_text))

        LOGGER.info(f"Polling model_url={model_url} status={status_code} expected={sorted(expected_statuses)}")

        if status_code in expected_statuses:
            return response

    pytest.fail(
        "Timed out waiting for expected HTTP status. "
        f"model_url={model_url}, "
        f"expected={sorted(expected_statuses)}, "
        f"last_status={getattr(last_response, 'status_code', None)}, "
        f"last_body={(getattr(last_response, 'text', '') or '')[:200]}, "
        f"seen_count={len(observed_responses)}"
    )


def create_maas_subscription(
    admin_client: DynamicClient,
    subscription_namespace: str,
    subscription_name: str,
    owner_group_name: str,
    model_name: str,
    model_namespace: str,
    tokens_per_minute: int,
    window: str = "1m",
    priority: int = 0,
    teardown: bool = True,
    wait_for_resource: bool = True,
) -> MaaSSubscription:

    return MaaSSubscription(
        client=admin_client,
        name=subscription_name,
        namespace=subscription_namespace,
        owner={
            "groups": [{"name": owner_group_name}],
        },
        model_refs=[
            {
                "name": model_name,
                "namespace": model_namespace,
                "tokenRateLimits": [{"limit": tokens_per_minute, "window": window}],
            }
        ],
        priority=priority,
        teardown=teardown,
        wait_for_resource=wait_for_resource,
    )


def wait_for_auth_admin_groups(
    auth: Auth,
    expected_admin_groups: list[str],
    timeout: int = 60,
) -> None:
    """Wait until Auth CR spec.adminGroups matches expected and Ready=True."""
    expected_groups = set(expected_admin_groups)
    for instance in TimeoutSampler(wait_timeout=timeout, sleep=2, func=lambda: auth.instance):
        current_groups = set(instance.spec.adminGroups or [])
        ready_condition = next(
            (
                condition
                for condition in (instance.status or {}).get("conditions") or []
                if condition.get("type") == "Ready"
            ),
            None,
        )
        if current_groups == expected_groups and ready_condition and ready_condition.get("status") == "True":
            return


def assert_models_belong_to_subscription(
    models: list[dict[str, Any]],
    expected_subscription_name: str,
) -> None:
    """Assert every model in the list references the expected subscription."""
    for model_entry in models:
        assert "subscriptions" in model_entry, f"Model '{model_entry.get('id')}' missing 'subscriptions' field"
        bound_sub_names = [sub["name"] for sub in model_entry["subscriptions"]]
        assert expected_subscription_name in bound_sub_names, (
            f"Model '{model_entry.get('id')}' should reference subscription "
            f"'{expected_subscription_name}', got {bound_sub_names}"
        )


def assert_models_response_for_subscription(
    response: requests.Response,
    expected_subscription_name: str,
) -> list[dict[str, Any]]:
    """Assert a 200 /v1/models response contains models from the expected subscription."""
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {(response.text or '')[:200]}"
    data = response.json()
    models: list[dict[str, Any]] = data.get("data") or []
    assert len(models) >= 1, f"Expected at least 1 model, got {len(models)}"
    assert_models_belong_to_subscription(
        models=models,
        expected_subscription_name=expected_subscription_name,
    )
    return models


def fetch_and_assert_models_for_subscription(
    session: requests.Session,
    models_url: str,
    token: str,
    expected_subscription_name: str,
    extra_headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """GET /v1/models and assert all returned models belong to the expected subscription."""
    from tests.ai_gateway.models_as_a_service.utils import build_maas_headers

    headers = build_maas_headers(token=token)
    if extra_headers:
        headers.update(extra_headers)
    response = session.get(url=models_url, headers=headers, timeout=30)
    return assert_models_response_for_subscription(
        response=response,
        expected_subscription_name=expected_subscription_name,
    )


def assert_model_info_schema(model: dict[str, Any]) -> None:
    """Assert a ModelInfo object from /v1/models has the expected structure and field types."""
    assert "id" in model, f"Missing 'id': {model}"
    assert isinstance(model["id"], str), f"'id' must be string, got {type(model['id']).__name__}"
    assert "object" in model, f"Missing 'object': {model}"
    assert model["object"] == "model", f"Expected object='model', got {model['object']!r}"
    assert "created" in model, f"Missing 'created': {model}"
    assert isinstance(model["created"], int), f"'created' must be int, got {type(model['created']).__name__}"
    assert "owned_by" in model, f"Missing 'owned_by': {model}"
    assert isinstance(model["owned_by"], str), f"'owned_by' must be string, got {type(model['owned_by']).__name__}"


def assert_subscription_info_schema(subscription: dict[str, Any]) -> None:
    """Assert a SubscriptionInfo object has the expected structure and field types."""
    assert "subscription_id_header" in subscription, f"Missing subscription_id_header: {subscription}"
    assert isinstance(subscription["subscription_id_header"], str), "subscription_id_header must be string"
    assert "subscription_description" in subscription, f"Missing subscription_description: {subscription}"
    assert isinstance(subscription["subscription_description"], str), "subscription_description must be string"
    assert "priority" in subscription, f"Missing priority: {subscription}"
    assert isinstance(subscription["priority"], int), "priority must be integer"
    assert "model_refs" in subscription, f"Missing model_refs: {subscription}"
    assert isinstance(subscription["model_refs"], list), "model_refs must be a list"
    for model_ref in subscription["model_refs"]:
        assert "name" in model_ref, f"model_ref missing name: {model_ref}"
        assert isinstance(model_ref["name"], str), "model_ref name must be string"
    if "display_name" in subscription:
        assert isinstance(subscription["display_name"], str), "display_name must be string"
    if "organization_id" in subscription:
        assert isinstance(subscription["organization_id"], str), "organization_id must be string"
    if "cost_center" in subscription:
        assert isinstance(subscription["cost_center"], str), "cost_center must be string"
    if "labels" in subscription:
        assert isinstance(subscription["labels"], dict), "labels must be a dict"


def maas_model_ref_instance_as_dict(maas_model_ref: MaaSModelRef) -> dict[str, Any]:
    """Return the current MaaSModelRef API object as a plain dict."""
    instance = maas_model_ref.instance
    if hasattr(instance, "to_dict"):
        return instance.to_dict()
    return dict(instance)


def resolved_model_alias_from_ref(maas_model_ref: MaaSModelRef) -> str | None:
    """Read resolvedModelAlias from a MaaSModelRef using attribute access."""
    status = getattr(maas_model_ref.instance, "status", None)
    if status is None:
        return None
    resolved_alias = getattr(status, "resolvedModelAlias", None)
    return str(resolved_alias) if resolved_alias else None


def status_condition(resource_instance: dict[str, Any], condition_type: str) -> dict[str, Any]:
    """Return the status condition dict for the given condition type."""
    conditions = (resource_instance.get("status") or {}).get("conditions") or []
    for condition in conditions:
        if condition.get("type") == condition_type:
            return condition
    raise AssertionError(f"condition {condition_type!r} not found in {conditions!r}")


def wait_for_model_identity_unique(
    maas_model_ref: MaaSModelRef,
    expected_status: str,
    timeout: int = 300,
) -> dict[str, Any]:
    """Wait until MaaSModelRef reports ModelIdentityUnique with the expected status."""
    maas_model_ref.wait_for_condition(
        condition=MODEL_IDENTITY_UNIQUE_CONDITION,
        status=expected_status,
        timeout=timeout,
    )
    return maas_model_ref_instance_as_dict(maas_model_ref=maas_model_ref)


def maas_model_ref_target_for_llmisvc(llm_service: LLMInferenceService) -> dict[str, str]:
    """Return the modelRef target dict for a MaaSModelRef backed by an LLMInferenceService."""
    return {
        "name": llm_service.name,
        "namespace": llm_service.namespace,
        "kind": "LLMInferenceService",
    }


def wait_for_model_ref_resolved_alias(
    maas_model_ref: MaaSModelRef,
    timeout: int = 300,
) -> str:
    """Wait until MaaSModelRef status includes a non-empty resolvedModelAlias."""
    LOGGER.info(f"[model-identity] waiting for resolvedModelAlias on {maas_model_ref.name}")
    for _sample in TimeoutSampler(
        wait_timeout=timeout,
        sleep=2,
        func=lambda: resolved_model_alias_from_ref(maas_model_ref=maas_model_ref),
    ):
        if _sample:
            return _sample
    pytest.fail(f"Timed out waiting for resolvedModelAlias on MaaSModelRef {maas_model_ref.name}")


def assert_model_ref_identity_unique(
    maas_model_ref: MaaSModelRef,
    expected_status: str,
    timeout: int = 300,
) -> dict[str, Any]:
    """Assert a MaaSModelRef reports the expected ModelIdentityUnique status."""
    resource_instance = wait_for_model_identity_unique(
        maas_model_ref=maas_model_ref,
        expected_status=expected_status,
        timeout=timeout,
    )
    if expected_status == "True":
        resolved_alias = wait_for_model_ref_resolved_alias(
            maas_model_ref=maas_model_ref,
            timeout=timeout,
        )
        LOGGER.info(f"[model-identity] {maas_model_ref.name} unique with alias={resolved_alias!r}")
    return resource_instance


def assert_model_refs_report_identity_collision(
    primary_model_ref: MaaSModelRef,
    secondary_model_ref: MaaSModelRef,
) -> None:
    """Assert two colliding MaaSModelRefs both report ModelIdentityUnique=False and cross-reference."""
    primary_instance = wait_for_model_identity_unique(
        maas_model_ref=primary_model_ref,
        expected_status="False",
    )
    secondary_instance = wait_for_model_identity_unique(
        maas_model_ref=secondary_model_ref,
        expected_status="False",
    )
    primary_condition = status_condition(
        resource_instance=primary_instance,
        condition_type=MODEL_IDENTITY_UNIQUE_CONDITION,
    )
    secondary_condition = status_condition(
        resource_instance=secondary_instance,
        condition_type=MODEL_IDENTITY_UNIQUE_CONDITION,
    )
    assert "message" in primary_condition, (
        f"Expected 'message' on {primary_model_ref.name} {MODEL_IDENTITY_UNIQUE_CONDITION} condition"
    )
    assert "message" in secondary_condition, (
        f"Expected 'message' on {secondary_model_ref.name} {MODEL_IDENTITY_UNIQUE_CONDITION} condition"
    )
    assert secondary_model_ref.name in primary_condition["message"], (
        f"Expected {primary_model_ref.name}'s condition to name {secondary_model_ref.name}: "
        f"{primary_condition['message']!r}"
    )
    assert primary_model_ref.name in secondary_condition["message"], (
        f"Expected {secondary_model_ref.name}'s condition to name {primary_model_ref.name}: "
        f"{secondary_condition['message']!r}"
    )
    LOGGER.info(f"[model-identity] collision detected between {primary_model_ref.name} and {secondary_model_ref.name}")


def assert_model_refs_runtime_ready(
    primary_model_ref: MaaSModelRef,
    secondary_model_ref: MaaSModelRef,
    timeout: int = 300,
) -> None:
    """Assert colliding MaaSModelRefs remain runtime-healthy while identity is non-unique."""
    for model_ref in (primary_model_ref, secondary_model_ref):
        model_ref.wait_for_condition(
            condition=RUNTIME_READY_CONDITION,
            status="True",
            timeout=timeout,
        )
    LOGGER.info(
        f"[model-identity] {primary_model_ref.name} and {secondary_model_ref.name} "
        f"report {RUNTIME_READY_CONDITION}=True during collision"
    )


def remove_collision_model_pair(
    admin_client: DynamicClient,
    namespace: str,
    collision_names: ModelIdentityCollisionNames,
    primary: bool,
) -> None:
    """Delete one LLMIS + MaaSModelRef pair from a model-identity collision scenario."""
    llmis_name = collision_names["primary_llmis_name"] if primary else collision_names["secondary_llmis_name"]
    role = "primary" if primary else "secondary"
    LOGGER.info(f"[model-identity] removing {role} collision pair {llmis_name}")

    model_ref = MaaSModelRef(
        client=admin_client,
        name=llmis_name,
        namespace=namespace,
        ensure_exists=True,
    )
    if model_ref.exists:
        model_ref.clean_up(wait=True)

    llm_service = LLMInferenceService(
        client=admin_client,
        name=llmis_name,
        namespace=namespace,
        ensure_exists=True,
    )
    if llm_service.exists:
        llm_service.clean_up(wait=True)


def _event_as_dict(event: Any) -> dict[str, Any]:
    """Return a Kubernetes Event as a plain dict."""
    if hasattr(event, "to_dict"):
        return event.to_dict()
    return dict(event)


def events_for_model_ref(
    admin_client: DynamicClient,
    namespace: str,
    model_ref_name: str,
    reason: str,
) -> list[dict[str, Any]]:
    """Return Kubernetes Events for a MaaSModelRef name and event reason."""
    events = Event.list(
        client=admin_client,
        namespace=namespace,
        field_selector=f"involvedObject.name={model_ref_name}",
    )
    if not events:
        return []
    return [
        event_dict for event in events for event_dict in [_event_as_dict(event)] if event_dict.get("reason") == reason
    ]


def wait_for_model_ref_event(
    admin_client: DynamicClient,
    namespace: str,
    model_ref_name: str,
    reason: str,
    timeout: int = 120,
) -> list[dict[str, Any]]:
    """Wait until a Kubernetes Event with the given reason exists for a MaaSModelRef."""
    LOGGER.info(f"[model-identity] waiting for {reason} event on {model_ref_name}")

    def matching_events() -> list[dict[str, Any]] | None:
        events = events_for_model_ref(
            admin_client=admin_client,
            namespace=namespace,
            model_ref_name=model_ref_name,
            reason=reason,
        )
        return events or None

    try:
        for events in TimeoutSampler(
            wait_timeout=timeout,
            sleep=2,
            func=matching_events,
        ):
            if events:
                return events
    except TimeoutExpiredError:
        pass
    return []


def assert_model_name_conflict_event_present(
    admin_client: DynamicClient,
    namespace: str,
    primary_model_ref_name: str,
    secondary_model_ref_name: str,
    timeout: int = 120,
) -> None:
    """Assert at least one colliding MaaSModelRef emitted a ModelNameConflict event."""
    LOGGER.info(
        f"[model-identity] waiting for {MODEL_NAME_CONFLICT_EVENT_REASON} event on "
        f"{primary_model_ref_name} or {secondary_model_ref_name}"
    )

    def conflict_events_present() -> bool:
        primary_events = events_for_model_ref(
            admin_client=admin_client,
            namespace=namespace,
            model_ref_name=primary_model_ref_name,
            reason=MODEL_NAME_CONFLICT_EVENT_REASON,
        )
        if primary_events:
            return True
        secondary_events = events_for_model_ref(
            admin_client=admin_client,
            namespace=namespace,
            model_ref_name=secondary_model_ref_name,
            reason=MODEL_NAME_CONFLICT_EVENT_REASON,
        )
        return bool(secondary_events)

    try:
        for found in TimeoutSampler(
            wait_timeout=timeout,
            sleep=2,
            func=conflict_events_present,
        ):
            if found:
                return
    except TimeoutExpiredError:
        pass
    pytest.fail(
        "Expected a ModelNameConflict event on at least one colliding MaaSModelRef "
        f"({primary_model_ref_name}, {secondary_model_ref_name})"
    )


def assert_model_name_conflict_resolved_event_present(
    admin_client: DynamicClient,
    namespace: str,
    model_ref_name: str,
    timeout: int = 120,
) -> None:
    """Assert the surviving MaaSModelRef emits a ModelNameConflictResolved event."""
    resolved_events = wait_for_model_ref_event(
        admin_client=admin_client,
        namespace=namespace,
        model_ref_name=model_ref_name,
        reason=MODEL_NAME_CONFLICT_RESOLVED_EVENT_REASON,
        timeout=timeout,
    )
    if resolved_events:
        LOGGER.info(f"[model-identity] {model_ref_name} unique again after sibling removal")
        return
    pytest.fail(
        f"Expected a {MODEL_NAME_CONFLICT_RESOLVED_EVENT_REASON} event on {model_ref_name} after sibling removal"
    )


@contextmanager
def maas_identity_collision_model_ref(
    admin_client: DynamicClient,
    namespace: str,
    service_account: str,
    collision_names: ModelIdentityCollisionNames,
    primary: bool,
) -> Generator[MaaSModelRef]:
    """Create one LLMIS + MaaSModelRef pair for a model-identity collision scenario."""
    llmis_name = collision_names["primary_llmis_name"] if primary else collision_names["secondary_llmis_name"]
    with maas_llmisvc_and_model_ref(
        admin_client=admin_client,
        llmis_name=llmis_name,
        namespace=namespace,
        service_account=service_account,
        model_name=collision_names["shared_model_name"],
    ) as (_llm_service, model_ref):
        yield model_ref


def assert_model_identity_collision_preserves_runtime_ready(
    admin_client: DynamicClient,
    namespace: str,
    service_account: str,
    collision_names: ModelIdentityCollisionNames,
) -> None:
    """Assert ModelIdentityUnique=False does not clear RuntimeReady on colliding MaaSModelRefs."""
    with maas_identity_collision_model_ref(
        admin_client=admin_client,
        namespace=namespace,
        service_account=service_account,
        collision_names=collision_names,
        primary=True,
    ) as primary_model_ref:
        assert_model_ref_identity_unique(
            maas_model_ref=primary_model_ref,
            expected_status="True",
        )

        with maas_identity_collision_model_ref(
            admin_client=admin_client,
            namespace=namespace,
            service_account=service_account,
            collision_names=collision_names,
            primary=False,
        ) as secondary_model_ref:
            assert_model_refs_report_identity_collision(
                primary_model_ref=primary_model_ref,
                secondary_model_ref=secondary_model_ref,
            )
            assert_model_refs_runtime_ready(
                primary_model_ref=primary_model_ref,
                secondary_model_ref=secondary_model_ref,
            )


def assert_model_identity_collision_detected_and_resolved(
    admin_client: DynamicClient,
    namespace: str,
    service_account: str,
    collision_names: ModelIdentityCollisionNames,
    survivor: Literal["primary", "secondary"] = "primary",
) -> None:
    """Assert colliding MaaSModelRefs are flagged, emit events, and recover after sibling removal."""
    with maas_identity_collision_model_ref(
        admin_client=admin_client,
        namespace=namespace,
        service_account=service_account,
        collision_names=collision_names,
        primary=True,
    ) as primary_model_ref:
        assert_model_ref_identity_unique(
            maas_model_ref=primary_model_ref,
            expected_status="True",
        )
        LOGGER.info(f"[model-identity] primary {primary_model_ref.name} verified unique; creating colliding sibling")

        with maas_identity_collision_model_ref(
            admin_client=admin_client,
            namespace=namespace,
            service_account=service_account,
            collision_names=collision_names,
            primary=False,
        ) as secondary_model_ref:
            assert_model_refs_report_identity_collision(
                primary_model_ref=primary_model_ref,
                secondary_model_ref=secondary_model_ref,
            )
            assert_model_name_conflict_event_present(
                admin_client=admin_client,
                namespace=namespace,
                primary_model_ref_name=primary_model_ref.name,
                secondary_model_ref_name=secondary_model_ref.name,
            )

            if survivor == "secondary":
                remove_collision_model_pair(
                    admin_client=admin_client,
                    namespace=namespace,
                    collision_names=collision_names,
                    primary=True,
                )
                assert_model_ref_identity_unique(
                    maas_model_ref=secondary_model_ref,
                    expected_status="True",
                )
                assert_model_name_conflict_resolved_event_present(
                    admin_client=admin_client,
                    namespace=namespace,
                    model_ref_name=secondary_model_ref.name,
                )
                return

        assert_model_ref_identity_unique(
            maas_model_ref=primary_model_ref,
            expected_status="True",
        )
        assert_model_name_conflict_resolved_event_present(
            admin_client=admin_client,
            namespace=namespace,
            model_ref_name=primary_model_ref.name,
        )


@contextmanager
def create_maas_routed_llmisvc(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    service_account: str,
    model_name: str,
    storage_uri: str = IDENTITY_COLLISION_LLMIS_STORAGE_URI,
    container_image: str = IDENTITY_COLLISION_LLMIS_IMAGE,
) -> Generator[LLMInferenceService]:
    """Create a Ready LLMInferenceService patched for MaaS gateway routing."""
    with (
        create_llmisvc(
            client=admin_client,
            name=name,
            namespace=namespace,
            storage_uri=storage_uri,
            container_image=container_image,
            container_resources={
                "limits": {"cpu": "2", "memory": "8Gi"},
                "requests": {"cpu": "1", "memory": "4Gi"},
            },
            service_account=service_account,
            model_name=model_name,
            wait=False,
            timeout=900,
        ) as llm_service,
        patch_llmisvc_with_maas_router_and_tiers(llm_service=llm_service, tiers=[]),
    ):
        llm_service.wait_for_condition(condition="Ready", status="True", timeout=900)
        yield llm_service


@contextmanager
def maas_llmisvc_and_model_ref(
    admin_client: DynamicClient,
    llmis_name: str,
    namespace: str,
    service_account: str,
    model_name: str,
) -> Generator[tuple[LLMInferenceService, MaaSModelRef]]:
    """Create a Ready LLMInferenceService and matching MaaSModelRef for identity-collision tests."""
    with (
        create_maas_routed_llmisvc(
            admin_client=admin_client,
            name=llmis_name,
            namespace=namespace,
            service_account=service_account,
            model_name=model_name,
        ) as llm_service,
        MaaSModelRef(
            client=admin_client,
            name=llmis_name,
            namespace=namespace,
            model_ref=maas_model_ref_target_for_llmisvc(llm_service=llm_service),
            teardown=True,
            wait_for_resource=True,
        ) as model_ref,
    ):
        yield llm_service, model_ref
