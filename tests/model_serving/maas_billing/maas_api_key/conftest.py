from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.cron_job import CronJob
from ocp_resources.deployment import Deployment
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from pytest_testconfig import config as py_config

from tests.model_serving.maas_billing.maas_api_key.utils import resolve_api_key_username
from tests.model_serving.maas_billing.maas_subscription.utils import (
    wait_for_auth_ready,
)
from tests.model_serving.maas_billing.utils import (
    assert_api_key_created_ok,
    create_and_yield_api_key_id,
    create_api_key,
    revoke_api_key,
)
from utilities.general import generate_random_name
from utilities.infra import get_openshift_token
from utilities.resources.auth import Auth

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="function")
def two_active_api_key_ids(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
) -> Generator[list[str], Any, Any]:
    """Create two active API keys and return their IDs for list tests."""
    ids = [
        create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=f"e2e-fixture-list-{index}-{generate_random_name()}",
        )[1]["id"]
        for index in range(1, 3)
    ]
    LOGGER.info(f"two_active_api_key_ids: created keys {ids}")
    yield ids
    for key_id in ids:
        LOGGER.info(f"Fixture teardown: revoking key {key_id}")
        revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=key_id,
            ocp_user_token=ocp_token_for_actor,
        )


@pytest.fixture(scope="function")
def three_active_api_key_ids(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
) -> Generator[list[str], Any, Any]:
    """Create three active API keys and yield their IDs for bulk-revoke tests."""
    key_ids = [
        create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=f"e2e-bulk-key-{index}-{generate_random_name()}",
        )[1]["id"]
        for index in range(1, 4)
    ]
    LOGGER.info(f"three_active_api_key_ids: created keys {key_ids}")
    yield key_ids
    for key_id in key_ids:
        LOGGER.info(f"three_active_api_key_ids: teardown revoking key {key_id}")
        revoke_resp, _ = revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=key_id,
            ocp_user_token=ocp_token_for_actor,
        )
        if revoke_resp.status_code not in (200, 404):
            raise AssertionError(f"Unexpected teardown status for key id={key_id}: {revoke_resp.status_code}")


@pytest.fixture(scope="function")
def active_api_key_id(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
) -> Generator[str, Any, Any]:
    """Create a single active API key and return its ID for revoke tests."""
    yield from create_and_yield_api_key_id(
        request_session_http=request_session_http,
        base_url=base_url,
        ocp_user_token=ocp_token_for_actor,
        key_name_prefix="e2e-fixture-key",
    )


@pytest.fixture(scope="function")
def free_user_username(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    active_api_key_id: str,
) -> str:
    """Resolve and return the free (non-admin) actor's username from their active API key."""
    username = resolve_api_key_username(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=active_api_key_id,
        ocp_user_token=ocp_token_for_actor,
    )
    LOGGER.info(f"free_user_username: resolved username from key id={active_api_key_id}")
    return username


@pytest.fixture(scope="function")
def admin_username(
    request_session_http: requests.Session,
    base_url: str,
    admin_ocp_token: str,
    admin_active_api_key_id: str,
) -> str:
    """Resolve and return the admin actor's username from their active API key."""
    username = resolve_api_key_username(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=admin_active_api_key_id,
        ocp_user_token=admin_ocp_token,
    )
    LOGGER.info(f"admin_username: resolved username from key id={admin_active_api_key_id}")
    return username


@pytest.fixture(scope="function")
def admin_active_api_key_id(
    request_session_http: requests.Session,
    base_url: str,
    admin_ocp_token: str,
) -> Generator[str, Any, Any]:
    """Create an active API key as the admin user, yield its ID, and revoke on teardown."""
    yield from create_and_yield_api_key_id(
        request_session_http=request_session_http,
        base_url=base_url,
        ocp_user_token=admin_ocp_token,
        key_name_prefix="e2e-authz-admin",
    )


@pytest.fixture(scope="class")
def admin_ocp_token(admin_client: DynamicClient) -> Generator[str, Any, Any]:
    """Temporarily adds dedicated-admins to Auth CR adminGroups so the admin token is recognised by MaaS."""
    auth = Auth(client=admin_client, name="auth")
    current_groups: list[str] = list(auth.instance.spec.adminGroups or [])
    patched_groups = list(set(current_groups + ["dedicated-admins"]))

    auth_conditions = (auth.instance.status or {}).get("conditions") or []
    ready_before = next(
        (condition for condition in auth_conditions if condition.get("type") == "Ready"),
        {},
    )
    baseline_time: str = ready_before.get("lastTransitionTime", "")

    LOGGER.info(f"admin_ocp_token: patching Auth CR adminGroups to {patched_groups}")
    with ResourceEditor(patches={auth: {"spec": {"adminGroups": patched_groups}}}):
        wait_for_auth_ready(auth=auth, baseline_time=baseline_time)
        auth_conditions_after = (auth.instance.status or {}).get("conditions") or []
        ready_after = next(
            (condition for condition in auth_conditions_after if condition.get("type") == "Ready"),
            {},
        )
        cleanup_baseline_time: str = ready_after.get("lastTransitionTime", "")
        yield get_openshift_token(client=admin_client)

    wait_for_auth_ready(auth=auth, baseline_time=cleanup_baseline_time)


@pytest.fixture(scope="function")
def revoked_api_key_id(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    active_api_key_id: str,
) -> str:
    """Revoke the active API key and return its ID.

    Asserts the DELETE response confirms status='revoked'.
    Used as a precondition fixture for tests that verify revoked state persists.
    """
    revoke_resp, revoke_body = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=active_api_key_id,
        ocp_user_token=ocp_token_for_actor,
    )
    assert revoke_resp.status_code == 200, (
        f"Expected 200 on DELETE /v1/api-keys/{active_api_key_id}, "
        f"got {revoke_resp.status_code}: {revoke_resp.text[:200]}"
    )
    assert revoke_body.get("status") == "revoked", f"Expected status='revoked' in DELETE response, got: {revoke_body}"
    LOGGER.info(f"revoked_api_key_id: revoked key id={active_api_key_id}")
    return active_api_key_id


@pytest.fixture(scope="function")
def short_expiration_api_key_id(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
) -> Generator[str, Any, Any]:
    """Create an API key with 1-hour expiration, yield its ID, and revoke on teardown."""
    yield from create_and_yield_api_key_id(
        request_session_http=request_session_http,
        base_url=base_url,
        ocp_user_token=ocp_token_for_actor,
        key_name_prefix="e2e-exp-short",
        expires_in="1h",
    )


@pytest.fixture()
def maas_cleanup_cronjob(
    admin_client: DynamicClient,
) -> CronJob:
    """Return the maas-api-key-cleanup CronJob, asserting it exists."""
    applications_namespace = py_config["applications_namespace"]
    cronjob = CronJob(
        client=admin_client,
        name="maas-api-key-cleanup",
        namespace=applications_namespace,
    )
    assert cronjob.exists, f"CronJob maas-api-key-cleanup not found in {applications_namespace}"
    return cronjob


@pytest.fixture()
def maas_cleanup_networkpolicy(
    admin_client: DynamicClient,
) -> NetworkPolicy:
    """Return the maas-api-cleanup-restrict NetworkPolicy, asserting it exists."""
    applications_namespace = py_config["applications_namespace"]
    network_policy = NetworkPolicy(
        client=admin_client,
        name="maas-api-cleanup-restrict",
        namespace=applications_namespace,
    )
    assert network_policy.exists, f"NetworkPolicy maas-api-cleanup-restrict not found in {applications_namespace}"
    return network_policy


@pytest.fixture()
def maas_api_pod_name(
    admin_client: DynamicClient,
) -> str:
    """Return the name of the single running maas-api pod (exactly one pod is expected)."""
    applications_namespace = py_config["applications_namespace"]
    # Derive the pod label selector from the Deployment itself — avoids hardcoding labels
    # that may differ between operator versions or environments.
    deployment = Deployment(client=admin_client, name="maas-api", namespace=applications_namespace)
    assert deployment.exists, f"Deployment maas-api not found in {applications_namespace}"
    match_labels = deployment.instance.spec.selector.matchLabels
    label_selector = ",".join(f"{k}={v}" for k, v in match_labels.items())
    pods = list(
        Pod.get(
            client=admin_client,
            namespace=applications_namespace,
            label_selector=label_selector,
        )
    )
    assert len(pods) == 1, f"Expected exactly 1 maas-api pod in {applications_namespace}, found {len(pods)}"
    assert pods[0].instance.status.phase == "Running", (
        f"maas-api pod '{pods[0].name}' is not Running (phase={pods[0].instance.status.phase})"
    )
    return pods[0].name


@pytest.fixture()
def ephemeral_api_key(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
) -> Generator[dict[str, Any], Any, Any]:
    """Create an ephemeral API key and revoke it on teardown."""
    creation_response, api_key_data = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_token_for_actor,
        request_session_http=request_session_http,
        api_key_name=f"e2e-ephemeral-{generate_random_name()}",
        expires_in="1h",
        ephemeral=True,
        raise_on_error=False,
    )
    assert_api_key_created_ok(resp=creation_response, body=api_key_data, required_fields=("key", "id"))
    LOGGER.info(
        f"[ephemeral] Created ephemeral key: id={api_key_data['id']}, expiresAt={api_key_data.get('expiresAt')}"
    )
    yield api_key_data
    revoke_response, _ = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=api_key_data["id"],
        ocp_user_token=ocp_token_for_actor,
    )
    if revoke_response.status_code not in (200, 404):
        raise AssertionError(
            f"Unexpected teardown status for ephemeral key id={api_key_data['id']}: {revoke_response.status_code}"
        )


@pytest.fixture(scope="function")
def active_api_key_with_plaintext(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
) -> Generator[dict[str, Any], Any, Any]:
    """Create an API key, yield the full response including plaintext key, and revoke on teardown."""
    key_name = f"e2e-auth-policy-{generate_random_name()}"
    _, api_key_data = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_token_for_actor,
        request_session_http=request_session_http,
        api_key_name=key_name,
    )
    LOGGER.info(f"active_api_key_with_plaintext: created key id={api_key_data['id']} name={key_name}")
    yield api_key_data

    revoke_response, _ = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=api_key_data["id"],
        ocp_user_token=ocp_token_for_actor,
    )
    if revoke_response.status_code not in (200, 404):
        raise AssertionError(
            f"Unexpected teardown status for key id={api_key_data['id']}: {revoke_response.status_code}"
        )


@pytest.fixture(scope="function")
def api_key_for_free_model_listing(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    maas_subscription_tinyllama_free: MaaSSubscription,
) -> Generator[str, Any, Any]:
    """API key bound to the free TinyLlama subscription at mint time. Revoked on teardown."""
    creation_response, body = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_token_for_actor,
        request_session_http=request_session_http,
        api_key_name=f"e2e-list-models-{generate_random_name()}",
        subscription=maas_subscription_tinyllama_free.name,
        raise_on_error=False,
    )
    assert_api_key_created_ok(resp=creation_response, body=body, required_fields=("key", "id"))
    yield body["key"]
    revoke_response, _ = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=body["id"],
        ocp_user_token=ocp_token_for_actor,
    )
    if revoke_response.status_code not in (200, 404):
        raise AssertionError(f"Unexpected teardown status for key id={body['id']}: {revoke_response.status_code}")


@pytest.fixture(scope="function")
def unconfigured_model_ref(
    admin_client: DynamicClient,
    maas_inference_service_tinyllama_free: LLMInferenceService,
    maas_unprivileged_model_namespace: Namespace,
) -> Generator[MaaSModelRef, Any, Any]:
    """Create a MaaSModelRef with no MaaSAuthPolicy and tear it down afterwards.

    Used to verify that the gateway-level deny-by-default policy blocks
    unauthenticated access to models that have no explicit auth policy.
    """
    model_ref_name = f"e2e-unconfigured-{generate_random_name()}"

    with MaaSModelRef(
        client=admin_client,
        name=model_ref_name,
        namespace=maas_unprivileged_model_namespace.name,
        model_ref={
            "name": maas_inference_service_tinyllama_free.name,
            "namespace": maas_inference_service_tinyllama_free.namespace,
            "kind": "LLMInferenceService",
        },
        teardown=True,
        wait_for_resource=True,
    ) as model_ref:
        model_ref.wait_for_condition(condition="Ready", status="True", timeout=300)
        LOGGER.info(f"unconfigured_model_ref: created model ref '{model_ref_name}' (no MaaSAuthPolicy)")
        yield model_ref
