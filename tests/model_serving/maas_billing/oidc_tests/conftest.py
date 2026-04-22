import os
from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import ResourceEditor
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutSampler

from tests.model_serving.maas_billing.oidc_tests.utils import (
    MAAS_API_AUTH_POLICY_NAME,
    OIDC_CLIENT_ID,
    create_oidc_subscription,
    fetch_models_with_header,
    get_maas_oidc_issuer_url,
    request_oidc_access_token,
)
from tests.model_serving.maas_billing.utils import (
    create_api_key,
    revoke_api_key,
)
from utilities.general import generate_random_name
from utilities.resources.auth_policy import AuthPolicy
from utilities.resources.models_as_service import ModelsAsService

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def oidc_subscription(
    admin_client: DynamicClient,
    maas_unprivileged_model_namespace: Any,
    maas_subscription_namespace: Any,
) -> Generator[Any, Any, Any]:
    """Create a MaaSSubscription owned by the MaaS OIDC group for external OIDC tests."""
    yield from create_oidc_subscription(
        admin_client=admin_client,
        subscription_namespace=maas_subscription_namespace.name,
        model_name=f"e2e-authz-model-{generate_random_name()}",
        model_namespace=maas_unprivileged_model_namespace.name,
    )


@pytest.fixture(scope="class")
def oidc_subscription_with_model(
    admin_client: DynamicClient,
    maas_inference_service_tinyllama: Any,
    maas_subscription_namespace: Any,
) -> Generator[Any, Any, Any]:
    """Create a MaaSSubscription referencing the deployed TinyLlama model.

    Used by tests that need ``/v1/models`` to return a real model for inference.
    """
    yield from create_oidc_subscription(
        admin_client=admin_client,
        subscription_namespace=maas_subscription_namespace.name,
        model_name=maas_inference_service_tinyllama.name,
        model_namespace=maas_inference_service_tinyllama.namespace,
    )


@pytest.fixture(scope="class")
def oidc_auth_policy_patched(
    is_byoidc: bool,
    admin_client: DynamicClient,
) -> Generator[None, Any, Any]:
    """Enable OIDC on the ModelsAsService CR so the operator patches the AuthPolicy."""
    if not is_byoidc:
        pytest.skip("External OIDC tests require a byoidc cluster")

    oidc_issuer_url = get_maas_oidc_issuer_url(admin_client=admin_client)
    LOGGER.info(f"oidc_auth_policy_patched: enabling externalOIDC with issuer '{oidc_issuer_url}'")

    maas_cr = ModelsAsService(
        client=admin_client,
        name="default-modelsasservice",
    )
    applications_namespace = py_config["applications_namespace"]

    oidc_patch = {
        "spec": {
            "externalOIDC": {
                "issuerUrl": oidc_issuer_url,
                "clientId": OIDC_CLIENT_ID,
            }
        }
    }

    with ResourceEditor(patches={maas_cr: oidc_patch}):
        maas_auth_policy = AuthPolicy(
            client=admin_client,
            name=MAAS_API_AUTH_POLICY_NAME,
            namespace=applications_namespace,
            ensure_exists=True,
        )
        maas_auth_policy.wait_for_condition(condition="Enforced", status="True", timeout=120)
        LOGGER.info("oidc_auth_policy_patched: operator applied OIDC rules to AuthPolicy")
        yield

    LOGGER.info("oidc_auth_policy_patched: externalOIDC removed, operator restoring AuthPolicy")


@pytest.fixture(scope="class")
def oidc_user_credentials(
    is_byoidc: bool,
) -> dict[str, str]:
    """Read first MaaS OIDC user credentials from environment variables."""
    if not is_byoidc:
        pytest.skip("OIDC user credentials require a byoidc cluster with Keycloak")

    username = os.environ.get("MAAS_OIDC_USER1")
    password = os.environ.get("MAAS_OIDC_PASSWORD1")
    if not username or not password:
        pytest.fail("MAAS_OIDC_USER1 and MAAS_OIDC_PASSWORD1 environment variables must be set", pytrace=False)

    LOGGER.info(f"oidc_user_credentials: using MaaS realm user '{username}'")
    return {"username": username, "password": password}


@pytest.fixture(scope="class")
def oidc_second_user_credentials(
    is_byoidc: bool,
) -> dict[str, str]:
    """Read second MaaS OIDC user credentials from environment variables."""
    if not is_byoidc:
        pytest.skip("OIDC second user credentials require a byoidc cluster with Keycloak")

    username = os.environ.get("MAAS_OIDC_USER2")
    password = os.environ.get("MAAS_OIDC_PASSWORD2")
    if not username or not password:
        pytest.fail("MAAS_OIDC_USER2 and MAAS_OIDC_PASSWORD2 environment variables must be set", pytrace=False)

    LOGGER.info(f"oidc_second_user_credentials: using MaaS realm user '{username}'")
    return {"username": username, "password": password}


@pytest.fixture(scope="class")
def oidc_client_secret(
    is_byoidc: bool,
) -> str:
    """Read the MaaS OIDC client secret from environment variables."""
    if not is_byoidc:
        pytest.skip("OIDC client secret requires a byoidc cluster with Keycloak")

    secret = os.environ.get("MAAS_OIDC_CLIENT_SECRET")
    if not secret:
        pytest.fail("MAAS_OIDC_CLIENT_SECRET environment variable must be set", pytrace=False)
    return secret


@pytest.fixture(scope="class")
def oidc_token_endpoint(
    admin_client: DynamicClient,
) -> str:
    """Resolve the Keycloak token endpoint URL from the MaaS OIDC realm."""
    oidc_issuer_url = get_maas_oidc_issuer_url(admin_client=admin_client)
    return f"{oidc_issuer_url}/protocol/openid-connect/token"


@pytest.fixture(scope="class")
def external_oidc_token(
    request_session_http: requests.Session,
    oidc_token_endpoint: str,
    oidc_user_credentials: dict[str, str],
    oidc_client_secret: str,
    oidc_auth_policy_patched: None,
) -> str:
    """Acquire a fresh OIDC access token from the MaaS Keycloak realm for the first user."""
    access_token = request_oidc_access_token(
        request_session_http=request_session_http,
        token_url=oidc_token_endpoint,
        client_id=OIDC_CLIENT_ID,
        client_secret=oidc_client_secret,
        username=oidc_user_credentials["username"],
        password=oidc_user_credentials["password"],
    )
    LOGGER.info(
        f"external_oidc_token: acquired token for user "
        f"'{oidc_user_credentials['username']}' (length={len(access_token)})"
    )
    return access_token


@pytest.fixture(scope="class")
def second_user_oidc_token(
    request_session_http: requests.Session,
    oidc_token_endpoint: str,
    oidc_second_user_credentials: dict[str, str],
    oidc_client_secret: str,
    oidc_auth_policy_patched: None,
) -> str:
    """Acquire a fresh OIDC access token for the second user (key isolation tests)."""
    access_token = request_oidc_access_token(
        request_session_http=request_session_http,
        token_url=oidc_token_endpoint,
        client_id=OIDC_CLIENT_ID,
        client_secret=oidc_client_secret,
        username=oidc_second_user_credentials["username"],
        password=oidc_second_user_credentials["password"],
    )
    LOGGER.info(
        f"second_user_oidc_token: acquired token for user "
        f"'{oidc_second_user_credentials['username']}' (length={len(access_token)})"
    )
    return access_token


@pytest.fixture(scope="function")
def oidc_minted_api_key(
    request_session_http: requests.Session,
    base_url: str,
    external_oidc_token: str,
) -> Generator[dict[str, Any], Any, Any]:
    """Create an API key using an external OIDC token and revoke it on teardown."""
    key_name = f"e2e-oidc-{generate_random_name()}"
    api_key_body: dict[str, Any] = {}
    for sample in TimeoutSampler(
        wait_timeout=60,
        sleep=5,
        func=create_api_key,
        base_url=base_url,
        ocp_user_token=external_oidc_token,
        request_session_http=request_session_http,
        api_key_name=key_name,
        raise_on_error=False,
    ):
        response, body = sample
        if response.status_code in (200, 201):
            api_key_body = body
            break
        LOGGER.warning(f"oidc_minted_api_key: retrying create, status={response.status_code}")

    if not api_key_body:
        raise AssertionError(f"oidc_minted_api_key: failed to create API key '{key_name}' within timeout")
    LOGGER.info(f"oidc_minted_api_key: created key id={api_key_body['id']} name={key_name}")
    yield api_key_body

    revoke_response, _ = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=api_key_body["id"],
        ocp_user_token=external_oidc_token,
    )
    assert revoke_response.status_code in (200, 404), (
        f"oidc_minted_api_key: unexpected teardown status for key id={api_key_body['id']}: "
        f"{revoke_response.status_code}"
    )


@pytest.fixture(scope="function")
def second_user_minted_api_key(
    request_session_http: requests.Session,
    base_url: str,
    second_user_oidc_token: str,
) -> Generator[dict[str, Any], Any, Any]:
    """Create an API key for the second OIDC user and revoke it on teardown."""
    key_name = f"e2e-oidc-user2-{generate_random_name()}"
    api_key_body: dict[str, Any] = {}
    for sample in TimeoutSampler(
        wait_timeout=60,
        sleep=5,
        func=create_api_key,
        base_url=base_url,
        ocp_user_token=second_user_oidc_token,
        request_session_http=request_session_http,
        api_key_name=key_name,
        raise_on_error=False,
    ):
        response, body = sample
        if response.status_code in (200, 201):
            api_key_body = body
            break
        LOGGER.warning(f"second_user_minted_api_key: retrying create, status={response.status_code}")

    if not api_key_body:
        raise AssertionError(f"second_user_minted_api_key: failed to create API key '{key_name}' within timeout")
    LOGGER.info(f"second_user_minted_api_key: created key id={api_key_body['id']} name={key_name}")
    yield api_key_body

    revoke_response, _ = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=api_key_body["id"],
        ocp_user_token=second_user_oidc_token,
    )
    assert revoke_response.status_code in (200, 404), (
        f"second_user_minted_api_key: unexpected teardown status for key id={api_key_body['id']}: "
        f"{revoke_response.status_code}"
    )


@pytest.fixture(scope="function")
def oidc_revoked_api_key_plaintext(
    request_session_http: requests.Session,
    base_url: str,
    external_oidc_token: str,
) -> str:
    """Create an API key, revoke it immediately, and return the plaintext key."""
    key_name = f"e2e-oidc-revoked-{generate_random_name()}"
    _, api_key_body = create_api_key(
        base_url=base_url,
        ocp_user_token=external_oidc_token,
        request_session_http=request_session_http,
        api_key_name=key_name,
    )
    key_id = api_key_body["id"]
    plaintext_key = api_key_body["key"]

    revoke_response, _ = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=key_id,
        ocp_user_token=external_oidc_token,
    )
    assert revoke_response.status_code == 200, f"Failed to revoke key {key_id}: {revoke_response.status_code}"

    LOGGER.info(f"oidc_revoked_api_key_plaintext: created and revoked key id={key_id}")
    return plaintext_key


@pytest.fixture(scope="function")
def oidc_api_key_with_spoofed_username(
    request_session_http: requests.Session,
    base_url: str,
    external_oidc_token: str,
) -> Generator[dict[str, Any], Any, Any]:
    """Create an API key with a spoofed X-MaaS-Username header and clean up after."""
    key_name = f"e2e-oidc-inject-{generate_random_name()}"
    response = request_session_http.post(
        url=f"{base_url}/v1/api-keys",
        headers={
            "Authorization": f"Bearer {external_oidc_token}",
            "Content-Type": "application/json",
            "X-MaaS-Username": "evil_hacker",
        },
        json={"name": key_name},
        timeout=30,
    )
    assert response.status_code in (200, 201), (
        f"API key creation with spoofed header failed: {response.status_code} {response.text[:300]}"
    )
    api_key_body: dict[str, Any] = response.json()
    LOGGER.info(f"oidc_api_key_with_spoofed_username: created key id={api_key_body['id']}")
    yield api_key_body

    revoke_response, _ = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=api_key_body["id"],
        ocp_user_token=external_oidc_token,
    )
    if revoke_response.status_code not in (200, 404):
        LOGGER.warning(
            f"oidc_api_key_with_spoofed_username: unexpected teardown status for key "
            f"id={api_key_body['id']}: {revoke_response.status_code}"
        )


@pytest.fixture(scope="function")
def baseline_models_response(
    request_session_http: requests.Session,
    base_url: str,
    oidc_minted_api_key: dict[str, Any],
) -> requests.Response:
    """Fetch /v1/models with a valid OIDC-minted API key (no spoofed headers)."""
    models_url = f"{base_url}/v1/models"
    response = fetch_models_with_header(
        session=request_session_http,
        models_url=models_url,
        api_key=oidc_minted_api_key["key"],
    )
    assert response.status_code == 200, f"Baseline models request failed: {response.status_code}"
    return response
