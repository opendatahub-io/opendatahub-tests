import base64
from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutSampler

from tests.model_serving.maas_billing.oidc_tests.utils import (
    MAAS_API_AUTH_POLICY_NAME,
    OIDC_CLIENT_ID,
    fetch_models_with_header,
    request_oidc_access_token,
)
from tests.model_serving.maas_billing.utils import (
    create_api_key,
    revoke_api_key,
)
from utilities.general import generate_random_name
from utilities.resources.auth_policy import AuthPolicy
from utilities.resources.models_as_service import ModelsAsService
from utilities.user_utils import get_byoidc_issuer_url

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def oidc_auth_policy_patched(
    is_byoidc: bool,
    admin_client: DynamicClient,
) -> Generator[None, Any, Any]:
    """Enable OIDC on the ModelsAsService CR so the operator patches the AuthPolicy."""
    if not is_byoidc:
        pytest.skip("External OIDC tests require a byoidc cluster")

    oidc_issuer_url = get_byoidc_issuer_url(admin_client=admin_client)
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
    admin_client: DynamicClient,
) -> dict[str, str]:
    """Read OIDC user credentials from the cluster's byoidc-credentials Secret."""
    if not is_byoidc:
        pytest.skip("OIDC user credentials require a byoidc cluster")

    credentials_secret = Secret(
        client=admin_client,
        name="byoidc-credentials",
        namespace="oidc",
        ensure_exists=True,
    )
    credential_data = credentials_secret.instance.data

    user_names = base64.b64decode(credential_data.users).decode().split(",")
    passwords = base64.b64decode(credential_data.passwords).decode().split(",")

    assert user_names and user_names != [""], "No usernames found in byoidc-credentials secret"
    assert passwords and passwords != [""], "No passwords found in byoidc-credentials secret"
    assert len(user_names) == len(passwords), (
        f"Mismatched credential counts: {len(user_names)} users, {len(passwords)} passwords"
    )

    non_admin_users = [
        (username, password)
        for username, password in zip(user_names, passwords, strict=True)
        if not username.startswith("odh-admin")
    ]
    if non_admin_users:
        selected_username, selected_password = non_admin_users[0]
    else:
        selected_username, selected_password = user_names[0], passwords[0]

    LOGGER.info(f"oidc_user_credentials: using byoidc user '{selected_username}'")
    return {"username": selected_username, "password": selected_password}


@pytest.fixture(scope="class")
def oidc_second_user_credentials(
    is_byoidc: bool,
    admin_client: DynamicClient,
    oidc_user_credentials: dict[str, str],
) -> dict[str, str]:
    """Read a different OIDC user's credentials for key isolation tests.

    Picks the first user that is different from ``oidc_user_credentials``
    to ensure cross-user key isolation can be tested on any cluster.
    """
    if not is_byoidc:
        pytest.skip("OIDC second user credentials require a byoidc cluster")

    first_username = oidc_user_credentials["username"]

    credentials_secret = Secret(
        client=admin_client,
        name="byoidc-credentials",
        namespace="oidc",
        ensure_exists=True,
    )
    credential_data = credentials_secret.instance.data

    user_names = base64.b64decode(credential_data.users).decode().split(",")
    passwords = base64.b64decode(credential_data.passwords).decode().split(",")

    non_admin_pairs = [
        (username, password)
        for username, password in zip(user_names, passwords, strict=True)
        if username != first_username and not username.startswith("odh-admin")
    ]

    fallback_pairs = [
        (username, password)
        for username, password in zip(user_names, passwords, strict=True)
        if username != first_username
    ]

    selected_pairs = non_admin_pairs or fallback_pairs
    assert selected_pairs, f"Need at least 2 different users for isolation tests. Only found: {first_username}"

    selected_username, selected_password = selected_pairs[0]
    LOGGER.info(
        f"oidc_second_user_credentials: using byoidc user '{selected_username}' (different from '{first_username}')"
    )
    return {"username": selected_username, "password": selected_password}


@pytest.fixture(scope="class")
def oidc_token_endpoint(
    admin_client: DynamicClient,
) -> str:
    """Resolve the Keycloak token endpoint URL from the cluster's OIDC issuer."""
    oidc_issuer_url = get_byoidc_issuer_url(admin_client=admin_client)
    return f"{oidc_issuer_url}/protocol/openid-connect/token"


@pytest.fixture(scope="class")
def external_oidc_token(
    request_session_http: requests.Session,
    oidc_token_endpoint: str,
    oidc_user_credentials: dict[str, str],
    oidc_auth_policy_patched: None,
) -> str:
    """Acquire a fresh OIDC access token from the cluster's Keycloak for the first user."""
    access_token = request_oidc_access_token(
        request_session_http=request_session_http,
        token_url=oidc_token_endpoint,
        client_id=OIDC_CLIENT_ID,
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
    oidc_auth_policy_patched: None,
) -> str:
    """Acquire a fresh OIDC access token for the second user (key isolation tests)."""
    access_token = request_oidc_access_token(
        request_session_http=request_session_http,
        token_url=oidc_token_endpoint,
        client_id=OIDC_CLIENT_ID,
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
