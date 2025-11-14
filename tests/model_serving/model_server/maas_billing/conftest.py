from typing import Generator

import pytest
import requests
from simple_logger.logger import get_logger
from utilities.plugins.constant import RestHeader, OpenAIEnpoints

from kubernetes.dynamic import DynamicClient
from ocp_resources.infrastructure import Infrastructure
from ocp_resources.oauth import OAuth
from ocp_resources.resource import ResourceEditor
from pyhelper_utils.shell import run_command
from utilities.general import generate_random_name
from utilities.user_utils import UserTestSession, wait_for_user_creation
from utilities.infra import login_with_user_password, get_openshift_token
from pathlib import Path


from tests.model_serving.model_server.maas_billing.utils import (
    detect_scheme_via_llmisvc,
    host_from_ingress_domain,
    mint_token,
    llmis_name,
    create_maas_group,
    wait_for_oauth_openshift_deployment,
    make_bcrypt_htpasswd_file_with_users,
    login_with_retry,
)


LOGGER = get_logger(name=__name__)
MODELS_INFO = OpenAIEnpoints.MODELS_INFO
CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS

MAAS_FREE_GROUP = "maas-free-users"
MAAS_PREMIUM_GROUP = "maas-premium-users"


@pytest.fixture(scope="session")
def request_session_http() -> Generator[requests.Session, None, None]:
    session = requests.Session()
    session.headers.update({"User-Agent": "odh-maas-billing-tests/1"})
    session.verify = False
    yield session
    session.close()


@pytest.fixture(scope="class")
def minted_token(request_session_http, base_url: str, current_client_token: str) -> str:
    """Mint a MaaS token once per test class and reuse it."""
    resp, body = mint_token(
        base_url=base_url,
        oc_user_token=current_client_token,
        minutes=30,
        http_session=request_session_http,
    )
    LOGGER.info("Mint token response status=%s", resp.status_code)
    assert resp.status_code in (200, 201), f"mint failed: {resp.status_code} {resp.text[:200]}"
    token = body.get("token", "")
    assert isinstance(token, str) and len(token) > 10, f"no usable token in response: {body}"
    LOGGER.info(f"Minted MaaS token len={len(token)}")
    return token


@pytest.fixture(scope="module")
def base_url(admin_client) -> str:
    scheme = detect_scheme_via_llmisvc(client=admin_client)
    host = host_from_ingress_domain(client=admin_client)
    return f"{scheme}://{host}/maas-api"


@pytest.fixture(scope="session")
def model_url(admin_client) -> str:
    """
    MODEL_URL:http(s)://<host>/llm/<deployment>/v1/chat/completions
    """
    scheme = detect_scheme_via_llmisvc(client=admin_client)
    host = host_from_ingress_domain(client=admin_client)
    deployment = llmis_name(client=admin_client)
    return f"{scheme}://{host}/llm/{deployment}{CHAT_COMPLETIONS}"


@pytest.fixture
def maas_headers(minted_token: str) -> dict:
    """Common headers for MaaS API calls."""
    return {"Authorization": f"Bearer {minted_token}", **RestHeader.HEADERS}


@pytest.fixture
def maas_models(
    request_session_http: requests.Session,
    base_url: str,
    maas_headers: dict,
):
    """
    Call /v1/models once and return the list of models.

    """
    models_url = f"{base_url}{MODELS_INFO}"
    resp = request_session_http.get(models_url, headers=maas_headers, timeout=60)

    assert resp.status_code == 200, f"/v1/models failed: {resp.status_code} {resp.text[:200]}"

    models = resp.json().get("data", [])
    assert models, "no models available"
    return models


@pytest.fixture(scope="session")
def maas_api_server_url(admin_client: DynamicClient) -> str:
    """
    Get cluster API server URL.
    """
    infrastructure = Infrastructure(client=admin_client, name="cluster", ensure_exists=True)
    return infrastructure.instance.status.apiServerURL


@pytest.fixture(scope="session")
def maas_original_user() -> str:
    """Username of the user who originally ran the tests (before IDP logins)."""
    current_user = run_command(command=["oc", "whoami"])[1].strip()
    LOGGER.info(f"Original user: {current_user}")
    return current_user


@pytest.fixture(scope="session")
def maas_user_credentials_both() -> dict[str, str]:
    """Randomized FREE and PREMIUM usernames/passwords plus IDP/Secret names."""
    random_suffix = generate_random_name()
    return {
        "free_user": f"maas-free-user-{random_suffix}",
        "free_pass": f"maas-free-password-{random_suffix}",
        "premium_user": f"maas-premium-user-{random_suffix}",
        "premium_pass": f"maas-premium-password-{random_suffix}",
        "idp_name": f"maas-htpasswd-idp-{random_suffix}",
        "secret_name": f"maas-htpasswd-secret-{random_suffix}",
    }


@pytest.fixture(scope="session")
def maas_created_htpasswd_secret_both(
    maas_user_credentials_both: dict[str, str],
) -> Generator[None, None, None]:
    """Create an htpasswd Secret with FREE and PREMIUM users and clean it up."""
    secret_name = maas_user_credentials_both["secret_name"]

    htpasswd_path = make_bcrypt_htpasswd_file_with_users(
        users=[
            (maas_user_credentials_both["free_user"], maas_user_credentials_both["free_pass"]),
            (maas_user_credentials_both["premium_user"], maas_user_credentials_both["premium_pass"]),
        ]
    )
    try:
        run_command(
            command=["oc", "-n", "openshift-config", "delete", "secret", secret_name, "--ignore-not-found=true"]
        )
        run_command(
            command=[
                "oc",
                "-n",
                "openshift-config",
                "create",
                "secret",
                "generic",
                secret_name,
                f"--from-file=htpasswd={htpasswd_path}",
            ],
            check=True,
        )
        yield
    finally:
        run_command(
            command=["oc", "-n", "openshift-config", "delete", "secret", secret_name, "--ignore-not-found=true"]
        )
        Path(htpasswd_path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def maas_updated_oauth_config(
    admin_client: DynamicClient,
    maas_user_credentials_both: dict[str, str],
    maas_created_htpasswd_secret_both,  # ensure secret exists first
) -> Generator[None, None, None]:
    """Patch OAuth to add a single combined MaaS htpasswd IDP, then restore it."""
    oauth = OAuth(name="cluster")

    spec = getattr(oauth.instance, "spec", {}) or {}
    original_identity_providers = spec.get("identityProviders", None)
    current_idps = list(original_identity_providers) if original_identity_providers else []

    combined_idp = {
        "name": maas_user_credentials_both["idp_name"],
        "mappingMethod": "claim",
        "type": "HTPasswd",
        "challenge": True,
        "login": True,
        "htpasswd": {"fileData": {"name": maas_user_credentials_both["secret_name"]}},
    }

    # Drop prior maas-* IDPs to avoid duplicates
    current_idps = [d for d in current_idps if not str(d.get("name", "")).startswith("maas-")]
    updated_providers = [combined_idp] + current_idps

    LOGGER.info("MaaS RBAC: updating OAuth with single combined htpasswd IDP")
    ResourceEditor(patches={oauth: {"spec": {"identityProviders": updated_providers}}}).update(backup_resources=True)
    wait_for_oauth_openshift_deployment()
    LOGGER.info("MaaS RBAC: OAuth updated with MaaS IDP")

    try:
        yield
    finally:
        LOGGER.info("MaaS RBAC: restoring OAuth identityProviders to original state")
        ResourceEditor(
            patches={
                oauth: {
                    "spec": {
                        "identityProviders": original_identity_providers
                        if original_identity_providers is not None
                        else None
                    }
                }
            }
        ).update(backup_resources=False)
        wait_for_oauth_openshift_deployment()


@pytest.fixture(scope="session")
def maas_free_user_session(
    maas_original_user: str,
    maas_api_server_url: str,
    is_byoidc: bool,
    maas_user_credentials_both: dict[str, str],
    maas_updated_oauth_config,  # ensure OAuth is patched
) -> Generator[UserTestSession, None, None]:
    """Create a FREE test IDP user session and clean it up."""
    if is_byoidc:
        pytest.skip("Working on OIDC support for tests that use htpasswd IDP for MaaS")

    free_username = maas_user_credentials_both["free_user"]
    free_password = maas_user_credentials_both["free_pass"]
    idp_name = maas_user_credentials_both["idp_name"]
    secret_name = maas_user_credentials_both["secret_name"]

    idp_session: UserTestSession | None = None
    try:
        if wait_for_user_creation(username=free_username, password=free_password, cluster_url=maas_api_server_url):
            LOGGER.info(f"Undoing login as test user and logging in as {maas_original_user}")
            login_with_user_password(api_address=maas_api_server_url, user=maas_original_user)

        idp_session = UserTestSession(
            idp_name=idp_name,
            secret_name=secret_name,
            username=free_username,
            password=free_password,
            original_user=maas_original_user,
            api_server_url=maas_api_server_url,
        )
        LOGGER.info(f"Created MaaS FREE test IDP user session: {idp_session.username}")
        yield idp_session
    finally:
        if idp_session:
            LOGGER.info(f"Cleaning up MaaS FREE test IDP user: {idp_session.username}")
            idp_session.cleanup()


@pytest.fixture(scope="session")
def maas_premium_user_session(
    maas_original_user: str,
    maas_api_server_url: str,
    is_byoidc: bool,
    maas_user_credentials_both: dict[str, str],
    maas_updated_oauth_config,  # ensure OAuth is patched
) -> Generator[UserTestSession, None, None]:
    """Create a PREMIUM test IDP user session and clean it up."""
    if is_byoidc:
        pytest.skip("Working on OIDC support for tests that use htpasswd IDP for MaaS")

    user = maas_user_credentials_both["premium_user"]
    pw = maas_user_credentials_both["premium_pass"]
    idp = maas_user_credentials_both["idp_name"]
    sec = maas_user_credentials_both["secret_name"]

    idp_session: UserTestSession | None = None
    try:
        if wait_for_user_creation(username=user, password=pw, cluster_url=maas_api_server_url):
            LOGGER.info("Undoing login as test user and logging in as %s", maas_original_user)
            login_with_user_password(api_address=maas_api_server_url, user=maas_original_user)

        idp_session = UserTestSession(
            idp_name=idp,
            secret_name=sec,
            username=user,
            password=pw,
            original_user=maas_original_user,
            api_server_url=maas_api_server_url,
        )
        LOGGER.info("Created MaaS PREMIUM test IDP user session: %s", idp_session.username)
        yield idp_session
    finally:
        if idp_session:
            LOGGER.info("Cleaning up MaaS PREMIUM test IDP user: %s", idp_session.username)
            idp_session.cleanup()


@pytest.fixture(scope="session")
def maas_free_group(
    admin_client: DynamicClient,
    maas_free_user_session: UserTestSession,
) -> Generator[str, None, None]:
    """Create a FREE-tier MaaS group and add the FREE test user to it."""
    with create_maas_group(
        admin_client=admin_client,
        group_name=MAAS_FREE_GROUP,
        users=[maas_free_user_session.username],
    ) as group:
        LOGGER.info(f"MaaS RBAC: free group {group.name} with user {maas_free_user_session.username}")
        yield group.name


@pytest.fixture(scope="session")
def maas_premium_group(
    admin_client: DynamicClient,
    maas_premium_user_session: UserTestSession,
) -> Generator[str, None, None]:
    """Create a PREMIUM-tier MaaS group and add the PREMIUM test user to it."""
    with create_maas_group(
        admin_client=admin_client,
        group_name=MAAS_PREMIUM_GROUP,
        users=[maas_premium_user_session.username],
    ) as group:
        LOGGER.info(
            "MaaS RBAC: premium group %s with user %s",
            group.name,
            maas_premium_user_session.username,
        )
        yield group.name


@pytest.fixture
def ocp_token_for_actor(
    request,
    maas_api_server_url: str,
    maas_original_user: str,
    admin_client: DynamicClient,
    maas_free_user_session: UserTestSession,
    maas_premium_user_session: UserTestSession,
) -> Generator[str, None, None]:
    """
    Log in as the requested actor ('admin' / 'free' / 'premium')
    and yield the OpenShift token for that user.
    After the test, log back in as the original user.
    """
    actor = getattr(request, "param", "admin")

    if actor == "admin":
        LOGGER.info("MaaS RBAC: using existing admin session")
        token = get_openshift_token(client=admin_client)
        yield token
        return

    if actor == "free":
        user = maas_free_user_session
    elif actor == "premium":
        user = maas_premium_user_session
    else:
        raise ValueError(f"Unknown actor kind: {actor!r}")

    LOGGER.info(f"MaaS RBAC: logging in as {user.username}")
    login_with_retry(api=maas_api_server_url, user=user.username, password=user.password)
    try:
        token = get_openshift_token()
        yield token
    finally:
        LOGGER.info(f"MaaS RBAC: logging back in as {maas_original_user}")
        login_with_retry(api=maas_api_server_url, user=maas_original_user, password=None)
