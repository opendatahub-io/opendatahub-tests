from typing import Generator

import pytest
import requests
from simple_logger.logger import get_logger
from utilities.plugins.constant import RestHeader, OpenAIEnpoints

from kubernetes.dynamic import DynamicClient
from ocp_resources.infrastructure import Infrastructure
from ocp_resources.oauth import OAuth
from ocp_resources.resource import ResourceEditor
from utilities.general import generate_random_name
from utilities.user_utils import UserTestSession, wait_for_user_creation, create_htpasswd_file
from utilities.infra import login_with_user_password, get_openshift_token
from utilities.general import wait_for_oauth_openshift_deployment
from ocp_resources.secret import Secret


from tests.model_serving.model_server.maas_billing.utils import (
    detect_scheme_via_llmisvc,
    host_from_ingress_domain,
    mint_token,
    llmis_name,
    create_maas_group,
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
def maas_user_credentials_both() -> dict[str, str]:
    """
    Randomized FREE and PREMIUM usernames/passwords plus per-tier IDP & Secret names.

    """
    random_suffix = generate_random_name()

    return {
        # FREE user
        "free_user": f"maas-free-user-{random_suffix}",
        "free_pass": f"maas-free-password-{random_suffix}",
        "free_idp_name": f"maas-free-htpasswd-idp-{random_suffix}",
        "free_secret_name": f"maas-free-htpasswd-secret-{random_suffix}",
        # PREMIUM user
        "premium_user": f"maas-premium-user-{random_suffix}",
        "premium_pass": f"maas-premium-password-{random_suffix}",
        "premium_idp_name": f"maas-premium-htpasswd-idp-{random_suffix}",
        "premium_secret_name": f"maas-premium-htpasswd-secret-{random_suffix}",
    }


@pytest.fixture(scope="session")
def maas_created_htpasswd_secret_both(
    admin_client: DynamicClient,
    maas_user_credentials_both: dict[str, str],
) -> Generator[None, None, None]:
    """
    Create two htpasswd Secrets (FREE + PREMIUM)

    """
    free_username = maas_user_credentials_both["free_user"]
    free_password = maas_user_credentials_both["free_pass"]
    free_secret_name = maas_user_credentials_both["free_secret_name"]

    premium_username = maas_user_credentials_both["premium_user"]
    premium_password = maas_user_credentials_both["premium_pass"]
    premium_secret_name = maas_user_credentials_both["premium_secret_name"]

    free_tmp_path, free_htpasswd_b64 = create_htpasswd_file(
        username=free_username,
        password=free_password,
    )
    premium_tmp_path, premium_htpasswd_b64 = create_htpasswd_file(
        username=premium_username,
        password=premium_password,
    )

    try:
        # Create FREE secret
        free_secret = Secret(
            client=admin_client,
            name=free_secret_name,
            namespace="openshift-config",
            htpasswd=free_htpasswd_b64,
            type="Opaque",
            teardown=False,
            wait_for_resource=True,
        )
        free_secret.deploy()

        # Create PREMIUM secret
        premium_secret = Secret(
            client=admin_client,
            name=premium_secret_name,
            namespace="openshift-config",
            htpasswd=premium_htpasswd_b64,
            type="Opaque",
            teardown=False,
            wait_for_resource=True,
        )
        premium_secret.deploy()

        yield

    finally:
        free_tmp_path.unlink(missing_ok=True)
        premium_tmp_path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def maas_updated_oauth_config(
    admin_client: DynamicClient,
    maas_user_credentials_both: dict[str, str],
    maas_created_htpasswd_secret_both,
) -> Generator[None, None, None]:
    """
    Patch OAuth to add two MaaS htpasswd IDPs (FREE + PREMIUM) backed by
    the per-user Secrets created above, then restore the original config.

    """
    oauth = OAuth(name="cluster")

    spec = getattr(oauth.instance, "spec", {}) or {}
    existing_idps = spec.get("identityProviders") or []

    free_idp = {
        "name": maas_user_credentials_both["free_idp_name"],
        "mappingMethod": "claim",
        "type": "HTPasswd",
        "challenge": True,
        "login": True,
        "htpasswd": {"fileData": {"name": maas_user_credentials_both["free_secret_name"]}},
    }

    premium_idp = {
        "name": maas_user_credentials_both["premium_idp_name"],
        "mappingMethod": "claim",
        "type": "HTPasswd",
        "challenge": True,
        "login": True,
        "htpasswd": {"fileData": {"name": maas_user_credentials_both["premium_secret_name"]}},
    }

    non_maas_idps = [idp for idp in existing_idps if not str(idp.get("name", "")).startswith("maas-")]
    updated_providers = [free_idp, premium_idp] + non_maas_idps

    LOGGER.info("MaaS RBAC: updating OAuth with FREE + PREMIUM htpasswd IDPs")
    idp_editor = ResourceEditor(
        patches={oauth: {"spec": {"identityProviders": updated_providers}}},
    )

    try:
        idp_editor.update(backup_resources=True)
        wait_for_oauth_openshift_deployment()
        LOGGER.info("MaaS RBAC: OAuth updated with MaaS IDPs")
        yield

    finally:
        LOGGER.info("MaaS RBAC: restoring OAuth identityProviders to original state")
        idp_editor.restore()
        wait_for_oauth_openshift_deployment()


@pytest.fixture(scope="session")
def maas_free_user_session(
    original_user: str,
    maas_api_server_url: str,
    is_byoidc: bool,
    maas_user_credentials_both: dict[str, str],
    maas_updated_oauth_config,
) -> Generator[UserTestSession, None, None]:
    """Create a FREE test IDP user session and clean it up."""
    if is_byoidc:
        pytest.skip("Working on OIDC support for tests that use htpasswd IDP for MaaS")

    username = maas_user_credentials_both["free_user"]
    password = maas_user_credentials_both["free_pass"]
    idp_name = maas_user_credentials_both["free_idp_name"]
    secret_name = maas_user_credentials_both["free_secret_name"]

    idp_session: UserTestSession | None = None
    try:
        wait_for_user_creation(
            username=username,
            password=password,
            cluster_url=maas_api_server_url,
        )

        LOGGER.info("Undoing login as test user and logging in as %s", original_user)
        login_with_user_password(api_address=maas_api_server_url, user=original_user)

        idp_session = UserTestSession(
            idp_name=idp_name,
            secret_name=secret_name,
            username=username,
            password=password,
            original_user=original_user,
            api_server_url=maas_api_server_url,
        )
        LOGGER.info("Created MaaS FREE test IDP user session: %s", idp_session.username)
        yield idp_session
    finally:
        if idp_session:
            LOGGER.info("Cleaning up MaaS FREE test IDP user: %s", idp_session.username)
            idp_session.cleanup()


@pytest.fixture(scope="session")
def maas_premium_user_session(
    original_user: str,
    maas_api_server_url: str,
    is_byoidc: bool,
    maas_user_credentials_both: dict[str, str],
    maas_updated_oauth_config,  # ensure OAuth is patched
) -> Generator[UserTestSession, None, None]:
    """Create a PREMIUM test IDP user session and clean it up."""
    if is_byoidc:
        pytest.skip("Working on OIDC support for tests that use htpasswd IDP for MaaS")

    username = maas_user_credentials_both["premium_user"]
    password = maas_user_credentials_both["premium_pass"]
    idp_name = maas_user_credentials_both["premium_idp_name"]
    secret_name = maas_user_credentials_both["premium_secret_name"]

    idp_session: UserTestSession | None = None
    try:
        wait_for_user_creation(
            username=username,
            password=password,
            cluster_url=maas_api_server_url,
        )

        LOGGER.info("Undoing login as test user and logging in as %s", original_user)
        login_with_user_password(api_address=maas_api_server_url, user=original_user)

        idp_session = UserTestSession(
            idp_name=idp_name,
            secret_name=secret_name,
            username=username,
            password=password,
            original_user=original_user,
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
    original_user: str,
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

    LOGGER.info("MaaS RBAC: logging in as %s", user.username)
    assert login_with_user_password(
        api_address=maas_api_server_url,
        user=user.username,
        password=user.password,
    ), f"Failed to log in as {user.username}"

    try:
        token = get_openshift_token()
        yield token
    finally:
        LOGGER.info("MaaS RBAC: logging back in as %s", original_user)
        assert login_with_user_password(
            api_address=maas_api_server_url,
            user=original_user,
        ), f"Failed to log back in as {original_user}"
