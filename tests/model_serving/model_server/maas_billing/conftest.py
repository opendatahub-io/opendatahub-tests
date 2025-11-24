from typing import Generator

import base64
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
from timeout_sampler import TimeoutExpiredError


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
    Randomized FREE and PREMIUM usernames/passwords plus a shared
    htpasswd Secret and IDP name for MaaS RBAC tests.
    """
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
def maas_rbac_idp_env(
    admin_client: DynamicClient,
    maas_user_credentials_both: dict[str, str],
    is_byoidc: bool,
) -> Generator[dict[str, str], None, None]:
    """
    - Creates a single htpasswd Secret with FREE + PREMIUM users.
    - Adds a temporary MaaS HTPasswd IDP to oauth/cluster using ResourceEditor.
    - Waits for oauth-openshift rollout after patch.
    - On teardown, ResourceEditor restores the original OAuth spec and we wait again.
    - Deletes the temporary htpasswd Secret.
    """
    if is_byoidc:
        pytest.skip("Working on OIDC support for tests that use htpasswd IDP for MaaS")

    free_username = maas_user_credentials_both["free_user"]
    free_password = maas_user_credentials_both["free_pass"]
    premium_username = maas_user_credentials_both["premium_user"]
    premium_password = maas_user_credentials_both["premium_pass"]
    secret_name = maas_user_credentials_both["secret_name"]
    idp_name = maas_user_credentials_both["idp_name"]

    free_htpasswd_file_path, free_htpasswd_b64 = create_htpasswd_file(
        username=free_username,
        password=free_password,
    )

    premium_htpasswd_file_path, premium_htpasswd_b64 = create_htpasswd_file(
        username=premium_username,
        password=premium_password,
    )

    try:
        free_bytes = base64.b64decode(s=free_htpasswd_b64)
        premium_bytes = base64.b64decode(s=premium_htpasswd_b64)
        combined_bytes = free_bytes + b"\n" + premium_bytes
        combined_htpasswd_b64 = base64.b64encode(s=combined_bytes).decode("utf-8")

        oauth_resource = OAuth(name="cluster", client=admin_client)
        oauth_spec = getattr(oauth_resource.instance, "spec", {}) or {}
        existing_identity_providers = oauth_spec.get("identityProviders") or []

        maas_identity_provider = {
            "name": idp_name,
            "mappingMethod": "claim",
            "type": "HTPasswd",
            "challenge": True,
            "login": True,
            "htpasswd": {"fileData": {"name": secret_name}},
        }

        updated_identity_providers = existing_identity_providers + [maas_identity_provider]

        LOGGER.info(
            f"MaaS RBAC: creating shared htpasswd Secret '{secret_name}' for users "
            f"'{free_username}' and '{premium_username}'"
        )

        # --- update OAuth + create Secret ---
        with (
            Secret(
                client=admin_client,
                name=secret_name,
                namespace="openshift-config",
                htpasswd=combined_htpasswd_b64,
                type="Opaque",
                teardown=True,
                wait_for_resource=True,
            ),
            ResourceEditor(patches={oauth_resource: {"spec": {"identityProviders": updated_identity_providers}}}),
        ):
            LOGGER.info(f"MaaS RBAC: updating OAuth with MaaS htpasswd IDP '{maas_identity_provider['name']}'")

            wait_for_oauth_openshift_deployment()
            LOGGER.info(f"MaaS RBAC: OAuth updated with MaaS IDP '{maas_identity_provider['name']}'")

            # >>> this is the yield that pytest uses <<<
            yield maas_user_credentials_both

            LOGGER.info("MaaS RBAC: restoring OAuth identityProviders to original state")

        # --- after exit: secret deleted + OAuth restored ---
        try:
            wait_for_oauth_openshift_deployment()
            LOGGER.info("MaaS RBAC: oauth-openshift rollout completed after restoring OAuth.")
        except TimeoutExpiredError as timeout_error:
            LOGGER.warning(
                f"MaaS RBAC: timeout while waiting for oauth-openshift rollout after restoring OAuth. "
                f"Continuing teardown. Details: {timeout_error}"
            )

        # --- final sanity check ---
        oauth_resource_after = OAuth(name="cluster", client=admin_client)
        oauth_spec_after = getattr(oauth_resource_after.instance, "spec", {}) or {}
        current_idps = oauth_spec_after.get("identityProviders") or []

        if any(idp.get("name") == idp_name for idp in current_idps):
            LOGGER.warning(
                f"MaaS RBAC: temporary MaaS IDP '{idp_name}' is STILL present in OAuth spec "
                f"after teardown — please investigate cluster OAuth configuration."
            )
        else:
            LOGGER.info("MaaS RBAC: OAuth identityProviders restoration & cleanup completed")

    finally:
        free_htpasswd_file_path.unlink(missing_ok=True)
        premium_htpasswd_file_path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def maas_free_user_session(
    original_user: str,
    maas_api_server_url: str,
    is_byoidc: bool,
    maas_rbac_idp_env: dict[str, str],
) -> Generator[UserTestSession, None, None]:
    if is_byoidc:
        pytest.skip("Working on OIDC support for tests that use htpasswd IDP for MaaS")

    username = maas_rbac_idp_env["free_user"]
    password = maas_rbac_idp_env["free_pass"]
    idp_name = maas_rbac_idp_env["idp_name"]
    secret_name = maas_rbac_idp_env["secret_name"]

    idp_session: UserTestSession | None = None
    try:
        wait_for_user_creation(
            username=username,
            password=password,
            cluster_url=maas_api_server_url,
        )

        LOGGER.info(f"MaaS RBAC: undoing login as test user and logging in as '{original_user}'")
        login_with_user_password(api_address=maas_api_server_url, user=original_user)

        idp_session = UserTestSession(
            idp_name=idp_name,
            secret_name=secret_name,
            username=username,
            password=password,
            original_user=original_user,
            api_server_url=maas_api_server_url,
        )
        LOGGER.info(f"MaaS RBAC: created FREE test IDP user session '{idp_session.username}'")
        yield idp_session
    finally:
        if idp_session:
            LOGGER.info(f"MaaS RBAC: cleaning up FREE test IDP user '{idp_session.username}'")
            idp_session.cleanup()


@pytest.fixture(scope="session")
def maas_premium_user_session(
    original_user: str,
    maas_api_server_url: str,
    is_byoidc: bool,
    maas_rbac_idp_env: dict[str, str],  # <-- same outer fixture
) -> Generator[UserTestSession, None, None]:
    if is_byoidc:
        pytest.skip("Working on OIDC support for tests that use htpasswd IDP for MaaS")

    username = maas_rbac_idp_env["premium_user"]
    password = maas_rbac_idp_env["premium_pass"]
    idp_name = maas_rbac_idp_env["idp_name"]
    secret_name = maas_rbac_idp_env["secret_name"]

    idp_session: UserTestSession | None = None
    try:
        wait_for_user_creation(
            username=username,
            password=password,
            cluster_url=maas_api_server_url,
        )

        LOGGER.info(f"MaaS RBAC: undoing login as test user and logging in as '{original_user}'")
        login_with_user_password(api_address=maas_api_server_url, user=original_user)

        idp_session = UserTestSession(
            idp_name=idp_name,
            secret_name=secret_name,
            username=username,
            password=password,
            original_user=original_user,
            api_server_url=maas_api_server_url,
        )
        LOGGER.info(f"MaaS RBAC: created PREMIUM test IDP user session '{idp_session.username}'")
        yield idp_session
    finally:
        if idp_session:
            LOGGER.info(f"MaaS RBAC: cleaning up PREMIUM test IDP user '{idp_session.username}'")
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
        LOGGER.info(f"MaaS RBAC: free group '{group.name}' with user '{maas_free_user_session.username}'")
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
        LOGGER.info(f"MaaS RBAC: premium group '{group.name}' with user '{maas_premium_user_session.username}'")
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

    """
    actor_param = getattr(request, "param", None)

    if isinstance(actor_param, dict):
        actor_kind = actor_param.get("kind", "admin")
    else:
        actor_kind = actor_param or "admin"

    if actor_kind == "admin":
        LOGGER.info("MaaS RBAC: using existing admin session to obtain token")
        yield get_openshift_token(client=admin_client)
    else:
        if actor_kind == "free":
            user_session = maas_free_user_session
        elif actor_kind == "premium":
            user_session = maas_premium_user_session
        else:
            raise ValueError(f"Unknown actor kind: {actor_kind!r}")

        LOGGER.info(f"MaaS RBAC: logging in as MaaS {actor_kind} user '{user_session.username}'")
        login_successful = login_with_user_password(
            api_address=maas_api_server_url,
            user=user_session.username,
            password=user_session.password,
        )
        assert login_successful, f"Failed to log in as {user_session.username}"

        try:
            LOGGER.info(f"MaaS RBAC: obtaining token for user '{user_session.username}'")
            yield get_openshift_token()
        finally:
            LOGGER.info(f"MaaS RBAC: logging back in as original user '{original_user}'")
            original_login_successful = login_with_user_password(
                api_address=maas_api_server_url,
                user=original_user,
            )
            assert original_login_successful, f"Failed to log back in as original user '{original_user}'"


@pytest.fixture
def maas_token_for_actor(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
) -> str:
    """
    Mint a MaaS token once per actor (admin / free / premium) and reuse it
    across all tests in the class instance.

    This is reusable by any MaaS tests that need a MaaS token for a given actor.
    """
    response, body = mint_token(
        base_url=base_url,
        oc_user_token=ocp_token_for_actor,
        http_session=request_session_http,
        minutes=30,
    )
    LOGGER.info("MaaS RBAC: mint token status=%s", response.status_code)
    assert response.status_code in (200, 201), f"mint failed: {response.status_code} {response.text[:200]}"

    token = body.get("token", "")
    assert isinstance(token, str) and len(token) > 10, "no usable MaaS token in response"

    LOGGER.info("MaaS RBAC: minted MaaS token len=%s for current actor", len(token))
    return token
