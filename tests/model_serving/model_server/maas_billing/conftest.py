from typing import Generator, Dict, List
import base64
import pytest
import requests
from simple_logger.logger import get_logger
from utilities.plugins.constant import OpenAIEnpoints
from ocp_resources.service_account import ServiceAccount
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.deployment import Deployment
from timeout_sampler import TimeoutSampler
from utilities.llmd_utils import create_llmisvc
from utilities.llmd_constants import ModelStorage, ContainerImages
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from pytest_testconfig import config as py_config
from utilities.constants import (
    MAAS_GATEWAY_NAMESPACE,
    MAAS_RATE_LIMIT_POLICY_NAME,
    MAAS_TOKEN_RATE_LIMIT_POLICY_NAME,
)

from ocp_resources.infrastructure import Infrastructure
from ocp_resources.oauth import OAuth
from ocp_resources.resource import ResourceEditor
from utilities.general import generate_random_name
from utilities.user_utils import UserTestSession, wait_for_user_creation, create_htpasswd_file
from utilities.infra import login_with_user_password, get_openshift_token
from utilities.general import wait_for_oauth_openshift_deployment
from ocp_resources.secret import Secret
from tests.model_serving.model_server.maas_billing.utils import get_total_tokens
from utilities.constants import DscComponents, MAAS_GATEWAY_NAME
from utilities.resources.rate_limit_policy import RateLimitPolicy
from utilities.resources.token_rate_limit_policy import TokenRateLimitPolicy
from tests.model_serving.model_server.maas_billing.utils import (
    detect_scheme_via_llmisvc,
    host_from_ingress_domain,
    mint_token,
    patch_llmisvc_with_maas_router,
    create_maas_group,
    build_maas_headers,
    get_maas_models_response,
    verify_chat_completions,
    maas_gateway_rate_limits_patched,
    endpoints_have_ready_addresses,
    gateway_probe_reaches_maas_api,
    maas_gateway_listeners,
    revoke_token,
)

LOGGER = get_logger(name=__name__)
MODELS_INFO = OpenAIEnpoints.MODELS_INFO
CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS

MAAS_FREE_GROUP = "tier-free-users"
MAAS_PREMIUM_GROUP = "tier-premium-users"


@pytest.fixture(scope="session")
def request_session_http() -> Generator[requests.Session, None, None]:
    session = requests.Session()
    session.headers.update({"User-Agent": "odh-maas-billing-tests/1"})
    session.verify = False
    yield session
    session.close()


@pytest.fixture(scope="class")
def minted_token(
    request_session_http,
    base_url: str,
    current_client_token: str,
    maas_controller_enabled_latest: None,
    maas_gateway_api: None,
    maas_request_ratelimit_policy: None,
    maas_token_ratelimit_policy: None,
    maas_api_gateway_reachable: None,
) -> str:
    """Mint a MaaS token once per test class and reuse it."""
    resp, body = mint_token(
        base_url=base_url,
        oc_user_token=current_client_token,
        minutes=30,
        http_session=request_session_http,
    )
    LOGGER.info(f"Mint token response status={resp.status_code}")
    assert resp.status_code in (200, 201), f"mint failed: {resp.status_code} {resp.text[:200]}"
    token = body.get("token", "")
    assert isinstance(token, str) and len(token) > 10, f"no usable token in response: {body}"
    LOGGER.info(f"Minted MaaS token len={len(token)}")
    return token


@pytest.fixture(scope="class")
def base_url(maas_scheme: str, maas_host: str) -> str:
    return f"{maas_scheme}://{maas_host}/maas-api"


@pytest.fixture(scope="class")
def model_url(
    maas_scheme: str,
    maas_host: str,
    admin_client: DynamicClient,
    maas_inference_service_tinyllama: LLMInferenceService,
) -> str:
    deployment = maas_inference_service_tinyllama.name
    url = f"{maas_scheme}://{maas_host}/llm/{deployment}{CHAT_COMPLETIONS}"
    LOGGER.info(f"MaaS: constructed model_url={url} (deployment={deployment})")
    return url


@pytest.fixture
def maas_headers(minted_token: str) -> dict:
    return build_maas_headers(token=minted_token)


@pytest.fixture
def maas_models(
    request_session_http: requests.Session,
    base_url: str,
    maas_headers: dict,
    maas_inference_service_tinyllama: LLMInferenceService,
    maas_gateway_api: None,
    maas_request_ratelimit_policy: None,
    maas_token_ratelimit_policy: None,
):
    resp = get_maas_models_response(
        session=request_session_http,
        base_url=base_url,
        headers=maas_headers,
    )

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
def maas_htpasswd_files(
    maas_user_credentials_both: dict[str, str],
) -> Generator[tuple[str, str, str, str], None, None]:
    """
    Create per-user htpasswd files for FREE and PREMIUM users and return
    their file paths + base64 contents.

    Cleanup of the temp files happens at teardown.
    """
    free_username = maas_user_credentials_both["free_user"]
    free_password = maas_user_credentials_both["free_pass"]
    premium_username = maas_user_credentials_both["premium_user"]
    premium_password = maas_user_credentials_both["premium_pass"]

    free_htpasswd_file_path, free_htpasswd_b64 = create_htpasswd_file(
        username=free_username,
        password=free_password,
    )
    premium_htpasswd_file_path, premium_htpasswd_b64 = create_htpasswd_file(
        username=premium_username,
        password=premium_password,
    )

    try:
        yield (
            free_htpasswd_file_path,
            free_htpasswd_b64,
            premium_htpasswd_file_path,
            premium_htpasswd_b64,
        )
    finally:
        free_htpasswd_file_path.unlink(missing_ok=True)
        premium_htpasswd_file_path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def maas_htpasswd_oauth_idp(
    admin_client: DynamicClient,
    maas_user_credentials_both: dict[str, str],
    maas_htpasswd_files: tuple[str, str, str, str],
    is_byoidc: bool,
):
    """
    - Combines FREE + PREMIUM htpasswd entries into a single Secret.
    - Adds the MaaS HTPasswd IDP to oauth/cluster using ResourceEditor.
    - Waits for oauth-openshift rollout after patch.
    - On teardown, waits again and verifies the IDP is gone.
    """
    if is_byoidc:
        pytest.skip("Working on OIDC support for tests that use htpasswd IDP for MaaS")
    else:
        (
            _free_htpasswd_file_path,
            free_htpasswd_b64,
            _premium_htpasswd_file_path,
            premium_htpasswd_b64,
        ) = maas_htpasswd_files

        free_username = maas_user_credentials_both["free_user"]
        premium_username = maas_user_credentials_both["premium_user"]
        secret_name = maas_user_credentials_both["secret_name"]
        idp_name = maas_user_credentials_both["idp_name"]

        free_bytes = base64.b64decode(s=free_htpasswd_b64)
        premium_bytes = base64.b64decode(s=premium_htpasswd_b64)
        combined_bytes = free_bytes + b"\n" + premium_bytes
        combined_htpasswd_b64 = base64.b64encode(s=combined_bytes).decode("utf-8")

        oauth_resource = OAuth(name="cluster", client=admin_client)
        oauth_spec = oauth_resource.instance.spec or {}
        existing_idps = oauth_spec.get("identityProviders", [])

        maas_idp = {
            "name": idp_name,
            "mappingMethod": "claim",
            "type": "HTPasswd",
            "challenge": True,
            "login": True,
            "htpasswd": {"fileData": {"name": secret_name}},
        }

        updated_idps = existing_idps + [maas_idp]

        LOGGER.info(
            f"MaaS RBAC: creating shared htpasswd Secret '{secret_name}' "
            f"for users '{free_username}' and '{premium_username}'"
        )

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
            ResourceEditor(patches={oauth_resource: {"spec": {"identityProviders": updated_idps}}}),
        ):
            LOGGER.info(f"MaaS RBAC: updating OAuth with MaaS htpasswd IDP '{maas_idp['name']}'")
            wait_for_oauth_openshift_deployment()
            LOGGER.info(f"MaaS RBAC: OAuth updated with MaaS IDP '{maas_idp['name']}'")
            yield
        wait_for_oauth_openshift_deployment()
        LOGGER.info(f"MaaS RBAC: OAuth identityProviders cleanup completed for IDP '{idp_name}'")


@pytest.fixture(scope="session")
def maas_rbac_idp_env(
    maas_htpasswd_oauth_idp,
    maas_user_credentials_both: dict[str, str],
):

    return maas_user_credentials_both


@pytest.fixture(scope="session")
def maas_free_user_session(
    original_user: str,
    maas_api_server_url: str,
    is_byoidc: bool,
    maas_rbac_idp_env: dict[str, str],
) -> Generator[UserTestSession, None, None]:
    if is_byoidc:
        pytest.skip("Working on OIDC support for tests that use htpasswd IDP for MaaS")
    else:
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
            login_with_user_password(
                api_address=maas_api_server_url,
                user=original_user,
            )

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
    maas_rbac_idp_env: dict[str, str],
) -> Generator[UserTestSession, None, None]:
    if is_byoidc:
        pytest.skip("Working on OIDC support for tests that use htpasswd IDP for MaaS")
    else:
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
            login_with_user_password(
                api_address=maas_api_server_url,
                user=original_user,
            )

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


@pytest.fixture(scope="class")
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
    actor_param = getattr(request, "param", {"type": "admin"})
    actor_type = actor_param.get("type", "admin")

    if actor_type == "admin":
        LOGGER.info("MaaS RBAC: using existing admin session to obtain token")
        yield get_openshift_token(client=admin_client)
    else:
        if actor_type == "free":
            user_session = maas_free_user_session
        elif actor_type == "premium":
            user_session = maas_premium_user_session
        else:
            raise ValueError(f"Unknown actor type: {actor_type!r}")

        LOGGER.info(f"MaaS RBAC: logging in as MaaS {actor_type} user '{user_session.username}'")
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


@pytest.fixture(scope="class")
def maas_token_for_actor(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    maas_controller_enabled_latest: None,
    maas_gateway_api: None,
    maas_api_gateway_reachable: None,
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
    LOGGER.info(f"MaaS RBAC: mint token status={response.status_code}")
    assert response.status_code in (200, 201), f"mint failed: {response.status_code} {response.text[:200]}"

    token = body.get("token", "")
    assert isinstance(token, str) and len(token) > 10, "no usable MaaS token in response"

    LOGGER.info(f"MaaS RBAC: minted MaaS token len={len(token)} for current actor")
    return token


@pytest.fixture(scope="class")
def maas_headers_for_actor(maas_token_for_actor: str) -> dict:
    """Headers for the current actor (admin/free/premium)."""
    return build_maas_headers(token=maas_token_for_actor)


@pytest.fixture(scope="class")
def maas_models_response_for_actor(
    request_session_http: requests.Session,
    base_url: str,
    maas_headers_for_actor: dict,
) -> requests.Response:
    """Validated /v1/models response for the current actor."""
    return get_maas_models_response(
        session=request_session_http,
        base_url=base_url,
        headers=maas_headers_for_actor,
    )


@pytest.fixture(scope="class")
def maas_models_for_actor(
    maas_models_response_for_actor: requests.Response,
) -> List[Dict]:

    models_list = maas_models_response_for_actor.json().get("data", [])
    assert models_list, "no models returned from /v1/models"
    return models_list


@pytest.fixture(scope="class")
def exercise_rate_limiter(
    actor_label: str,
    scenario: dict,
    request_session_http: requests.Session,
    model_url: str,
    maas_headers_for_actor: Dict[str, str],
    maas_models_for_actor: List[Dict],
) -> List[int]:

    models_list = maas_models_for_actor

    max_requests = scenario["max_requests"]
    max_tokens = scenario["max_tokens"]
    log_prefix = scenario["log_prefix"]

    status_codes_list: List[int] = []

    for attempt_index in range(max_requests):
        LOGGER.info(f"{log_prefix}[{actor_label}]: attempt {attempt_index + 1}/{max_requests}")

        response = verify_chat_completions(
            request_session_http=request_session_http,
            model_url=model_url,
            headers=maas_headers_for_actor,
            models_list=models_list,
            prompt_text="Repeat the word 'token' 60 times, separated by spaces. No extra text.",
            max_tokens=max_tokens,
            request_timeout_seconds=60,
            log_prefix=f"{log_prefix}[{actor_label}]",
            expected_status_codes=(200, 429),
        )

        status_codes_list.append(response.status_code)

        total_tokens = get_total_tokens(resp=response)

        if scenario["id"] == "token-rate" and response.status_code == 200:
            total_tokens = get_total_tokens(resp=response, fail_if_missing=True)
            LOGGER.info(f"{log_prefix}[{actor_label}]: total_tokens={total_tokens}")
    LOGGER.info(f"{log_prefix}[{actor_label}]: status_codes={status_codes_list}")
    return status_codes_list


@pytest.fixture(scope="class")
def maas_inference_service_tinyllama(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    model_service_account: ServiceAccount,
    maas_gateway_api: None,
    maas_request_ratelimit_policy: None,
    maas_token_ratelimit_policy: None,
) -> Generator[LLMInferenceService, None, None]:
    """
    TinyLlama S3-backed LLMInferenceService wired through MaaS for tests.
    """
    with create_llmisvc(
        client=admin_client,
        name="llm-s3-tinyllama",
        namespace=unprivileged_model_namespace.name,
        storage_uri=ModelStorage.TINYLLAMA_S3,
        container_image=ContainerImages.VLLM_CPU,
        container_resources={
            "limits": {"cpu": "2", "memory": "12Gi"},
            "requests": {"cpu": "1", "memory": "8Gi"},
        },
        service_account=model_service_account.name,
        wait=False,
        timeout=900,
    ) as llm_service:
        with patch_llmisvc_with_maas_router(
            llm_service=llm_service,
        ):
            inst = llm_service.instance
            storage_uri = inst.spec.model.uri
            assert storage_uri == ModelStorage.TINYLLAMA_S3, f"Unexpected storage_uri on TinyLlama LLMI: {storage_uri}"

            llm_service.wait_for_condition(
                condition="Ready",
                status="True",
                timeout=900,
            )

            LOGGER.info(
                f"MaaS: TinyLlama LLMI {llm_service.namespace}/{llm_service.name} "
                f"Ready and patched (storage_uri={storage_uri})"
            )

            yield llm_service


@pytest.fixture(scope="class")
def maas_scheme(admin_client: DynamicClient, unprivileged_model_namespace: Namespace) -> str:
    return detect_scheme_via_llmisvc(
        client=admin_client,
        namespace=unprivileged_model_namespace.name,
    )


@pytest.fixture(scope="class")
def maas_host(admin_client):
    return host_from_ingress_domain(client=admin_client)


@pytest.fixture(scope="class")
def maas_gateway_rate_limits(
    admin_client: DynamicClient,
    maas_gateway_api: None,
    maas_request_ratelimit_policy: None,
    maas_token_ratelimit_policy: None,
    maas_tier_mapping_cm,
) -> Generator[None, None, None]:
    with maas_gateway_rate_limits_patched(
        admin_client=admin_client,
        namespace=MAAS_GATEWAY_NAMESPACE,
        token_policy_name=MAAS_TOKEN_RATE_LIMIT_POLICY_NAME,
        request_policy_name=MAAS_RATE_LIMIT_POLICY_NAME,
    ):
        yield


@pytest.fixture(scope="session")
def maas_gateway_api_hostname(admin_client: DynamicClient) -> str:
    return host_from_ingress_domain(client=admin_client)


@pytest.fixture(scope="session")
def maas_controller_enabled_latest(
    dsc_resource: DataScienceCluster,
    maas_gateway_api: None,
    maas_request_ratelimit_policy: None,
    maas_token_ratelimit_policy: None,
) -> Generator[DataScienceCluster, None, None]:
    """
    Ensure MaaS (KServe modelsAsService) is MANAGED for the session.
    Restore DSC to original state on teardown.
    """

    component_patch = {
        DscComponents.KSERVE: {"modelsAsService": {"managementState": DscComponents.ManagementState.MANAGED}}
    }

    with ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}}):
        dsc_resource.wait_for_condition(
            condition="ModelsAsServiceReady",
            status="True",
            timeout=900,
        )
        dsc_resource.wait_for_condition(
            condition="Ready",
            status="True",
            timeout=600,
        )
        yield dsc_resource

    dsc_resource.wait_for_condition(condition="Ready", status="True", timeout=600)


@pytest.fixture(scope="session")
def maas_tier_mapping_cm(
    admin_client: DynamicClient,
) -> ConfigMap:

    LOGGER.info(f"DEBUG py_config keys={list(py_config.keys())}")
    LOGGER.info(f"DEBUG applications_namespace={py_config.get('applications_namespace')}")
    config_map = ConfigMap(
        client=admin_client,
        name="tier-to-group-mapping",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )

    LOGGER.info(
        f"MaaS tier mapping ConfigMap detected: namespace={py_config['applications_namespace']}, name={config_map.name}"
    )

    return config_map


@pytest.fixture(scope="class")
def maas_api_deployment_available(
    admin_client: DynamicClient,
) -> None:
    maas_api_deployment = Deployment(
        client=admin_client,
        name="maas-api",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )
    maas_api_deployment.wait_for_condition(
        condition="Available",
        status="True",
        timeout=600,
    )


@pytest.fixture(scope="class")
def maas_api_endpoints_ready(
    admin_client: DynamicClient,
    maas_api_deployment_available: None,
) -> None:
    for ready in TimeoutSampler(
        wait_timeout=300,
        sleep=5,
        func=endpoints_have_ready_addresses,
        admin_client=admin_client,
        namespace=py_config["applications_namespace"],
        name="maas-api",
    ):
        if ready:
            return


@pytest.fixture(scope="class")
def maas_api_gateway_reachable(
    request_session_http: requests.Session,
    base_url: str,
    maas_api_endpoints_ready: None,
) -> None:
    probe_url = f"{base_url}/v1/models"

    for gateway_reachable, _status_code, _response_text in TimeoutSampler(
        wait_timeout=300,
        sleep=5,
        func=gateway_probe_reaches_maas_api,
        http_session=request_session_http,
        probe_url=probe_url,
        request_timeout_seconds=30,
    ):
        if gateway_reachable:
            return


@pytest.fixture(scope="session")
def maas_gateway_api(
    admin_client: DynamicClient,
    maas_gateway_api_hostname: str,
) -> Generator[None, None, None]:
    """
    Ensure MaaS Gateway exists once per test session.
    """
    with Gateway(
        client=admin_client,
        name=MAAS_GATEWAY_NAME,
        namespace=MAAS_GATEWAY_NAMESPACE,
        gateway_class_name="openshift-default",
        listeners=maas_gateway_listeners(hostname=maas_gateway_api_hostname),
        annotations={"opendatahub.io/managed": "false"},
        label={
            "app.kubernetes.io/name": "maas",
            "app.kubernetes.io/instance": MAAS_GATEWAY_NAME,
            "app.kubernetes.io/component": "gateway",
            "opendatahub.io/managed": "false",
        },
        ensure_exists=False,
        wait_for_resource=True,
        teardown=True,
    ):
        yield


@pytest.fixture(scope="session")
def maas_gateway_target_ref() -> dict:
    return {
        "group": "gateway.networking.k8s.io",
        "kind": "Gateway",
        "name": MAAS_GATEWAY_NAME,
    }


@pytest.fixture(scope="session")
def maas_request_ratelimit_policy(
    admin_client: DynamicClient,
    maas_gateway_api: None,
    maas_gateway_target_ref: dict,
) -> Generator[None, None, None]:
    with RateLimitPolicy(
        client=admin_client,
        name=MAAS_RATE_LIMIT_POLICY_NAME,
        namespace=MAAS_GATEWAY_NAMESPACE,
        target_ref=maas_gateway_target_ref,
        limits={
            "bootstrap": {
                "counters": [{"expression": "auth.identity.userid"}],
                "rates": [{"limit": 1000, "window": "1m"}],
            }
        },
        ensure_exists=False,
        wait_for_resource=True,
        teardown=True,
    ):
        yield


@pytest.fixture(scope="session")
def maas_token_ratelimit_policy(
    admin_client: DynamicClient,
    maas_gateway_api: None,
    maas_gateway_target_ref: dict,
) -> Generator[None, None, None]:
    with TokenRateLimitPolicy(
        client=admin_client,
        name=MAAS_TOKEN_RATE_LIMIT_POLICY_NAME,
        namespace=MAAS_GATEWAY_NAMESPACE,
        target_ref=maas_gateway_target_ref,
        limits={
            "bootstrap": {
                "counters": [{"expression": "auth.identity.userid"}],
                "rates": [{"limit": 1000000, "window": "1m"}],
            }
        },
        ensure_exists=False,
        wait_for_resource=True,
        teardown=True,
    ):
        yield


@pytest.fixture
def ensure_working_maas_token_pre_revoke(
    request_session_http,
    model_url,
    maas_headers_for_actor,
    maas_models_response_for_actor,
    actor_label,
) -> List[Dict]:
    models_list = maas_models_response_for_actor.json().get("data", [])

    verify_chat_completions(
        request_session_http=request_session_http,
        model_url=model_url,
        headers=maas_headers_for_actor,
        models_list=models_list,
        prompt_text="hi",
        max_tokens=16,
        request_timeout_seconds=60,
        log_prefix=f"MaaS revoke pre-check [{actor_label}]",
        expected_status_codes=(200,),
    )

    return models_list


@pytest.fixture
def revoke_maas_tokens_for_actor(
    request_session_http,
    base_url: str,
    ocp_token_for_actor: str,
    actor_label: str,
) -> None:
    revoke_url = f"{base_url}/v1/tokens"
    LOGGER.info(f"[{actor_label}] revoke request: DELETE {revoke_url}")

    r_del = revoke_token(
        base_url=base_url,
        oc_user_token=ocp_token_for_actor,
        http_session=request_session_http,
    )

    LOGGER.info(f"[{actor_label}] revoke response: status={r_del.status_code} body={(r_del.text or '')[:200]}")
