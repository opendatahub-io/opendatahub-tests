from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret

from tests.ai_gateway.models_as_a_service.external_model.utils import (
    EXTERNAL_ENDPOINT,
    INFERENCE_EXTERNAL_MODEL_API_GROUP,
    external_provider_ref,
    get_httproute,
    get_service,
    wait_for_httproute,
)
from tests.ai_gateway.models_as_a_service.utils import (
    build_maas_headers,
    create_api_key,
    revoke_api_key,
)
from utilities.general import generate_random_name
from utilities.infra import get_openshift_token
from utilities.resources.external_model import ExternalModel
from utilities.resources.external_provider import ExternalProvider

LOGGER = structlog.get_logger(name=__name__)

# ai-gateway-controller#129 ("credential_injection - Qualify Azure api-key auth")
# qualifies these provider types for the apikey credential strategy. Both inject
# the credential into the provider-boundary "api-key" request header (distinct
# from Anthropic's "x-api-key" and OpenAI's filter-side "Authorization: Bearer").
AZURE_PROVIDER_TYPES = ["azure", "azure-openai"]
AZURE_AUTH_HEADER = "api-key"

# The dummy credential value stored under the Secret's ``api-key`` data key. Asserted
# verbatim against what the upstream reflects back as the forwarded header.
AZURE_CREDENTIAL_VALUE = "e2e-azure-test-key"

# httpbin.org/anything reflects the received request (headers + body) back as JSON.
# Routing the qualified Azure upstream here lets the test observe the injected
# credential on the real forwarded request, instead of inferring it from CR state.
ECHO_PATH = "/anything"


def _base_name(provider_type: str) -> str:
    return f"e2e-{provider_type}-cred"


# ─── Fixtures: qualification chain ──────────────────────────────────────────────


@pytest.fixture(scope="class", params=AZURE_PROVIDER_TYPES)
def azure_provider_type(request: pytest.FixtureRequest) -> str:
    """The Azure provider type under test (``azure`` and ``azure-openai``)."""
    return request.param


@pytest.fixture(scope="class")
def azure_credential_secret(
    admin_client: DynamicClient,
    maas_unprivileged_model_namespace: Namespace,
    azure_provider_type: str,
) -> Generator[Secret, Any, Any]:
    """Opaque Secret holding a dummy API key. The CRD requires the data key ``api-key``."""
    with Secret(
        client=admin_client,
        name=f"{_base_name(azure_provider_type)}-api-key",
        namespace=maas_unprivileged_model_namespace.name,
        type="Opaque",
        string_data={"api-key": AZURE_CREDENTIAL_VALUE},
        teardown=True,
        wait_for_resource=True,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def azure_external_provider(
    admin_client: DynamicClient,
    maas_unprivileged_model_namespace: Namespace,
    azure_provider_type: str,
    azure_credential_secret: Secret,
) -> Generator[ExternalProvider, Any, Any]:
    """ExternalProvider with ``auth.type: apikey`` — the strategy ai-gateway-controller#129 qualifies for Azure."""
    with ExternalProvider(
        client=admin_client,
        name=f"{_base_name(azure_provider_type)}-provider",
        namespace=maas_unprivileged_model_namespace.name,
        provider=azure_provider_type,
        endpoint=EXTERNAL_ENDPOINT,
        auth={
            "type": "apikey",
            "secretRef": {"name": azure_credential_secret.name},
        },
        teardown=True,
        wait_for_resource=True,
    ) as provider:
        provider.wait_for_condition(condition="Ready", status="True", timeout=300)
        yield provider


@pytest.fixture(scope="class")
def azure_external_model(
    admin_client: DynamicClient,
    maas_unprivileged_model_namespace: Namespace,
    azure_provider_type: str,
    azure_external_provider: ExternalProvider,
) -> Generator[ExternalModel, Any, Any]:
    """ExternalModel with ``apiFormat: openai-chat`` — the only format ai-gateway-controller#129 qualifies for Azure."""
    with ExternalModel(
        client=admin_client,
        name=f"{_base_name(azure_provider_type)}-model",
        namespace=maas_unprivileged_model_namespace.name,
        external_provider_refs=[external_provider_ref(provider_name=azure_external_provider.name)],
        teardown=True,
        wait_for_resource=True,
    ) as external_model:
        # Ready + an HTTPRoute together prove the provider was qualified: before
        # ai-gateway-controller#129 the resolver skipped Azure (APIFormatUnsupported), so neither happened.
        external_model.wait_for_condition(condition="Ready", status="True", timeout=300)
        wait_for_httproute(
            client=admin_client,
            name=external_model.name,
            namespace=maas_unprivileged_model_namespace.name,
            timeout=120,
        )
        yield external_model


@pytest.fixture(scope="class")
def azure_model_ref(
    admin_client: DynamicClient,
    maas_unprivileged_model_namespace: Namespace,
    azure_external_model: ExternalModel,
) -> Generator[MaaSModelRef, Any, Any]:
    """MaaSModelRef linking to the Azure ExternalModel CR."""
    with MaaSModelRef(
        client=admin_client,
        name=azure_external_model.name,
        namespace=maas_unprivileged_model_namespace.name,
        model_ref={
            "name": azure_external_model.name,
            "namespace": maas_unprivileged_model_namespace.name,
            "kind": "ExternalModel",
            "apiGroup": INFERENCE_EXTERNAL_MODEL_API_GROUP,
        },
        teardown=True,
        wait_for_resource=True,
    ) as model_ref:
        yield model_ref


# ─── Tests: Qualification ───────────────────────────────────────────────────────


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "azure_external_provider",
    "azure_external_model",
    "azure_model_ref",
)
class TestAzureCredentialInjectionQualification:
    """ai-gateway-controller#129 qualifies azure / azure-openai so the resolver builds a real route."""

    @pytest.mark.tier1
    def test_external_provider_ready(
        self,
        azure_external_provider: ExternalProvider,
    ) -> None:
        """Given an apikey Azure ExternalProvider, when reconciled, then it reports Ready."""
        assert azure_external_provider.exists, f"ExternalProvider '{azure_external_provider.name}' was not created"
        LOGGER.info(f"ExternalProvider '{azure_external_provider.name}' (apikey) is Ready")

    @pytest.mark.tier1
    def test_external_model_reconciled_to_httproute(
        self,
        admin_client: DynamicClient,
        maas_unprivileged_model_namespace: Namespace,
        azure_external_model: ExternalModel,
    ) -> None:
        """Given a qualified Azure provider, when the model reconciles, then an HTTPRoute exists."""
        route = get_httproute(
            client=admin_client,
            name=azure_external_model.name,
            namespace=maas_unprivileged_model_namespace.name,
        )
        assert route is not None, (
            f"HTTPRoute '{azure_external_model.name}' not found — the Azure provider was not "
            "qualified for a route (regression of ai-gateway-controller#129)"
        )
        LOGGER.info(f"HTTPRoute '{azure_external_model.name}' created for qualified Azure provider")

    @pytest.mark.tier1
    def test_reconciler_created_backend_service(
        self,
        admin_client: DynamicClient,
        maas_unprivileged_model_namespace: Namespace,
        azure_external_provider: ExternalProvider,
    ) -> None:
        """Given a reconciled Azure ExternalProvider, when listing Services, then one exists."""
        svc = get_service(
            client=admin_client,
            name=azure_external_provider.name,
            namespace=maas_unprivileged_model_namespace.name,
        )
        assert svc is not None, (
            f"Service '{azure_external_provider.name}' not found in '{maas_unprivileged_model_namespace.name}'"
        )
        LOGGER.info(f"Service '{azure_external_provider.name}' created by reconciler")


# ─── Fixtures: injected-header echo chain ───────────────────────────────────────
#
# These fixtures mirror the azure_* chain but route the upstream at an echo path
# (httpbin.org/anything) so the forwarded request is reflected back. The actor is
# the cluster admin minted directly via get_openshift_token — this deliberately
# avoids ``ocp_token_for_actor`` (and therefore the htpasswd oauth/cluster IDP it
# eagerly provisions), so the class runs on hosted-control-plane clusters where
# oauth/cluster is immutable.


@pytest.fixture(scope="class")
def azure_echo_external_model(
    admin_client: DynamicClient,
    maas_unprivileged_model_namespace: Namespace,
    azure_provider_type: str,
    azure_external_provider: ExternalProvider,
) -> Generator[ExternalModel, Any, Any]:
    """ExternalModel whose provider ref points at the echo path so the upstream reflects the request."""
    echo_ref = external_provider_ref(provider_name=azure_external_provider.name)
    echo_ref["path"] = ECHO_PATH
    with ExternalModel(
        client=admin_client,
        name=f"{_base_name(azure_provider_type)}-echo",
        namespace=maas_unprivileged_model_namespace.name,
        external_provider_refs=[echo_ref],
        teardown=True,
        wait_for_resource=True,
    ) as external_model:
        external_model.wait_for_condition(condition="Ready", status="True", timeout=300)
        wait_for_httproute(
            client=admin_client,
            name=external_model.name,
            namespace=maas_unprivileged_model_namespace.name,
            timeout=120,
        )
        yield external_model


@pytest.fixture(scope="class")
def azure_echo_model_ref(
    admin_client: DynamicClient,
    maas_unprivileged_model_namespace: Namespace,
    azure_echo_external_model: ExternalModel,
) -> Generator[MaaSModelRef, Any, Any]:
    with MaaSModelRef(
        client=admin_client,
        name=azure_echo_external_model.name,
        namespace=maas_unprivileged_model_namespace.name,
        model_ref={
            "name": azure_echo_external_model.name,
            "namespace": maas_unprivileged_model_namespace.name,
            "kind": "ExternalModel",
            "apiGroup": INFERENCE_EXTERNAL_MODEL_API_GROUP,
        },
        teardown=True,
        wait_for_resource=True,
    ) as model_ref:
        yield model_ref


@pytest.fixture(scope="class")
def azure_echo_auth_policy(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
    azure_echo_model_ref: MaaSModelRef,
) -> Generator[MaaSAuthPolicy, Any, Any]:
    with MaaSAuthPolicy(
        client=admin_client,
        name=f"{azure_echo_model_ref.name}-access",
        namespace=maas_subscription_namespace.name,
        model_refs=[{"name": azure_echo_model_ref.name, "namespace": azure_echo_model_ref.namespace}],
        subjects={"groups": [{"name": "system:authenticated"}]},
        teardown=True,
        wait_for_resource=True,
    ) as auth_policy:
        auth_policy.wait_for_condition(condition="Ready", status="True", timeout=300)
        yield auth_policy


@pytest.fixture(scope="class")
def azure_echo_subscription(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
    azure_echo_model_ref: MaaSModelRef,
    azure_echo_auth_policy: MaaSAuthPolicy,
) -> Generator[MaaSSubscription, Any, Any]:
    with MaaSSubscription(
        client=admin_client,
        name=f"{azure_echo_model_ref.name}-subscription",
        namespace=maas_subscription_namespace.name,
        owner={"groups": [{"name": "system:authenticated"}]},
        model_refs=[
            {
                "name": azure_echo_model_ref.name,
                "namespace": azure_echo_model_ref.namespace,
                "tokenRateLimits": [{"limit": 10000, "window": "1h"}],
            }
        ],
        priority=0,
        teardown=True,
        wait_for_resource=True,
    ) as subscription:
        subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
        azure_echo_model_ref.wait_for_condition(condition="Ready", status="True", timeout=300)
        yield subscription


@pytest.fixture(scope="class")
def azure_echo_api_key(
    request_session_http: requests.Session,
    base_url: str,
    admin_client: DynamicClient,
    azure_provider_type: str,
    azure_echo_subscription: MaaSSubscription,
) -> Generator[str, Any, Any]:
    """API key bound to the echo subscription, minted as cluster admin (no htpasswd IDP)."""
    admin_token = get_openshift_token(client=admin_client)
    key_name = f"e2e-{azure_provider_type}-echo-{generate_random_name()}"
    _, body = create_api_key(
        base_url=base_url,
        ocp_user_token=admin_token,
        request_session_http=request_session_http,
        api_key_name=key_name,
        subscription=azure_echo_subscription.name,
        expires_in="1h",
    )
    LOGGER.info(f"Created Azure echo API key: id={body['id']}")
    yield body["key"]
    revoke_response, _ = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=body["id"],
        ocp_user_token=admin_token,
    )
    assert revoke_response.status_code in (200, 404), (
        f"Unexpected revocation status for key id={body['id']}: {revoke_response.status_code}"
    )


@pytest.fixture(scope="class")
def azure_echo_inference_url(
    maas_scheme: str,
    maas_host: str,
    maas_unprivileged_model_namespace: Namespace,
    azure_echo_external_model: ExternalModel,
) -> str:
    """Chat completions URL for the echo-routed Azure model on the MaaS gateway."""
    return (
        f"{maas_scheme}://{maas_host}"
        f"/{maas_unprivileged_model_namespace.name}"
        f"/{azure_echo_external_model.name}/v1/chat/completions"
    )


# ─── Tests: Injected header observed on the forwarded request ───────────────────


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "authorino_tls_configured",
    "maas_api_endpoints_ready",
    "azure_external_provider",
    "azure_echo_external_model",
    "azure_echo_model_ref",
    "azure_echo_auth_policy",
    "azure_echo_subscription",
)
class TestAzureCredentialInjectionHeaderEcho:
    """End-to-end proof that the qualified Azure credential lands in the outbound ``api-key`` header.

    The qualification class proves ai-gateway-controller#129 qualifies Azure (route + Service built). This
    routes the upstream at an echo endpoint that reflects the received request, so the
    test observes the injected header — name AND value — on the real forwarded request.
    It is the only check that validates the injection itself rather than its preconditions.
    """

    @pytest.mark.tier2
    @pytest.mark.skip_on_disconnected
    def test_credential_injected_into_outbound_api_key_header(
        self,
        request_session_http: requests.Session,
        azure_echo_inference_url: str,
        azure_echo_external_model: ExternalModel,
        azure_echo_api_key: str,
    ) -> None:
        """Given a valid key, when a chat request is forwarded, then the upstream received
        ``api-key: <credential>`` — proving ai-gateway-controller#129 injected the credential
        into the egress request."""
        response = request_session_http.post(
            url=azure_echo_inference_url,
            headers=build_maas_headers(token=azure_echo_api_key),
            json={
                "model": azure_echo_external_model.name,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
            timeout=60,
        )
        LOGGER.info(f"Echo upstream response: HTTP {response.status_code}: {(response.text or '')[:800]}")
        assert response.status_code == 200, (
            f"Expected 200 reflected by the echo upstream, got {response.status_code}: {(response.text or '')[:300]}"
        )

        try:
            echoed = response.json()
        except ValueError:
            pytest.fail(
                "Echo upstream response was not JSON — the ExtProc may have transformed it: "
                f"{(response.text or '')[:300]}"
            )

        echoed_headers = echoed.get("headers") or {}
        # httpbin title-cases header names and may return a value as str or list; normalize both.
        header_lookup = {str(k).lower(): v for k, v in echoed_headers.items()}
        injected = header_lookup.get(AZURE_AUTH_HEADER.lower())
        if isinstance(injected, list):
            injected = injected[0] if injected else None

        assert injected is not None, (
            f"Forwarded request carried no '{AZURE_AUTH_HEADER}' header — the credential was not injected. "
            f"Echoed headers: {sorted(echoed_headers)}"
        )
        assert injected == AZURE_CREDENTIAL_VALUE, (
            f"Forwarded '{AZURE_AUTH_HEADER}' header = {injected!r}, expected {AZURE_CREDENTIAL_VALUE!r}"
        )
        LOGGER.info(f"Credential injection confirmed: forwarded request carried {AZURE_AUTH_HEADER}={injected!r}")
