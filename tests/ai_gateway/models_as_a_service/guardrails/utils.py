import hashlib
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import portforward
import pytest
import requests
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.event import Event
from ocp_resources.maas_model_ref import MaaSModelRef
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_gateway.image_constants import AiGatewayImages
from tests.ai_gateway.models_as_a_service.guardrails.constants import (
    ACCEPTED_CONDITION,
    ECHO_SIM_PORT,
    GUARDRAIL_CHECK_HEADER_PREFIX,
    GUARDRAIL_ENFORCEMENT_TIMEOUT_SECONDS,
    MAAS_API_SELECTION_PATH,
    NEMO_CALLER_SERVICE_CA_FILE,
    NEMO_CALLER_TOKEN_FILE,
    NEMO_CHECKS_PATH,
)
from tests.ai_gateway.models_as_a_service.praxis.constants import (
    LEGACY_IPP_PLUGINS_CONFIGMAP_NAME_BASE,
    LEGACY_IPP_POST_PROCESSING_NAME_BASE,
    PRAXIS_EXTPROC_CONFIG_DATA_KEY,
)
from tests.ai_gateway.models_as_a_service.utils import create_api_key, revoke_api_key
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.general import generate_random_name
from utilities.resources.ai_guardrail import AIGuardrail
from utilities.resources.auth_policy import AuthPolicy
from utilities.resources.maa_s_subscription import MaaSSubscription
from utilities.resources.nemo_guardrails import NemoGuardrails

LOGGER = structlog.get_logger(name=__name__)


def guardrail_check_id(namespace: str, guardrail_name: str, check_name: str) -> str:
    """Return the stable selection identifier for an AIGuardrail check.

    Args:
        namespace: Namespace of the AIGuardrail.
        guardrail_name: Name of the AIGuardrail.
        check_name: Name of the check inside spec.checks.

    Returns:
        SHA-256 identifier derived from namespace, guardrail name, and check name.
    """
    return hashlib.sha256(f"{namespace}:{guardrail_name}:{check_name}".encode()).hexdigest()


def guardrail_check_header(namespace: str, guardrail_name: str, check_name: str) -> str:
    """Return the selection header ai-gateway-controller compiles for an AIGuardrail check."""
    return f"{GUARDRAIL_CHECK_HEADER_PREFIX}{guardrail_check_id(namespace, guardrail_name, check_name)}"


def nemo_sensitive_data_detection_config(input_entities: list[str], output_entities: list[str]) -> dict[str, str]:
    """Return NeMo ConfigMap data that detects entities independently by request phase.

    Built-in Presidio recognizers avoid custom recognizer state being reused
    across guardrail evaluations. ``passthrough`` lets the server run checks
    without a main LLM.

    Returns:
        ConfigMap data with ``config.yaml`` and an empty ``rails.co``.
    """
    config_yaml = {
        "passthrough": True,
        "rails": {
            "config": {
                "sensitive_data_detection": {
                    "input": {"entities": input_entities},
                    "output": {"entities": output_entities},
                }
            },
            "input": {"flows": ["detect sensitive data on input"]},
            "output": {"flows": ["detect sensitive data on output"]},
        },
    }
    return {"config.yaml": yaml.dump(config_yaml), "rails.co": ""}


def nemo_email_detection_config() -> dict[str, str]:
    """Return NeMo ConfigMap data that blocks e-mail addresses on input and output."""
    return nemo_sensitive_data_detection_config(
        input_entities=["EMAIL_ADDRESS"],
        output_entities=["EMAIL_ADDRESS"],
    )


def nemo_phase_detection_config() -> dict[str, str]:
    """Return NeMo ConfigMap data that accepts e-mail input and blocks e-mail output."""
    return nemo_sensitive_data_detection_config(
        input_entities=["PHONE_NUMBER"],
        output_entities=["EMAIL_ADDRESS"],
    )


def echo_sim_container(served_model_name: str) -> dict[str, Any]:
    """Return the LLMInferenceService main container running llm-d-inference-sim in echo mode.

    Echo mode returns the last user message as the completion, so output guardrail
    verdicts are a deterministic function of the prompt.

    Args:
        served_model_name: Model id the simulator serves; must match the request ``model`` field.

    Returns:
        Container spec for ``spec.template.containers``.
    """
    return {
        "name": "main",
        "image": AiGatewayImages.LLMD_INFERENCE_SIM,
        "command": ["/app/llm-d-inference-sim"],
        "args": [
            "--port",
            str(ECHO_SIM_PORT),
            "--model",
            served_model_name,
            "--mode",
            "echo",
            "--no-mm-encoder-only",
            "--ssl-certfile",
            "/var/run/kserve/tls/tls.crt",
            "--ssl-keyfile",
            "/var/run/kserve/tls/tls.key",
        ],
        "ports": [{"name": "https", "containerPort": ECHO_SIM_PORT, "protocol": "TCP"}],
        "livenessProbe": {"httpGet": {"path": "/health", "port": "https", "scheme": "HTTPS"}},
        "readinessProbe": {"httpGet": {"path": "/ready", "port": "https", "scheme": "HTTPS"}},
        "resources": {
            "requests": {"cpu": "100m", "memory": "256Mi"},
            "limits": {"cpu": "500m", "memory": "512Mi"},
        },
    }


def wait_for_nemo_guardrails_ready(nemo_guardrails: NemoGuardrails, timeout: int = 600) -> None:
    """Wait until a NemoGuardrails server reports phase Ready.

    Args:
        nemo_guardrails: NemoGuardrails resource to watch.
        timeout: Maximum seconds to wait.
    """
    phase: str | None = None
    try:
        for phase in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: (nemo_guardrails.instance.status or {}).get("phase"),
        ):
            if phase == "Ready":
                return
    except TimeoutExpiredError:
        pytest.fail(f"NemoGuardrails {nemo_guardrails.namespace}/{nemo_guardrails.name} not Ready; phase={phase}")


def _ai_guardrail_bound(ai_guardrail: AIGuardrail) -> bool:
    instance = ai_guardrail.instance
    status = instance.status or {}
    accepted = next(
        (condition for condition in status.get("conditions") or [] if condition["type"] == "Accepted"),
        None,
    )
    return bool(
        accepted
        and accepted["status"] == "True"
        and accepted.get("observedGeneration") == instance.metadata.generation
        and status.get("bindingRevision")
    )


def wait_for_ai_guardrail_bound(ai_guardrail: AIGuardrail, timeout: int = 300) -> None:
    """Wait until MaaS can select an AIGuardrail's checks.

    MaaS only selects a policy whose Accepted condition is True for the current
    generation and whose status.bindingRevision is set.

    Args:
        ai_guardrail: AIGuardrail to watch.
        timeout: Maximum seconds to wait.
    """
    try:
        for bound in TimeoutSampler(wait_timeout=timeout, sleep=5, func=_ai_guardrail_bound, ai_guardrail=ai_guardrail):
            if bound:
                return
    except TimeoutExpiredError:
        pytest.fail(
            f"AIGuardrail {ai_guardrail.namespace}/{ai_guardrail.name} not bound; "
            f"status={(ai_guardrail.instance.status or {})}"
        )


def wait_for_ai_guardrail_refused(ai_guardrail: AIGuardrail, reason: str, timeout: int = 300) -> dict[str, Any]:
    """Wait until an AIGuardrail's Accepted condition is False with the given reason for its current generation.

    Args:
        ai_guardrail: AIGuardrail to watch.
        reason: Expected Accepted condition reason.
        timeout: Maximum seconds to wait.

    Returns:
        The Accepted condition.
    """
    accepted: dict[str, Any] = {}
    try:
        for accepted in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=ai_guardrail_condition,
            ai_guardrail=ai_guardrail,
            condition_type=ACCEPTED_CONDITION,
        ):
            if (
                accepted.get("status") == "False"
                and accepted.get("reason") == reason
                and accepted.get("observedGeneration") == ai_guardrail.instance.metadata.generation
            ):
                return accepted
    except TimeoutExpiredError:
        pass
    pytest.fail(
        f"AIGuardrail {ai_guardrail.namespace}/{ai_guardrail.name} not refused with reason {reason!r}; "
        f"Accepted={accepted}"
    )


def ai_guardrail_warning_events(admin_client: DynamicClient, ai_guardrail: AIGuardrail, reason: str) -> list[Any]:
    """Return the recent Warning Events recorded on an AIGuardrail with the given reason.

    Args:
        admin_client: Cluster admin client.
        ai_guardrail: AIGuardrail the Events are about.
        reason: Event reason to match.

    Returns:
        Matching Event objects.
    """
    events = Event.list(
        client=admin_client,
        namespace=ai_guardrail.namespace,
        field_selector=f"involvedObject.name={ai_guardrail.name},type=Warning",
    )
    return [event for event in events if event.reason == reason]


def _guardrail_checks_compiled(admin_client: DynamicClient, check_headers: list[str]) -> bool:
    plugins_configmap = ConfigMap(
        client=admin_client,
        name=LEGACY_IPP_PLUGINS_CONFIGMAP_NAME_BASE,
        namespace=MAAS_GATEWAY_NAMESPACE,
    )
    if not plugins_configmap.exists:
        return False
    extproc_config = (plugins_configmap.instance.data or {}).get(PRAXIS_EXTPROC_CONFIG_DATA_KEY, "")
    if "ai_guardrails" not in extproc_config or not all(header in extproc_config for header in check_headers):
        return False

    deployment = Deployment(
        client=admin_client,
        name=LEGACY_IPP_POST_PROCESSING_NAME_BASE,
        namespace=MAAS_GATEWAY_NAMESPACE,
    )
    if not deployment.exists:
        return False
    deployment_instance = deployment.instance
    deployment_status = deployment_instance.status or {}
    desired_replicas = deployment_instance.spec.replicas or 0
    return (
        desired_replicas > 0
        and deployment_status.get("observedGeneration", 0) >= deployment_instance.metadata.generation
        and deployment_status.get("updatedReplicas", 0) == desired_replicas
        and deployment_status.get("availableReplicas", 0) == desired_replicas
        and deployment_status.get("replicas", 0) == desired_replicas
    )


def compiled_guardrail_providers(admin_client: DynamicClient, check_headers: list[str]) -> list[dict[str, Any]]:
    """Return the NeMo provider blocks of the compiled ai_guardrails callouts gated on the given headers.

    Args:
        admin_client: Cluster admin client.
        check_headers: Selection headers of the checks to inspect.

    Returns:
        The ``provider`` mapping of every matching ``ai_guardrails`` filter in the default tenant's extproc.yaml.
    """
    plugins_configmap = ConfigMap(
        client=admin_client,
        name=LEGACY_IPP_PLUGINS_CONFIGMAP_NAME_BASE,
        namespace=MAAS_GATEWAY_NAMESPACE,
        ensure_exists=True,
    )
    extproc_config = yaml.safe_load(plugins_configmap.instance.data[PRAXIS_EXTPROC_CONFIG_DATA_KEY])
    return [
        guardrail_filter["provider"]
        for filter_chain in extproc_config.get("filter_chains", [])
        for guardrail_filter in filter_chain.get("filters", [])
        if guardrail_filter.get("filter") == "ai_guardrails"
        and any(
            header in condition.get("when", {}).get("headers", {})
            for condition in guardrail_filter.get("conditions", [])
            for header in check_headers
        )
    ]


def verify_guardrail_callouts_unauthenticated(admin_client: DynamicClient, check_headers: list[str]) -> None:
    """Assert every compiled ai_guardrails callout for the checks targets NeMo over plain HTTP.

    An https endpoint or a token file means ai-gateway-controller found kube-rbac-proxy
    in front of the NeMo Service.

    Args:
        admin_client: Cluster admin client.
        check_headers: Selection headers of the checks to inspect.
    """
    providers = compiled_guardrail_providers(admin_client=admin_client, check_headers=check_headers)
    assert providers, f"No ai_guardrails callouts compiled for {check_headers}"
    for provider in providers:
        assert provider["endpoint"].startswith("http://"), f"NeMo callout is not plain HTTP: {provider}"
        assert "token_file" not in provider, f"NeMo callout carries a bearer token: {provider}"


def nemo_service_url(nemo_guardrails: NemoGuardrails, workload_namespace: str, authenticated: bool) -> str:
    """Return the in-cluster NeMo Service URL ai-gateway-controller derives for a NemoGuardrails provider.

    TrustyAI names the Service after the CR when the CR lives in the workload namespace, and
    ``<cr-namespace>-<cr-name>`` otherwise. kube-rbac-proxy serves https when auth is enabled.

    Args:
        nemo_guardrails: The NemoGuardrails provider.
        workload_namespace: The provider's status.workloadNamespace.
        authenticated: Whether the provider carries the enable-auth annotation.

    Returns:
        The expected AIGuardrail status.providerEndpoint.
    """
    service_name = (
        nemo_guardrails.name
        if nemo_guardrails.namespace == workload_namespace
        else f"{nemo_guardrails.namespace}-{nemo_guardrails.name}"
    )
    scheme = "https" if authenticated else "http"
    return f"{scheme}://{service_name}.{workload_namespace}.svc.cluster.local"


def ai_guardrail_condition(ai_guardrail: AIGuardrail, condition_type: str) -> dict[str, Any]:
    """Return an AIGuardrail status condition, or an empty dict when it is not published.

    Args:
        ai_guardrail: AIGuardrail to read.
        condition_type: Condition type, e.g. ``ProviderReady``.

    Returns:
        The condition as a plain dict.
    """
    for condition in (ai_guardrail.instance.status or {}).get("conditions") or []:
        if condition["type"] == condition_type:
            return dict(condition)
    return {}


def wait_for_ai_guardrail_provider_endpoint(
    ai_guardrail: AIGuardrail,
    expected_endpoint: str,
    timeout: int = 300,
) -> None:
    """Wait until an AIGuardrail publishes the expected status.providerEndpoint.

    Args:
        ai_guardrail: AIGuardrail to watch.
        expected_endpoint: Expected in-cluster NeMo base URL.
        timeout: Maximum seconds to wait.
    """
    endpoint: str | None = None
    try:
        for endpoint in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: (ai_guardrail.instance.status or {}).get("providerEndpoint"),
        ):
            if endpoint == expected_endpoint:
                return
    except TimeoutExpiredError:
        pytest.fail(
            f"AIGuardrail {ai_guardrail.namespace}/{ai_guardrail.name} providerEndpoint={endpoint!r}, "
            f"expected {expected_endpoint!r}"
        )


def verify_guardrail_callouts_authenticated(
    admin_client: DynamicClient,
    check_headers: list[str],
    expected_endpoint: str,
) -> None:
    """Assert every compiled ai_guardrails callout for the checks calls kube-rbac-proxy with a token over TLS.

    Args:
        admin_client: Cluster admin client.
        check_headers: Selection headers of the checks to inspect.
        expected_endpoint: The AIGuardrail's https providerEndpoint.
    """
    providers = compiled_guardrail_providers(admin_client=admin_client, check_headers=check_headers)
    assert providers, f"No ai_guardrails callouts compiled for {check_headers}"
    for provider in providers:
        assert provider["endpoint"] == f"{expected_endpoint}{NEMO_CHECKS_PATH}", (
            f"NeMo callout does not target {expected_endpoint}{NEMO_CHECKS_PATH}: {provider}"
        )
        assert provider.get("token_file") == NEMO_CALLER_TOKEN_FILE, (
            f"NeMo callout does not carry the caller token: {provider}"
        )
        assert (provider.get("tls") or {}).get("ca_file") == NEMO_CALLER_SERVICE_CA_FILE, (
            f"NeMo callout does not trust the service CA: {provider}"
        )


def wait_for_guardrail_checks_compiled(
    admin_client: DynamicClient,
    check_headers: list[str],
    timeout: int = GUARDRAIL_ENFORCEMENT_TIMEOUT_SECONDS,
) -> None:
    """Wait until the default tenant's Praxis ExtProc enforces the given guardrail checks.

    ai-gateway-controller compiles header-gated ``ai_guardrails`` filters into the
    post-auth ``extproc.yaml`` and rolls a nonzero payload-processing Deployment.

    Args:
        admin_client: Cluster admin client.
        check_headers: Selection headers that must appear in the compiled config.
        timeout: Maximum seconds to wait.
    """
    try:
        for compiled in TimeoutSampler(
            wait_timeout=timeout,
            sleep=10,
            func=_guardrail_checks_compiled,
            admin_client=admin_client,
            check_headers=check_headers,
        ):
            if compiled:
                return
    except TimeoutExpiredError:
        pytest.fail(
            f"Guardrail checks {check_headers} not compiled into "
            f"{MAAS_GATEWAY_NAMESPACE}/{LEGACY_IPP_PLUGINS_CONFIGMAP_NAME_BASE} with a rolled-out "
            f"{LEGACY_IPP_POST_PROCESSING_NAME_BASE} Deployment"
        )


@contextmanager
def guarded_maas_subscription(
    admin_client: DynamicClient,
    namespace: str,
    name: str,
    model_ref: MaaSModelRef,
    guardrails: list[dict[str, Any]] | None,
    owner_groups: list[str] | None = None,
) -> Generator[MaaSSubscription, Any, Any]:
    """Create a Ready MaaSSubscription with optional owner groups and guardrail attachments.

    Args:
        admin_client: Cluster admin client.
        namespace: Tenant (subscription) namespace; attached AIGuardrails resolve here.
        name: Subscription name.
        model_ref: MaaSModelRef included in the subscription.
        guardrails: ``spec.guardrails`` attachments, or None for an unguarded subscription.
        owner_groups: Subscription owner groups, defaulting to ``system:authenticated``.

    Yields:
        The Ready MaaSSubscription.
    """
    with MaaSSubscription(
        client=admin_client,
        name=name,
        namespace=namespace,
        owner={"groups": [{"name": group} for group in (owner_groups or ["system:authenticated"])]},
        model_refs=[
            {
                "name": model_ref.name,
                "namespace": model_ref.namespace,
                "tokenRateLimits": [{"limit": 100000, "window": "1m"}],
            }
        ],
        guardrails=guardrails,
        teardown=True,
        wait_for_resource=True,
    ) as subscription:
        subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
        yield subscription


@contextmanager
def maas_api_key_for_subscription(
    request_session_http: requests.Session,
    base_url: str,
    ocp_user_token: str,
    subscription: MaaSSubscription,
) -> Generator[str, Any, Any]:
    """Mint a MaaS API key bound to one subscription and revoke it on exit.

    Binding at mint time pins subscription (and therefore guardrail) selection for every request.

    Args:
        request_session_http: HTTP session for the MaaS gateway.
        base_url: MaaS API base URL.
        ocp_user_token: OpenShift token of the key owner.
        subscription: Subscription the key is bound to.

    Yields:
        The API key value.
    """
    _, body = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_user_token,
        request_session_http=request_session_http,
        api_key_name=f"e2e-guardrails-{generate_random_name()}",
        subscription=subscription.name,
    )
    try:
        yield str(body["key"])
    finally:
        revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=body["id"],
            ocp_user_token=ocp_user_token,
        )


def chat_completion_payload(model: str, prompt: str, stream: bool = False) -> dict[str, Any]:
    """Return a single-turn chat completions request body.

    Args:
        model: Model id served behind MaaS.
        prompt: User message.
        stream: Whether to request a server-sent-events response.

    Returns:
        Request body for POST /v1/chat/completions.
    """
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "stream": stream,
    }


def wait_for_chat_response(
    request_session_http: requests.Session,
    model_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    is_expected: Callable[[requests.Response], bool],
    description: str,
    timeout: int = GUARDRAIL_ENFORCEMENT_TIMEOUT_SECONDS,
) -> requests.Response:
    """Send chat completions until the response matches, absorbing gateway propagation delays.

    Args:
        request_session_http: HTTP session for the MaaS gateway.
        model_url: Chat completions URL of the model.
        headers: Request headers including the MaaS API key.
        payload: Request body.
        is_expected: Predicate the response must satisfy.
        description: Human-readable expectation used in logs and failure messages.
        timeout: Maximum seconds to wait.

    Returns:
        The first response that satisfies ``is_expected``.
    """
    last_response: requests.Response | None = None
    try:
        for response in TimeoutSampler(
            wait_timeout=timeout,
            sleep=10,
            func=request_session_http.post,
            url=model_url,
            headers=headers,
            json=payload,
            timeout=60,
        ):
            last_response = response
            LOGGER.info(f"MaaS guardrails: waiting for {description}; got status={response.status_code}")
            if is_expected(response):
                return response
    except TimeoutExpiredError:
        pass
    pytest.fail(
        f"Timed out waiting for {description}. "
        f"last_status={getattr(last_response, 'status_code', None)}, "
        f"last_content_type={getattr(last_response, 'headers', {}).get('content-type') if last_response else None}, "
        f"last_body_length={len(getattr(last_response, 'content', b'') or b'')}"
    )


def _wait_for_guardrail_selection(
    request_session_http: requests.Session,
    maas_api_infra_namespace: str,
    subscription: MaaSSubscription,
    model_ref: MaaSModelRef,
    is_expected: Callable[[dict[str, Any]], bool],
    description: str,
    groups: list[str] | None = None,
    username: str = "guardrails-e2e",
    requested_subscription: str | None = None,
    requested_model: str | None = None,
) -> dict[str, Any]:
    """Wait for a MaaS selection response matching the expected subscription metadata.

    Args:
        request_session_http: HTTP session used for the local MaaS API request.
        maas_api_infra_namespace: Namespace containing the MaaS API service.
        subscription: Subscription selected explicitly by the request.
        model_ref: Model requested by the selection request.
        is_expected: Predicate applied to a successful selection response.
        description: Expected selection result used in the timeout message.
        groups: Identity groups sent to the selection endpoint.
        username: Identity username sent to the selection endpoint.
        requested_subscription: Explicit subscription name, or None for group-based auto-selection.
        requested_model: Model identifier sent to the selection endpoint.

    Returns:
        The selection response matching ``is_expected``.
    """
    payload: dict[str, Any] = {
        "groups": groups or ["system:authenticated"],
        "username": username,
        "requestedModel": requested_model or f"{model_ref.namespace}/{model_ref.name}",
    }
    if requested_subscription is not None:
        payload["requestedSubscription"] = requested_subscription
    last_response: requests.Response | None = None

    with portforward.forward(
        pod_or_service="maas-api",
        namespace=maas_api_infra_namespace,
        from_port=0,
        to_port=8443,
        waiting=20,
    ) as forwarder:
        selection_url = f"https://localhost:{forwarder.from_port}{MAAS_API_SELECTION_PATH}"
        try:
            for response in TimeoutSampler(
                wait_timeout=GUARDRAIL_ENFORCEMENT_TIMEOUT_SECONDS,
                sleep=10,
                func=request_session_http.post,
                url=selection_url,
                json=payload,
                timeout=30,
            ):
                last_response = response
                if response.status_code != 200:
                    continue
                response_body = response.json()
                if is_expected(response_body):
                    return response_body
        except TimeoutExpiredError:
            pass

    pytest.fail(
        f"Timed out waiting for {description}. "
        f"last_status={getattr(last_response, 'status_code', None)}, "
        f"last_content_type={getattr(last_response, 'headers', {}).get('content-type') if last_response else None}, "
        f"last_body_length={len(getattr(last_response, 'content', b'') or b'')}"
    )


def wait_for_guardrail_unavailable_selection(
    request_session_http: requests.Session,
    maas_api_infra_namespace: str,
    subscription: MaaSSubscription,
    model_ref: MaaSModelRef,
) -> dict[str, Any]:
    """Wait for MaaS selection to fail closed for a subscription with an unavailable guardrail."""
    return _wait_for_guardrail_selection(
        request_session_http=request_session_http,
        maas_api_infra_namespace=maas_api_infra_namespace,
        subscription=subscription,
        model_ref=model_ref,
        is_expected=lambda selection: selection.get("error") == "guardrail_unavailable",
        description="MaaS selection to fail closed for an unavailable guardrail",
        requested_subscription=subscription.name,
    )


def wait_for_selected_guardrail_checks(
    request_session_http: requests.Session,
    maas_api_infra_namespace: str,
    subscription: MaaSSubscription,
    model_ref: MaaSModelRef,
    expected_check_ids: set[str],
    requested_model: str | None = None,
) -> dict[str, Any]:
    """Wait for MaaS selection to return exactly the expected effective guardrail checks."""
    return _wait_for_guardrail_selection(
        request_session_http=request_session_http,
        maas_api_infra_namespace=maas_api_infra_namespace,
        subscription=subscription,
        model_ref=model_ref,
        is_expected=lambda selection: (
            selection.get("name") == subscription.name
            and selection.get("selectedGuardrailChecks") == dict.fromkeys(expected_check_ids, True)
        ),
        description="MaaS selection to return the expected guardrail checks",
        requested_subscription=subscription.name,
        requested_model=requested_model,
    )


def wait_for_selected_guardrail_checks_for_groups(
    request_session_http: requests.Session,
    maas_api_infra_namespace: str,
    subscription: MaaSSubscription,
    model_ref: MaaSModelRef,
    groups: list[str],
    expected_check_ids: set[str],
    username: str = "guardrails-e2e",
    requested_model: str | None = None,
) -> dict[str, Any]:
    """Wait for group-based MaaS selection without explicitly naming a subscription."""
    return _wait_for_guardrail_selection(
        request_session_http=request_session_http,
        maas_api_infra_namespace=maas_api_infra_namespace,
        subscription=subscription,
        model_ref=model_ref,
        is_expected=lambda selection: (
            selection.get("selectedGuardrailChecks") == dict.fromkeys(expected_check_ids, True)
        ),
        description="MaaS selection to return guardrails for the requested identity groups",
        groups=groups,
        username=username,
        requested_subscription=None,
        requested_model=requested_model,
    )


def _gateway_guardrail_header_catalog(
    admin_client: DynamicClient,
    policy_name: str,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> set[str]:
    auth_policy = AuthPolicy(
        client=admin_client,
        name=policy_name,
        namespace=gateway_namespace,
    )
    if not auth_policy.exists:
        return set()

    spec = auth_policy.instance.spec or {}
    defaults = spec.get("defaults") or {}
    rules = defaults.get("rules") or {}
    response = rules.get("response") or {}
    success = response.get("success") or {}
    headers = success.get("headers") or {}
    return {
        header_name.lower()
        for header_name in dict(headers)
        if header_name.lower().startswith(GUARDRAIL_CHECK_HEADER_PREFIX)
    }


def wait_for_guardrail_header_catalog(
    admin_client: DynamicClient,
    policy_name: str,
    expected_headers: set[str],
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
    timeout: int = GUARDRAIL_ENFORCEMENT_TIMEOUT_SECONDS,
) -> set[str]:
    """Wait until a gateway AuthPolicy exposes exactly the expected guardrail headers.

    Args:
        admin_client: Cluster admin client.
        policy_name: Gateway AuthPolicy to inspect.
        expected_headers: Guardrail header names that must be present.
        gateway_namespace: Namespace containing the gateway AuthPolicy.
        timeout: Maximum seconds to wait.

    Returns:
        The current guardrail header catalog.
    """
    try:
        for headers in TimeoutSampler(
            wait_timeout=timeout,
            sleep=10,
            func=_gateway_guardrail_header_catalog,
            admin_client=admin_client,
            policy_name=policy_name,
            gateway_namespace=gateway_namespace,
        ):
            if headers == {header.lower() for header in expected_headers}:
                return headers
    except TimeoutExpiredError:
        pass

    current_headers = _gateway_guardrail_header_catalog(
        admin_client=admin_client,
        policy_name=policy_name,
        gateway_namespace=gateway_namespace,
    )
    pytest.fail(
        f"Timed out waiting for AuthPolicy {gateway_namespace}/{policy_name} to expose guardrail headers "
        f"{sorted(expected_headers)}; current headers={sorted(current_headers)}"
    )
