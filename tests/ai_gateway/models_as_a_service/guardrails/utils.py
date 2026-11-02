import hashlib
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import pytest
import requests
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.maas_model_ref import MaaSModelRef
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_gateway.image_constants import AiGatewayImages
from tests.ai_gateway.models_as_a_service.guardrails.constants import (
    ECHO_SIM_PORT,
    GUARDRAIL_CHECK_HEADER_PREFIX,
    GUARDRAIL_ENFORCEMENT_TIMEOUT_SECONDS,
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
from utilities.resources.maa_s_subscription import MaaSSubscription
from utilities.resources.nemo_guardrails import NemoGuardrails

LOGGER = structlog.get_logger(name=__name__)


def guardrail_check_header(namespace: str, guardrail_name: str, check_name: str) -> str:
    """Return the selection header ai-gateway-controller compiles for an AIGuardrail check.

    Args:
        namespace: Namespace of the AIGuardrail.
        guardrail_name: Name of the AIGuardrail.
        check_name: Name of the check inside spec.checks.

    Returns:
        Header name ``x-aigateway-guardrail-<sha256(namespace:guardrail:check)>``.
    """
    check_id = hashlib.sha256(f"{namespace}:{guardrail_name}:{check_name}".encode()).hexdigest()
    return f"{GUARDRAIL_CHECK_HEADER_PREFIX}{check_id}"


def nemo_email_detection_config() -> dict[str, str]:
    """Return NeMo ConfigMap data that blocks e-mail addresses on input and output.

    A regex recognizer keeps detection independent of Presidio NLP models, and
    ``passthrough`` lets the server run checks without a main LLM.

    Returns:
        ConfigMap data with ``config.yaml`` and an empty ``rails.co``.
    """
    config_yaml = {
        "passthrough": True,
        "rails": {
            "config": {
                "sensitive_data_detection": {
                    "recognizers": [
                        {
                            "name": "Email Regex",
                            "supported_entity": "EMAIL",
                            "patterns": [
                                {
                                    "name": "email_regex",
                                    "regex": r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+(\.[a-zA-Z0-9\-]+)+",
                                    "score": 1,
                                }
                            ],
                        }
                    ],
                    "input": {"entities": ["EMAIL"]},
                    "output": {"entities": ["EMAIL"]},
                }
            },
            "input": {"flows": ["detect sensitive data on input"]},
            "output": {"flows": ["detect sensitive data on output"]},
        },
    }
    return {"config.yaml": yaml.dump(config_yaml), "rails.co": ""}


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
        deployment_status.get("observedGeneration", 0) >= deployment_instance.metadata.generation
        and deployment_status.get("updatedReplicas", 0) == desired_replicas
        and deployment_status.get("availableReplicas", 0) == desired_replicas
        and deployment_status.get("replicas", 0) == desired_replicas
    )


def verify_guardrail_callouts_unauthenticated(admin_client: DynamicClient, check_headers: list[str]) -> None:
    """Assert every compiled ai_guardrails callout for the checks targets NeMo over plain HTTP.

    An https endpoint or a token file means ai-gateway-controller found kube-rbac-proxy
    in front of the NeMo Service.

    Args:
        admin_client: Cluster admin client.
        check_headers: Selection headers of the checks to inspect.
    """
    plugins_configmap = ConfigMap(
        client=admin_client,
        name=LEGACY_IPP_PLUGINS_CONFIGMAP_NAME_BASE,
        namespace=MAAS_GATEWAY_NAMESPACE,
        ensure_exists=True,
    )
    extproc_config = yaml.safe_load(plugins_configmap.instance.data[PRAXIS_EXTPROC_CONFIG_DATA_KEY])
    providers = [
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
    assert providers, f"No ai_guardrails callouts compiled for {check_headers}"
    for provider in providers:
        assert provider["endpoint"].startswith("http://"), f"NeMo callout is not plain HTTP: {provider}"
        assert "token_file" not in provider, f"NeMo callout carries a bearer token: {provider}"


def wait_for_guardrail_checks_compiled(
    admin_client: DynamicClient,
    check_headers: list[str],
    timeout: int = GUARDRAIL_ENFORCEMENT_TIMEOUT_SECONDS,
) -> None:
    """Wait until the default tenant's Praxis ExtProc enforces the given guardrail checks.

    ai-gateway-controller compiles header-gated ``ai_guardrails`` filters into the
    post-auth ``extproc.yaml`` and rolls the payload-processing Deployment.

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
) -> Generator[MaaSSubscription, Any, Any]:
    """Create a Ready MaaSSubscription for all authenticated users with optional guardrail attachments.

    Args:
        admin_client: Cluster admin client.
        namespace: Tenant (subscription) namespace; attached AIGuardrails resolve here.
        name: Subscription name.
        model_ref: MaaSModelRef included in the subscription.
        guardrails: ``spec.guardrails`` attachments, or None for an unguarded subscription.

    Yields:
        The Ready MaaSSubscription.
    """
    with MaaSSubscription(
        client=admin_client,
        name=name,
        namespace=namespace,
        owner={"groups": [{"name": "system:authenticated"}]},
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
        f"last_body={(getattr(last_response, 'text', '') or '')[:300]}"
    )
