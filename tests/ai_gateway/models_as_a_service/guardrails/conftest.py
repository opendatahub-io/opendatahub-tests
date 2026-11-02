from collections.abc import Generator
from typing import Any

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.namespace import Namespace

from tests.ai_gateway.models_as_a_service.guardrails.constants import (
    AI_GUARDRAIL_NAME,
    AI_GUARDRAIL_TIMEOUT,
    INPUT_CHECK_NAME,
    NEMO_CONFIGMAP_NAME,
    NEMO_PII_CONFIG_ID,
    NEMO_SERVER_NAME,
    OUTPUT_CHECK_NAME,
)
from tests.ai_gateway.models_as_a_service.guardrails.utils import (
    echo_sim_container,
    guarded_maas_subscription,
    guardrail_check_header,
    maas_api_key_for_subscription,
    nemo_email_detection_config,
    verify_guardrail_callouts_unauthenticated,
    wait_for_ai_guardrail_bound,
    wait_for_guardrail_checks_compiled,
    wait_for_nemo_guardrails_ready,
)
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE
from utilities.general import generate_random_name
from utilities.plugins.constant import OpenAIEnpoints
from utilities.resources.ai_guardrail import AIGuardrail
from utilities.resources.llm_inference_service import LLMInferenceService
from utilities.resources.maa_s_subscription import MaaSSubscription
from utilities.resources.nemo_guardrails import NemoGuardrails


@pytest.fixture(scope="class")
def guardrails_echo_llmisvc(
    admin_client: DynamicClient,
    maas_unprivileged_model_namespace: Namespace,
    maas_gateway_api: None,
    maas_request_ratelimit_policy: None,
) -> Generator[LLMInferenceService, Any, Any]:
    """llm-d-inference-sim LLMInferenceService in echo mode, routed through the MaaS gateway."""
    name = generate_random_name(prefix="guardrails-echo-sim")
    with LLMInferenceService(
        client=admin_client,
        name=name,
        namespace=maas_unprivileged_model_namespace.name,
        annotations={
            "alpha.maas.opendatahub.io/tiers": "[]",
            "security.opendatahub.io/enable-auth": "true",
        },
        # The simulator needs no weights; uri only satisfies the LLMInferenceService schema.
        model={"uri": "hf://placeholder/no-model", "name": name},
        storage_initializer={"enabled": False},
        replicas=1,
        router={
            "gateway": {"refs": [{"name": MAAS_GATEWAY_NAME, "namespace": MAAS_GATEWAY_NAMESPACE}]},
            "route": {},
        },
        template={"containers": [echo_sim_container(served_model_name=name)]},
        teardown=True,
    ) as llm_service:
        llm_service.wait_for_condition(condition="Ready", status="True", timeout=600)
        yield llm_service


@pytest.fixture(scope="class")
def guardrails_model_ref(
    admin_client: DynamicClient,
    guardrails_echo_llmisvc: LLMInferenceService,
) -> Generator[MaaSModelRef, Any, Any]:
    """MaaSModelRef exposing the echo simulator through MaaS."""
    with MaaSModelRef(
        client=admin_client,
        name=guardrails_echo_llmisvc.name,
        namespace=guardrails_echo_llmisvc.namespace,
        model_ref={
            "name": guardrails_echo_llmisvc.name,
            "namespace": guardrails_echo_llmisvc.namespace,
            "kind": "LLMInferenceService",
        },
        teardown=True,
        wait_for_resource=True,
    ) as model_ref:
        yield model_ref


@pytest.fixture(scope="class")
def guardrails_maas_auth_policy(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
    guardrails_model_ref: MaaSModelRef,
) -> Generator[MaaSAuthPolicy, Any, Any]:
    """MaaSAuthPolicy granting all authenticated users access to the echo simulator."""
    with MaaSAuthPolicy(
        client=admin_client,
        name="guardrails-echo-sim-access",
        namespace=maas_subscription_namespace.name,
        model_refs=[{"name": guardrails_model_ref.name, "namespace": guardrails_model_ref.namespace}],
        subjects={"groups": [{"name": "system:authenticated"}]},
        teardown=True,
        wait_for_resource=True,
    ) as auth_policy:
        yield auth_policy


@pytest.fixture(scope="class")
def guardrails_nemo_configmap(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
) -> Generator[ConfigMap, Any, Any]:
    """NeMo config that blocks e-mail addresses on input and output; the operator copies it to the workload."""
    with ConfigMap(
        client=admin_client,
        name=NEMO_CONFIGMAP_NAME,
        namespace=maas_subscription_namespace.name,
        data=nemo_email_detection_config(),
        teardown=True,
    ) as configmap:
        yield configmap


@pytest.fixture(scope="class")
def guardrails_nemo_server(
    admin_client: DynamicClient,
    maas_api_infra_namespace: str,
    maas_subscription_namespace: Namespace,
    guardrails_nemo_configmap: ConfigMap,
) -> Generator[NemoGuardrails, Any, Any]:
    """Ready NemoGuardrails CR in the tenant namespace whose server workload runs in the AI Gateway infra namespace."""
    with NemoGuardrails(
        client=admin_client,
        name=NEMO_SERVER_NAME,
        namespace=maas_subscription_namespace.name,
        # Praxis calls NeMo directly over HTTP; kube-rbac-proxy must not front the Service.
        annotations={"security.opendatahub.io/enable-auth": "false"},
        nemo_configs=[
            {
                "name": NEMO_PII_CONFIG_ID,
                "configMaps": [guardrails_nemo_configmap.name],
                "default": True,
            }
        ],
        replicas=1,
        expose_route=False,
        teardown=True,
    ) as nemo_server:
        wait_for_nemo_guardrails_ready(nemo_guardrails=nemo_server)
        workload_namespace = nemo_server.instance.status.workloadNamespace
        assert workload_namespace == maas_api_infra_namespace, (
            f"NeMo workload deployed to '{workload_namespace}', expected infra namespace '{maas_api_infra_namespace}'"
        )
        yield nemo_server


@pytest.fixture(scope="class")
def pii_ai_guardrail(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
    guardrails_nemo_server: NemoGuardrails,
) -> Generator[AIGuardrail, Any, Any]:
    """Bound AIGuardrail offering one input-phase and one output-phase PII check."""
    with AIGuardrail(
        client=admin_client,
        name=AI_GUARDRAIL_NAME,
        namespace=maas_subscription_namespace.name,
        provider={
            "nemo": {"ref": {"name": guardrails_nemo_server.name}},
            "timeout": AI_GUARDRAIL_TIMEOUT,
        },
        checks=[
            {"name": INPUT_CHECK_NAME, "configId": NEMO_PII_CONFIG_ID, "phases": ["Input"]},
            {"name": OUTPUT_CHECK_NAME, "configId": NEMO_PII_CONFIG_ID, "phases": ["Output"]},
        ],
        teardown=True,
    ) as ai_guardrail:
        wait_for_ai_guardrail_bound(ai_guardrail=ai_guardrail)
        yield ai_guardrail


@pytest.fixture(scope="class")
def input_check_header(pii_ai_guardrail: AIGuardrail) -> str:
    """Selection header compiled for the input-phase PII check."""
    return guardrail_check_header(
        namespace=pii_ai_guardrail.namespace,
        guardrail_name=pii_ai_guardrail.name,
        check_name=INPUT_CHECK_NAME,
    )


@pytest.fixture(scope="class")
def output_check_header(pii_ai_guardrail: AIGuardrail) -> str:
    """Selection header compiled for the output-phase PII check."""
    return guardrail_check_header(
        namespace=pii_ai_guardrail.namespace,
        guardrail_name=pii_ai_guardrail.name,
        check_name=OUTPUT_CHECK_NAME,
    )


@pytest.fixture(scope="class")
def input_guarded_subscription(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
    guardrails_model_ref: MaaSModelRef,
    pii_ai_guardrail: AIGuardrail,
) -> Generator[MaaSSubscription, Any, Any]:
    """Subscription attaching only the input-phase PII check."""
    with guarded_maas_subscription(
        admin_client=admin_client,
        namespace=maas_subscription_namespace.name,
        name=generate_random_name(prefix="e2e-guardrails-input"),
        model_ref=guardrails_model_ref,
        guardrails=[{"ref": {"name": pii_ai_guardrail.name}, "checks": [INPUT_CHECK_NAME]}],
    ) as subscription:
        yield subscription


@pytest.fixture(scope="class")
def output_guarded_subscription(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
    guardrails_model_ref: MaaSModelRef,
    pii_ai_guardrail: AIGuardrail,
) -> Generator[MaaSSubscription, Any, Any]:
    """Subscription attaching only the output-phase PII check."""
    with guarded_maas_subscription(
        admin_client=admin_client,
        namespace=maas_subscription_namespace.name,
        name=generate_random_name(prefix="e2e-guardrails-output"),
        model_ref=guardrails_model_ref,
        guardrails=[{"ref": {"name": pii_ai_guardrail.name}, "checks": [OUTPUT_CHECK_NAME]}],
    ) as subscription:
        yield subscription


@pytest.fixture(scope="class")
def unguarded_subscription(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
    guardrails_model_ref: MaaSModelRef,
) -> Generator[MaaSSubscription, Any, Any]:
    """Subscription on the same model with no guardrail attachments."""
    with guarded_maas_subscription(
        admin_client=admin_client,
        namespace=maas_subscription_namespace.name,
        name=generate_random_name(prefix="e2e-guardrails-none"),
        model_ref=guardrails_model_ref,
        guardrails=None,
    ) as subscription:
        yield subscription


@pytest.fixture(scope="class")
def guardrail_checks_enforced(
    admin_client: DynamicClient,
    input_guarded_subscription: MaaSSubscription,
    output_guarded_subscription: MaaSSubscription,
    input_check_header: str,
    output_check_header: str,
) -> None:
    """Wait until both attached checks are compiled, unauthenticated, into the rolled-out Praxis ExtProc."""
    check_headers = [input_check_header, output_check_header]
    wait_for_guardrail_checks_compiled(admin_client=admin_client, check_headers=check_headers)
    verify_guardrail_callouts_unauthenticated(admin_client=admin_client, check_headers=check_headers)


@pytest.fixture(scope="class")
def input_guarded_api_key(
    request_session_http: requests.Session,
    base_url: str,
    current_client_token: str,
    maas_api_gateway_reachable: None,
    input_guarded_subscription: MaaSSubscription,
) -> Generator[str, Any, Any]:
    """API key bound to the input-guarded subscription."""
    with maas_api_key_for_subscription(
        request_session_http=request_session_http,
        base_url=base_url,
        ocp_user_token=current_client_token,
        subscription=input_guarded_subscription,
    ) as api_key:
        yield api_key


@pytest.fixture(scope="class")
def output_guarded_api_key(
    request_session_http: requests.Session,
    base_url: str,
    current_client_token: str,
    maas_api_gateway_reachable: None,
    output_guarded_subscription: MaaSSubscription,
) -> Generator[str, Any, Any]:
    """API key bound to the output-guarded subscription."""
    with maas_api_key_for_subscription(
        request_session_http=request_session_http,
        base_url=base_url,
        ocp_user_token=current_client_token,
        subscription=output_guarded_subscription,
    ) as api_key:
        yield api_key


@pytest.fixture(scope="class")
def unguarded_api_key(
    request_session_http: requests.Session,
    base_url: str,
    current_client_token: str,
    maas_api_gateway_reachable: None,
    unguarded_subscription: MaaSSubscription,
) -> Generator[str, Any, Any]:
    """API key bound to the unguarded subscription."""
    with maas_api_key_for_subscription(
        request_session_http=request_session_http,
        base_url=base_url,
        ocp_user_token=current_client_token,
        subscription=unguarded_subscription,
    ) as api_key:
        yield api_key


@pytest.fixture(scope="class")
def guardrails_model_url(
    maas_scheme: str,
    maas_host: str,
    guardrails_echo_llmisvc: LLMInferenceService,
) -> str:
    """Chat completions URL of the echo simulator on the MaaS gateway."""
    return f"{maas_scheme}://{maas_host}/llm/{guardrails_echo_llmisvc.name}{OpenAIEnpoints.CHAT_COMPLETIONS}"
