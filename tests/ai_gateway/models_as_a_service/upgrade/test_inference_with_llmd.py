import time

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace

from tests.ai_gateway.models_as_a_service.maas_api_key.utils import (
    MAAS_GATEWAY_AUTH_POLICY_NAME,
    wait_for_auth_policy_accepted,
)
from tests.ai_gateway.models_as_a_service.upgrade.utils import (
    verify_maas_auth_policy_exists,
    verify_maas_model_ref_exists,
    verify_maas_subscription_ready,
)
from tests.ai_gateway.models_as_a_service.utils import (
    MaaSTenantResource,
    assert_api_key_created_ok,
    build_maas_headers,
    create_api_key,
    verify_maas_gateway_programmed,
    verify_maas_tenant_ready,
)
from tests.model_serving.model_server.upgrade.utils import (  # noqa: NIT001
    get_llmisvc_restart_counts,
    load_baseline_from_configmap,
)
from utilities.logger import RedactedString
from utilities.resources.llm_inference_service import LLMInferenceService

LOGGER = structlog.get_logger(name=__name__)
MAAS_LLMD_UPGRADE_PROMPT = "Reply with exactly this text: <MaaS upgrade smoke test>"


def _assert_maas_stack_ready(
    admin_client: DynamicClient,
    gateway: Gateway,
    tenant: MaaSTenantResource,
    namespace: Namespace,
    llmisvc: LLMInferenceService,
    model_ref: MaaSModelRef,
    auth_policy: MaaSAuthPolicy,
    subscription: MaaSSubscription,
    phase: str,
) -> None:
    """Assert that the MaaS control plane and LLM-d workload are ready."""
    # Verify the shared MaaS control plane is ready before checking the workload.
    LOGGER.info(event=f"[{phase}] Checking MaaS Gateway is Programmed")
    verify_maas_gateway_programmed(gateway=gateway, timeout=300)

    LOGGER.info(event=f"[{phase}] Checking MaaS Tenant is Ready")
    verify_maas_tenant_ready(tenant_resource=tenant, timeout=300)

    # Verify the llm-d namespace and model-serving resource are present and ready.
    assert namespace.exists, f"LLM-d namespace '{namespace.name}' does not exist"
    assert llmisvc.exists, f"LLMInferenceService '{llmisvc.name}' not found in namespace '{llmisvc.namespace}'"
    LOGGER.info(event=f"[{phase}] Checking LLMInferenceService is Ready")
    llmisvc.wait_for_condition(condition="Ready", status="True", timeout=900)

    # Verify MaaS resolves the LLMInferenceService as a runtime model.
    LOGGER.info(event=f"[{phase}] Checking MaaS ModelRef is RuntimeReady")
    verify_maas_model_ref_exists(model_ref=model_ref)
    model_ref.wait_for_condition(condition="RuntimeReady", status="True", timeout=300)

    # Verify the user policy and the generated gateway policy are reconciled.
    LOGGER.info(event=f"[{phase}] Checking MaaS AuthPolicy is Ready")
    verify_maas_auth_policy_exists(auth_policy=auth_policy)
    auth_policy.wait_for_condition(condition="Ready", status="True", timeout=300)
    wait_for_auth_policy_accepted(
        admin_client=admin_client,
        policy_name=MAAS_GATEWAY_AUTH_POLICY_NAME,
        namespace=gateway.namespace,
        timeout=300,
        reconciliation_hint=(
            "Ensure the LLM-d MaaSAuthPolicy is Ready and the generated gateway policy is reconciled."
        ),
    )

    # Verify the subscription authorizes access to this model.
    LOGGER.info(event=f"[{phase}] Checking MaaS Subscription is Ready")
    verify_maas_subscription_ready(subscription=subscription)
    subscription.wait_for_condition(condition="Ready", status="True", timeout=300)


def _assert_maas_inference_succeeds(
    request_session_http: requests.Session,
    maas_upgrade_base_url: str,
    api_key: str,
    llmisvc: LLMInferenceService,
    phase: str,
    prompt: str = MAAS_LLMD_UPGRADE_PROMPT,
    max_tokens: int = 6,
    temperature: float = 0,
) -> None:
    """Send an authenticated MaaS chat-completion request and validate its response."""
    # Build the endpoint and canonical model identity from the configured LLMInferenceService.
    inference_base_url = maas_upgrade_base_url.removesuffix("/maas-api")
    endpoint = f"{inference_base_url.rstrip('/')}/v1/chat/completions"
    resource = llmisvc.instance.to_dict()
    model = resource.get("spec", {}).get("model", {})
    model_name = model.get("name") or llmisvc.name
    model_identity = f"publishers/{llmisvc.namespace.strip('/')}/models/{model_name.strip('/')}"

    # Send an authenticated request using the parameters for this coverage case.
    request_started = time.monotonic()
    response = request_session_http.post(
        url=endpoint,
        headers=build_maas_headers(token=api_key),
        json={
            "model": model_identity,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    elapsed_seconds = round(time.monotonic() - request_started, 2)

    # Validate the stable response contract without logging or asserting generated text.
    assert response.status_code == 200, (
        f"[{phase}] MaaS chat completion returned HTTP {response.status_code}: {response.text[:200]}"
    )
    response_body = response.json()
    assert isinstance(response_body, dict), f"[{phase}] MaaS response must be a JSON object"

    model_alias = model_identity.rsplit("/", maxsplit=1)[-1]
    assert response_body.get("model") in (model_identity, model_alias), (
        f"[{phase}] MaaS response returned an unexpected model: {response_body.get('model')!r}"
    )
    choices = response_body.get("choices")
    assert isinstance(choices, list) and choices, f"[{phase}] MaaS response must contain a choice"
    assert isinstance(choices[0], dict), f"[{phase}] MaaS response choice must be an object"

    first_choice = choices[0]
    LOGGER.info(
        event=f"[{phase}] MaaS inference succeeded",
        model=response_body.get("model"),
        status_code=response.status_code,
        elapsed_seconds=elapsed_seconds,
        finish_reason=first_choice.get("finish_reason"),
    )


@pytest.mark.pre_upgrade
class TestMaaSInferenceWithLlmDPreUpgrade:
    """Verify the MaaS LLM-d data plane before the platform upgrade."""

    @pytest.mark.dependency(name="maas_llmd_stack_ready_pre_upgrade")
    def test_maas_stack_ready_pre_upgrade(
        self,
        admin_client: DynamicClient,
        maas_upgrade_gateway: Gateway,
        maas_upgrade_tenant: MaaSTenantResource,
        maas_inference_with_llmd_namespace: Namespace,
        maas_inference_with_llmd_llmisvc: LLMInferenceService,
        maas_inference_with_llmd_model_ref: MaaSModelRef,
        maas_inference_with_llmd_auth_policy: MaaSAuthPolicy,
        maas_inference_with_llmd_subscription: MaaSSubscription,
        maas_inference_with_llmd_api_key: str,
    ) -> None:
        """Given pre-upgrade MaaS resources, when readiness is checked, then the stack is ready."""
        assert maas_inference_with_llmd_api_key, "Pre-upgrade MaaS API key is empty"
        _assert_maas_stack_ready(
            admin_client=admin_client,
            gateway=maas_upgrade_gateway,
            tenant=maas_upgrade_tenant,
            namespace=maas_inference_with_llmd_namespace,
            llmisvc=maas_inference_with_llmd_llmisvc,
            model_ref=maas_inference_with_llmd_model_ref,
            auth_policy=maas_inference_with_llmd_auth_policy,
            subscription=maas_inference_with_llmd_subscription,
            phase="PRE-UPGRADE",
        )

    @pytest.mark.dependency(depends=["maas_llmd_stack_ready_pre_upgrade"])
    def test_maas_inference_with_llmd_pre_upgrade(
        self,
        request_session_http: requests.Session,
        maas_upgrade_base_url: str,
        maas_inference_with_llmd_llmisvc: LLMInferenceService,
        maas_inference_with_llmd_api_key: str,
    ) -> None:
        """Given a ready pre-upgrade MaaS stack, when chat completion runs, then inference succeeds."""
        _assert_maas_inference_succeeds(
            request_session_http=request_session_http,
            maas_upgrade_base_url=maas_upgrade_base_url,
            api_key=maas_inference_with_llmd_api_key,
            llmisvc=maas_inference_with_llmd_llmisvc,
            phase="PRE-UPGRADE",
        )


@pytest.mark.post_upgrade
class TestMaaSInferenceWithLlmDPostUpgrade:
    """Verify that the MaaS LLM-d data plane survives the platform upgrade."""

    @pytest.mark.dependency(name="maas_llmd_stack_ready_post_upgrade")
    def test_maas_stack_ready_post_upgrade(
        self,
        admin_client: DynamicClient,
        maas_upgrade_gateway: Gateway,
        maas_upgrade_tenant: MaaSTenantResource,
        maas_inference_with_llmd_namespace: Namespace,
        maas_inference_with_llmd_llmisvc: LLMInferenceService,
        maas_inference_with_llmd_model_ref: MaaSModelRef,
        maas_inference_with_llmd_auth_policy: MaaSAuthPolicy,
        maas_inference_with_llmd_subscription: MaaSSubscription,
        maas_inference_with_llmd_api_key: str,
    ) -> None:
        """Given an upgraded cluster, when the saved MaaS resources are checked, then they remain ready."""
        assert maas_inference_with_llmd_api_key, "Post-upgrade MaaS API key could not be loaded"
        _assert_maas_stack_ready(
            admin_client=admin_client,
            gateway=maas_upgrade_gateway,
            tenant=maas_upgrade_tenant,
            namespace=maas_inference_with_llmd_namespace,
            llmisvc=maas_inference_with_llmd_llmisvc,
            model_ref=maas_inference_with_llmd_model_ref,
            auth_policy=maas_inference_with_llmd_auth_policy,
            subscription=maas_inference_with_llmd_subscription,
            phase="POST-UPGRADE",
        )

    @pytest.mark.dependency(
        name="maas_llmisvc_pods_stable_post_upgrade",
        depends=["maas_llmd_stack_ready_post_upgrade"],
    )
    def test_maas_llmisvc_pods_unchanged_post_upgrade(
        self,
        admin_client: DynamicClient,
        maas_inference_with_llmd_llmisvc: LLMInferenceService,
    ) -> None:
        """Given a ready upgraded MaaS stack, when pod state is compared, then its LLMISVC pods are unchanged."""
        baselines = load_baseline_from_configmap(
            client=admin_client,
            namespace=maas_inference_with_llmd_llmisvc.namespace,
        )
        assert maas_inference_with_llmd_llmisvc.name in baselines, (
            f"LLMInferenceService '{maas_inference_with_llmd_llmisvc.name}' is missing from the upgrade baseline"
        )
        baseline = baselines[maas_inference_with_llmd_llmisvc.name]
        assert "restart_counts" in baseline, (
            f"LLMInferenceService '{maas_inference_with_llmd_llmisvc.name}' baseline has no restart counts"
        )

        current_restart_counts = get_llmisvc_restart_counts(
            client=admin_client,
            llmisvc=maas_inference_with_llmd_llmisvc,
        )
        assert current_restart_counts, (
            f"No pods found for LLMInferenceService '{maas_inference_with_llmd_llmisvc.name}' "
            f"in namespace '{maas_inference_with_llmd_llmisvc.namespace}'"
        )
        assert current_restart_counts == baseline["restart_counts"], (
            "MaaS LLMInferenceService pods changed during the upgrade: "
            f"expected={baseline['restart_counts']}, current={current_restart_counts}"
        )

    @pytest.mark.dependency(
        depends=[
            "maas_llmd_stack_ready_post_upgrade",
            "maas_llmisvc_pods_stable_post_upgrade",
        ]
    )
    def test_maas_inference_with_llmd_post_upgrade(
        self,
        request_session_http: requests.Session,
        maas_upgrade_base_url: str,
        maas_inference_with_llmd_llmisvc: LLMInferenceService,
        maas_inference_with_llmd_api_key: str,
    ) -> None:
        """Given a ready post-upgrade MaaS stack, when the same key calls chat completion, then inference succeeds."""
        _assert_maas_inference_succeeds(
            request_session_http=request_session_http,
            maas_upgrade_base_url=maas_upgrade_base_url,
            api_key=maas_inference_with_llmd_api_key,
            llmisvc=maas_inference_with_llmd_llmisvc,
            phase="POST-UPGRADE",
        )

    @pytest.mark.dependency(
        depends=[
            "maas_llmd_stack_ready_post_upgrade",
            "maas_llmisvc_pods_stable_post_upgrade",
        ]
    )
    def test_maas_inference_with_new_api_key_post_upgrade(
        self,
        request_session_http: requests.Session,
        current_client_token: str,
        maas_upgrade_base_url: str,
        maas_inference_with_llmd_subscription: MaaSSubscription,
        maas_inference_with_llmd_llmisvc: LLMInferenceService,
    ) -> None:
        """Given a new post-upgrade API key, when it sends varied requests, then both succeed."""
        # Create a target-version key; unlike the pre-upgrade key, it need not cross runs.
        response, body = create_api_key(
            base_url=maas_upgrade_base_url,
            ocp_user_token=current_client_token,
            request_session_http=request_session_http,
            api_key_name="maas-post-upgrade-api-key",  # pragma: allowlist secret
            subscription=maas_inference_with_llmd_subscription.name,
            expires_in="1h",
        )
        assert_api_key_created_ok(
            resp=response,
            body=body,
            required_fields=("id", "key"),  # pragma: allowlist secret
        )
        api_key = RedactedString(value=body["key"])

        # Confirm the new key works for the standard request.
        _assert_maas_inference_succeeds(
            request_session_http=request_session_http,
            maas_upgrade_base_url=maas_upgrade_base_url,
            api_key=api_key,
            llmisvc=maas_inference_with_llmd_llmisvc,
            phase="POST-UPGRADE-NEW-KEY-REQUEST-1",
        )
        # Reuse the same key with a different prompt and token limit.
        _assert_maas_inference_succeeds(
            request_session_http=request_session_http,
            maas_upgrade_base_url=maas_upgrade_base_url,
            api_key=api_key,
            llmisvc=maas_inference_with_llmd_llmisvc,
            phase="POST-UPGRADE-NEW-KEY-REQUEST-2",
            prompt="Reply with exactly: MaaS upgrade follow-up.",
            max_tokens=8,
        )
