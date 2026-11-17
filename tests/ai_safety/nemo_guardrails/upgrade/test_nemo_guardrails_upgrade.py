import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.nemo_guardrails import NemoGuardrails
from ocp_resources.pod import Pod
from ocp_resources.route import Route

from tests.ai_safety.nemo_guardrails.constants import (
    CHAT_ENDPOINT,
    CHECK_ENDPOINT,
    MODEL_NAME,
    SAFE_PROMPTS,
)
from tests.ai_safety.nemo_guardrails.utils import (
    send_request,
    validate_nemo_guardrails_images,
    verify_auth_required,
)


@pytest.mark.tier1
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails-upgrade"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoGuardrailsLLMAsPreUpgrade:
    """
    Tests for NeMo Guardrails server operations with LLM-as-a-Judge configuration (pre upgrade).

    This test class validates:
    1. Server deployment with LLM-as-a-judge config
    2. Backend LLM communication
    3. Authentication enforcement via kube-rbac-proxy
    4. Container image validation
    """

    @pytest.mark.pre_upgrade
    def test_nemo_llm_judge_validate_images_pre_upgrade(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_llm_judge: NemoGuardrails,
        trustyai_operator_configmap,
    ):
        """
        Test to verify NeMo Guardrails pod images.

        Given: NeMo Guardrails LLM-as-a-judge deployment with auth enabled
        When: Pod images are inspected
        Then: Images match ConfigMap and are pinned with sha256 digest
        """
        # Get NeMo Guardrails pod
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={nemo_guardrails_llm_judge.name}",
            )
        )
        assert len(pods) > 0, f"No pods found for {nemo_guardrails_llm_judge.name}"

        # Validate images (ignoring registry differences, checking image name + SHA256)
        validate_nemo_guardrails_images(pod=pods[0], trustyai_operator_configmap=trustyai_operator_configmap)

    @pytest.mark.pre_upgrade
    def test_nemo_llm_judge_deployment_pre_upgrade(
        self,
        llm_d_inference_sim_isvc: InferenceService,
        nemo_guardrails_llm_judge: NemoGuardrails,
        nemo_guardrails_llm_judge_route: Route,
        nemo_guardrails_llm_judge_healthcheck,
    ):
        """
        Test NeMo Guardrails server deploys successfully.

        Given: NeMo Guardrails CR with LLM-as-a-judge config
        When: Deployment is created
        Then: Server is running and route exists
        """
        assert nemo_guardrails_llm_judge.exists
        assert nemo_guardrails_llm_judge_route.exists
        assert nemo_guardrails_llm_judge_route.host is not None

    @pytest.mark.parametrize("endpoint", [CHAT_ENDPOINT, CHECK_ENDPOINT])
    @pytest.mark.pre_upgrade
    def test_nemo_llm_judge_backend_communication_pre_upgrade(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        llm_d_inference_sim_isvc: InferenceService,
        nemo_guardrails_llm_judge: NemoGuardrails,
        nemo_guardrails_llm_judge_route: Route,
        nemo_guardrails_llm_judge_healthcheck,
        endpoint: str,
    ):
        """
        Test NeMo Guardrails can communicate with backend LLM.

        Given: NeMo Guardrails with backend LLM configured
        When: Chat request is sent
        Then: Request successfully reaches backend and returns response
        """
        url = f"https://{nemo_guardrails_llm_judge_route.host}{endpoint}"

        response = send_request(
            url=url,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            message=SAFE_PROMPTS[0],
            model=MODEL_NAME,
            configuration=None,
        )

        # Just verify we got a valid response (not testing guardrails functionality)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        response_json = response.json()

        # Validate response format based on endpoint
        if endpoint == CHAT_ENDPOINT:
            assert "choices" in response_json, "Chat endpoint should contain choices"
            assert len(response_json["choices"]) > 0, "Response should have at least one choice"
        elif endpoint == CHECK_ENDPOINT:
            assert "status" in response_json, "Check endpoint should contain status"
            assert response_json["status"] in ["blocked", "success"], "Check should return a valid status"

    @pytest.mark.parametrize("endpoint", [CHAT_ENDPOINT, CHECK_ENDPOINT])
    @pytest.mark.pre_upgrade
    def test_nemo_llm_judge_with_authentication_pre_upgrade(
        self,
        openshift_ca_bundle_file: str,
        llm_d_inference_sim_isvc: InferenceService,
        nemo_guardrails_llm_judge: NemoGuardrails,
        nemo_guardrails_llm_judge_route: Route,
        nemo_guardrails_llm_judge_healthcheck,
        endpoint: str,
    ):
        """
        Test that authentication is enforced via kube-rbac-proxy.

        Given: NeMo Guardrails with auth enabled
        When: Request is made without authentication token
        Then: Returns 401/403 Unauthorized
        """
        url = f"https://{nemo_guardrails_llm_judge_route.host}{endpoint}"

        response = send_request(
            url=url,
            token=None,  # No token
            ca_bundle_file=openshift_ca_bundle_file,
            message=SAFE_PROMPTS[0],
            model=MODEL_NAME,
            configuration=None,
        )

        verify_auth_required(response=response)

@pytest.mark.tier1
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails-upgrade"})],
    indirect=True,
)
class TestNemoGuardrailsLLMAsPostUpgrade:
    """
    Tests for NeMo Guardrails server operations with LLM-as-a-Judge configuration (post upgrade).

    This test class validates:
    1. Server deployment with LLM-as-a-judge config
    2. Backend LLM communication
    3. Authentication enforcement via kube-rbac-proxy
    4. Container image validation
    """

    @pytest.mark.post_upgrade
    def test_nemo_llm_judge_validate_images_post_upgrade(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_llm_judge: NemoGuardrails,
        trustyai_operator_configmap,
    ):
        """
        Test to verify NeMo Guardrails pod images.

        Given: NeMo Guardrails LLM-as-a-judge deployment with auth enabled
        When: Pod images are inspected
        Then: Images match ConfigMap and are pinned with sha256 digest
        """
        # Get NeMo Guardrails pod
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={nemo_guardrails_llm_judge.name}",
            )
        )
        assert len(pods) > 0, f"No pods found for {nemo_guardrails_llm_judge.name}"

        # Validate images (ignoring registry differences, checking image name + SHA256)
        validate_nemo_guardrails_images(pod=pods[0], trustyai_operator_configmap=trustyai_operator_configmap)

    @pytest.mark.post_upgrade
    def test_nemo_llm_judge_deployment_post_upgrade(
        self,
        llm_d_inference_sim_isvc: InferenceService,
        nemo_guardrails_llm_judge: NemoGuardrails,
        nemo_guardrails_llm_judge_route: Route,
        nemo_guardrails_llm_judge_healthcheck,
    ):
        """
        Test NeMo Guardrails server deploys successfully.

        Given: NeMo Guardrails CR with LLM-as-a-judge config
        When: Deployment is created
        Then: Server is running and route exists
        """
        assert nemo_guardrails_llm_judge.exists
        assert nemo_guardrails_llm_judge_route.exists
        assert nemo_guardrails_llm_judge_route.host is not None

    @pytest.mark.parametrize("endpoint", [CHAT_ENDPOINT, CHECK_ENDPOINT])
    @pytest.mark.post_upgrade
    def test_nemo_llm_judge_backend_communication_post_upgrade(
        self,
        openshift_ca_bundle_file: str,
        current_client_token: str,
        llm_d_inference_sim_isvc: InferenceService,
        nemo_guardrails_llm_judge: NemoGuardrails,
        nemo_guardrails_llm_judge_route: Route,
        nemo_guardrails_llm_judge_healthcheck,
        endpoint: str,
    ):
        """
        Test NeMo Guardrails can communicate with backend LLM.

        Given: NeMo Guardrails with backend LLM configured
        When: Chat request is sent
        Then: Request successfully reaches backend and returns response
        """
        url = f"https://{nemo_guardrails_llm_judge_route.host}{endpoint}"

        response = send_request(
            url=url,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            message=SAFE_PROMPTS[0],
            model=MODEL_NAME,
            configuration=None,
        )

        # Just verify we got a valid response (not testing guardrails functionality)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        response_json = response.json()

        # Validate response format based on endpoint
        if endpoint == CHAT_ENDPOINT:
            assert "choices" in response_json, "Chat endpoint should contain choices"
            assert len(response_json["choices"]) > 0, "Response should have at least one choice"
        elif endpoint == CHECK_ENDPOINT:
            assert "status" in response_json, "Check endpoint should contain status"
            assert response_json["status"] in ["blocked", "success"], "Check should return a valid status"

    @pytest.mark.parametrize("endpoint", [CHAT_ENDPOINT, CHECK_ENDPOINT])
    @pytest.mark.post_upgrade
    def test_nemo_llm_judge_with_authentication_post_upgrade(
        self,
        openshift_ca_bundle_file: str,
        llm_d_inference_sim_isvc: InferenceService,
        nemo_guardrails_llm_judge: NemoGuardrails,
        nemo_guardrails_llm_judge_route: Route,
        nemo_guardrails_llm_judge_healthcheck,
        endpoint: str,
    ):
        """
        Test that authentication is enforced via kube-rbac-proxy.

        Given: NeMo Guardrails with auth enabled
        When: Request is made without authentication token
        Then: Returns 401/403 Unauthorized
        """
        url = f"https://{nemo_guardrails_llm_judge_route.host}{endpoint}"

        response = send_request(
            url=url,
            token=None,  # No token
            ca_bundle_file=openshift_ca_bundle_file,
            message=SAFE_PROMPTS[0],
            model=MODEL_NAME,
            configuration=None,
        )

        verify_auth_required(response=response)
