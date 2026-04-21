"""Test suite for NeMo Guardrails in OpenDataHub/RHOAI."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.nemo_guardrails import NemoGuardrails
from ocp_resources.pod import Pod
from ocp_resources.route import Route

from tests.model_explainability.nemo_guardrails.constants import (
    CHAT_ENDPOINT,
    CHECK_ENDPOINT,
    CLEAN_PROMPT,
    MODEL_NAME,
    PII_PROMPT,
    SAFE_PROMPTS,
)
from tests.model_explainability.nemo_guardrails.utils import (
    send_request,
    validate_nemo_guardrails_images,
    verify_auth_required,
)


@pytest.mark.smoke
@pytest.mark.model_explainability
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoGuardrailsLLMAsJudge:
    """
    Tests for NeMo Guardrails server operations with LLM-as-a-Judge configuration.

    This test class validates:
    1. Server deployment with LLM-as-a-judge config
    2. Backend LLM communication
    3. Authentication enforcement via kube-rbac-proxy
    4. Container image validation
    """

    def test_nemo_llm_judge_validate_images(
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

    def test_nemo_llm_judge_deployment(
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
    def test_nemo_llm_judge_backend_communication(
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
    def test_nemo_llm_judge_with_authentication(
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


@pytest.mark.tier2
@pytest.mark.model_explainability
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoGuardrailsMultiServer:
    """
    Tests for multiple independent NeMo Guardrails servers in the same namespace.

    This test class validates:
    1. Multiple NeMo Guardrails CRs can coexist
    2. Each server operates independently
    3. Different configurations work correctly
    4. Separate routes and services are created
    """

    @pytest.mark.parametrize("endpoint", [CHAT_ENDPOINT, CHECK_ENDPOINT])
    def test_nemo_multi_server_isolation(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        openshift_ca_bundle_file: str,
        llm_d_inference_sim_isvc: InferenceService,
        nemo_guardrails_presidio: NemoGuardrails,
        nemo_guardrails_second_server: NemoGuardrails,
        nemo_guardrails_presidio_route: Route,
        nemo_guardrails_second_server_route: Route,
        nemo_guardrails_presidio_healthcheck,
        nemo_guardrails_second_server_healthcheck,
        endpoint: str,
    ):
        """
        Test that multiple NeMo Guardrails servers operate independently.

        Given: Two NeMo Guardrails servers in the same namespace
        When: Both servers are queried
        Then: Each server responds correctly and independently
        """
        # Test first server
        url1 = f"https://{nemo_guardrails_presidio_route.host}{endpoint}"
        response1 = send_request(
            url=url1,
            token=None,
            ca_bundle_file=openshift_ca_bundle_file,
            message=CLEAN_PROMPT,
            model=MODEL_NAME,
            configuration=None,
        )
        assert response1.status_code == 200, f"Expected 200, got {response1.status_code}"
        response1_json = response1.json()

        # Validate response format based on endpoint
        if endpoint == CHAT_ENDPOINT:
            assert "choices" in response1_json, "Chat endpoint should contain choices"
        elif endpoint == CHECK_ENDPOINT:
            assert "status" in response1_json, "Check endpoint should contain status"

        # Test second server
        url2 = f"https://{nemo_guardrails_second_server_route.host}{endpoint}"
        response2 = send_request(
            url=url2,
            token=None,
            ca_bundle_file=openshift_ca_bundle_file,
            message=CLEAN_PROMPT,
            model=MODEL_NAME,
            configuration=None,
        )
        assert response2.status_code == 200, f"Expected 200, got {response2.status_code}"
        response2_json = response2.json()

        # Validate response format based on endpoint
        if endpoint == CHAT_ENDPOINT:
            assert "choices" in response2_json, "Chat endpoint should contain choices"
        elif endpoint == CHECK_ENDPOINT:
            assert "status" in response2_json, "Check endpoint should contain status"

        # Verify different hosts
        assert nemo_guardrails_presidio_route.host != nemo_guardrails_second_server_route.host

    def test_nemo_multi_server_different_configs(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_llm_judge: NemoGuardrails,
        nemo_guardrails_presidio: NemoGuardrails,
    ):
        """
        Test that multiple servers can have different configurations.

        Given: Two NeMo Guardrails servers with different configs
        When: CR specs are inspected
        Then: Each has distinct nemoConfigs
        """
        # Verify different servers
        assert nemo_guardrails_llm_judge.name != nemo_guardrails_presidio.name
        assert nemo_guardrails_llm_judge.namespace == nemo_guardrails_presidio.namespace

        # Fetch actual nemoConfigs from CR specs
        llm_judge_configs = nemo_guardrails_llm_judge.instance.spec.nemoConfigs
        presidio_configs = nemo_guardrails_presidio.instance.spec.nemoConfigs

        # Assert configs are not equal
        assert llm_judge_configs != presidio_configs, "NemoConfigs should differ between servers"

        # Extract ConfigMap names from each config
        llm_judge_configmaps = {cm for config in llm_judge_configs for cm in config.get("configMaps", [])}
        presidio_configmaps = {cm for config in presidio_configs for cm in config.get("configMaps", [])}

        # Assert ConfigMap sets differ (servers don't share ConfigMaps)
        assert llm_judge_configmaps != presidio_configmaps, (
            f"ConfigMaps should differ: llm_judge={llm_judge_configmaps}, presidio={presidio_configmaps}"
        )


@pytest.mark.tier2
@pytest.mark.model_explainability
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoGuardrailsMultiConfig:
    """
    Tests for a single NeMo Guardrails server with multiple named configurations.

    This test class validates:
    1. Multiple nemoConfig entries in spec
    2. Default configuration selection
    3. Configuration mounting to correct paths
    """

    @pytest.mark.parametrize("endpoint", [CHAT_ENDPOINT, CHECK_ENDPOINT])
    def test_nemo_multi_config_default_selection(
        self,
        openshift_ca_bundle_file: str,
        llm_d_inference_sim_isvc: InferenceService,
        nemo_guardrails_multi_config: NemoGuardrails,
        nemo_guardrails_multi_config_route: Route,
        nemo_guardrails_multi_config_healthcheck,
        endpoint: str,
    ):
        """
        Test that default configuration is used correctly.

        Given: NeMo Guardrails with multiple configurations
        When: Request is made without specifying config
        Then: Default configuration (config-a) is used
        """
        url = f"https://{nemo_guardrails_multi_config_route.host}{endpoint}"

        # Test with safe prompt (should use default config)
        response = send_request(
            url=url,
            token=None,
            ca_bundle_file=openshift_ca_bundle_file,
            message=SAFE_PROMPTS[0],
            model=MODEL_NAME,
            configuration=None,
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        response_json = response.json()

        # Validate response format and config selection based on endpoint
        if endpoint == CHAT_ENDPOINT:
            assert "choices" in response_json, "Chat endpoint should contain choices"
            assert "guardrails" in response_json, "Response should contain guardrails metadata"
            assert response_json["guardrails"]["config_id"] == "config-a", "Should use default config-a"
        elif endpoint == CHECK_ENDPOINT:
            assert "status" in response_json, "Check endpoint should contain status"
            assert "guardrails_data" in response_json, "Check response should contain guardrails_data"
            # Verify the check was performed (guardrails_data should have config information)
            assert response_json["guardrails_data"], "guardrails_data should not be empty"

    @pytest.mark.parametrize("endpoint", [CHAT_ENDPOINT, CHECK_ENDPOINT])
    def test_nemo_multi_config(
        self,
        openshift_ca_bundle_file: str,
        llm_d_inference_sim_isvc: InferenceService,
        nemo_guardrails_multi_config: NemoGuardrails,
        nemo_guardrails_multi_config_route: Route,
        nemo_guardrails_multi_config_healthcheck,
        endpoint: str,
    ):
        """
        Test explicit configuration selection.

        Given: NeMo Guardrails with multiple configurations
        When: Request is made with specific config_id
        Then: Specified configuration is used
        """
        url = f"https://{nemo_guardrails_multi_config_route.host}{endpoint}"

        # Test with safe prompt (should use LLM-as-a-judge config-a)
        response = send_request(
            url=url,
            token=None,
            ca_bundle_file=openshift_ca_bundle_file,
            message=SAFE_PROMPTS[0],
            model=MODEL_NAME,
            configuration="config-a",
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        response_json = response.json()

        # Validate response format and config selection based on endpoint
        if endpoint == CHAT_ENDPOINT:
            assert "choices" in response_json, "Chat endpoint should contain choices"
            assert "guardrails" in response_json, "Response should contain guardrails metadata"
            assert response_json["guardrails"]["config_id"] == "config-a", "Should use config-a"
        elif endpoint == CHECK_ENDPOINT:
            assert "status" in response_json, "Check endpoint should contain status"
            assert "guardrails_data" in response_json, "Check response should contain guardrails_data"
            # LLM-as-a-judge config should perform checks
            assert response_json["guardrails_data"], "guardrails_data should not be empty for config-a"

        # Test with pii prompt (should use presidio config-b)
        response = send_request(
            url=url,
            token=None,
            ca_bundle_file=openshift_ca_bundle_file,
            message=PII_PROMPT,
            model=MODEL_NAME,
            configuration="config-b",
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        response_json = response.json()

        # Validate response format and config selection based on endpoint
        if endpoint == CHAT_ENDPOINT:
            assert "choices" in response_json, "Chat endpoint should contain choices"
            assert "guardrails" in response_json, "Response should contain guardrails metadata"
            assert response_json["guardrails"]["config_id"] == "config-b", "Should use config-b"
        elif endpoint == CHECK_ENDPOINT:
            assert "status" in response_json, "Check endpoint should contain status"
            assert "guardrails_data" in response_json, "Check response should contain guardrails_data"
            # Presidio config should perform PII checks
            assert response_json["guardrails_data"], "guardrails_data should not be empty for config-b"

    def test_nemo_multi_config_mount_paths(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_multi_config: NemoGuardrails,
    ):
        """
        Test that multiple configurations are mounted to correct paths.

        Given: NeMo Guardrails with multiple configurations
        When: Pod volumes are inspected
        Then: ConfigMaps are mounted to /app/config/{name}
        """
        # Get expected config names from CR spec
        nemo_configs = nemo_guardrails_multi_config.instance.spec.nemoConfigs
        expected_config_names = {config["name"] for config in nemo_configs}
        assert len(expected_config_names) >= 2, "Expected at least 2 configs in CR spec"

        # Get NeMo Guardrails pod
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=model_namespace.name,
                label_selector=f"app={nemo_guardrails_multi_config.name}",
            )
        )
        assert len(pods) > 0, f"No pods found for {nemo_guardrails_multi_config.name}"

        pod = pods[0]

        # Collect all volumeMounts from containers and initContainers
        all_volume_mounts = []

        # Check main containers
        for container in pod.instance.spec.containers:
            if container.volumeMounts:
                all_volume_mounts.extend(container.volumeMounts)

        # Check init containers
        if pod.instance.spec.initContainers:
            for init_container in pod.instance.spec.initContainers:
                if init_container.volumeMounts:
                    all_volume_mounts.extend(init_container.volumeMounts)

        # Find main service container (contains "nemo" in name)
        main_container = next(
            (c for c in pod.instance.spec.containers if "nemo" in c.name.lower()),
            pod.instance.spec.containers[0],  # Fallback to first container
        )

        assert main_container.volumeMounts, "Main container should have volume mounts"

        # Verify each config has a mount at /app/config/{name}
        main_container_mounts = {vm.mountPath: vm.name for vm in main_container.volumeMounts}

        for config_name in expected_config_names:
            expected_mount_path = f"/app/config/{config_name}"
            assert expected_mount_path in main_container_mounts, (
                f"Missing mount path {expected_mount_path} in main container. "
                f"Found mounts: {list(main_container_mounts.keys())}"
            )


@pytest.mark.tier1
@pytest.mark.model_explainability
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-nemo-guardrails"})],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestNemoGuardrailsSecretMounting:
    """
    Tests for mounting model API tokens as secrets via environment variables.

    This test class validates:
    1. Secrets are mounted as environment variables
    2. Secrets are used for model API authentication
    3. Multiple secrets can be mounted
    """

    def test_nemo_secret_env_var_mounted(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        nemo_guardrails_llm_judge: NemoGuardrails,
        nemo_api_token_secret,
    ):
        """
        Test that API token secret is mounted as environment variable.

        Given: NeMo Guardrails with env var from secret
        When: Pod environment is inspected
        Then: OPENAI_API_KEY env var is present with secretKeyRef
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

        # Verify environment variable with secret reference
        pod = pods[0]
        containers = pod.instance.spec.containers
        main_container = next((c for c in containers if "nemo" in c.name.lower()), containers[0])

        env_vars = {env.name: env for env in main_container.env}
        assert "OPENAI_API_KEY" in env_vars, "OPENAI_API_KEY not found in environment"

        # Verify it's from a secret
        api_key_env = env_vars["OPENAI_API_KEY"]
        assert api_key_env.valueFrom is not None, "Expected valueFrom for OPENAI_API_KEY"
        assert api_key_env.valueFrom.secretKeyRef is not None, "Expected secretKeyRef"
        assert api_key_env.valueFrom.secretKeyRef.name == nemo_api_token_secret.name

    @pytest.mark.parametrize("endpoint", [CHAT_ENDPOINT, CHECK_ENDPOINT])
    def test_nemo_secret_used_for_auth(
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
        Test that mounted secret is used for model API calls.

        Given: NeMo Guardrails with OPENAI_API_KEY from secret
        When: Request is made to NeMo Guardrails
        Then: Request succeeds (secret is used for backend model auth)
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

        # Should succeed if secret is properly mounted and used
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        response_json = response.json()

        # Validate response format based on endpoint
        if endpoint == CHAT_ENDPOINT:
            assert "choices" in response_json, "Chat endpoint should contain choices"
        elif endpoint == CHECK_ENDPOINT:
            assert "status" in response_json, "Check endpoint should contain status"
