import pytest
import requests
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.namespace import Namespace
from timeout_sampler import retry

from tests.model_explainability.guardrails.constants import (
    AUTOCONFIG_DETECTOR_LABEL,
    AUTOCONFIG_GATEWAY_ENDPOINT,
    CHAT_COMPLETIONS_DETECTION_ENDPOINT,
    HAP_INPUT_DETECTION_PROMPT,
    HARMLESS_PROMPT,
    PII_ENDPOINT,
    PII_INPUT_DETECTION_PROMPT,
    PII_OUTPUT_DETECTION_PROMPT,
    PROMPT_INJECTION_INPUT_DETECTION_PROMPT,
    STANDALONE_DETECTION_ENDPOINT,
    TEST_TLS_CERTIFICATE,
)
from tests.model_explainability.guardrails.utils import (
    create_detector_config,
    send_and_verify_negative_detection,
    send_and_verify_standalone_detection,
    send_and_verify_unsuitable_input_detection,
    send_and_verify_unsuitable_output_detection,
    verify_health_info_response,
)
from tests.model_explainability.utils import validate_tai_component_images
from utilities.constants import (
    BUILTIN_DETECTOR_CONFIG,
    HAP_DETECTOR,
    LLM_D_CHAT_GENERATION_CONFIG,
    PROMPT_INJECTION_DETECTOR,
    LLMdInferenceSimConfig,
    Timeout,
)
from utilities.plugins.constant import OpenAIEnpoints

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.smoke
@pytest.mark.model_explainability
def test_guardrailsorchestrator_crd_exists(
    admin_client: DynamicClient,
) -> None:
    """Verify GuardrailsOrchestrator CRD exists on the cluster."""
    crd_name = "guardrailsorchestrators.trustyai.opendatahub.io"

    crd_resource = CustomResourceDefinition(
        client=admin_client,
        name=crd_name,
        ensure_exists=True,
    )

    assert crd_resource.exists, f"CRD {crd_name} does not exist on the cluster"


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace, orchestrator_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-image"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "openai": LLM_D_CHAT_GENERATION_CONFIG,
                        "detectors": BUILTIN_DETECTOR_CONFIG,
                    })
                },
            },
            {"orchestrator_config": True, "enable_built_in_detectors": False, "enable_guardrails_gateway": False},
        )
    ],
    indirect=True,
)
def test_validate_guardrails_orchestrator_images(
    model_namespace,
    orchestrator_config,
    guardrails_orchestrator,
    guardrails_orchestrator_pod,
    trustyai_operator_configmap,
):
    """Test to verify Guardrails pod images.
    Checks if the image tag from the ConfigMap is used within the Pod and if it's pinned using a sha256 digest.
    """
    validate_tai_component_images(pod=guardrails_orchestrator_pod, tai_operator_configmap=trustyai_operator_configmap)


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace, orchestrator_config, guardrails_gateway_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-builtin"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "openai": LLM_D_CHAT_GENERATION_CONFIG,
                        "detectors": BUILTIN_DETECTOR_CONFIG,
                    })
                },
            },
            {
                "guardrails_gateway_config_data": {
                    "config.yaml": yaml.dump({
                        "orchestrator": {
                            "host": "localhost",
                            "port": 8032,
                        },
                        "detectors": [
                            {
                                "name": "regex",
                                "input": True,
                                "output": True,
                                "detector_params": {"regex": ["email", "ssn"]},
                            },
                        ],
                        "routes": [
                            {"name": "pii", "detectors": ["regex"]},
                            {"name": "passthrough", "detectors": []},
                        ],
                    })
                },
            },
            {
                "orchestrator_config": True,
                "enable_built_in_detectors": True,
                "enable_guardrails_gateway": True,
                "guardrails_gateway_config": True,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "guardrails_gateway_config")
class TestGuardrailsOrchestratorWithBuiltInDetectors:
    """
    Tests if basic functions of the GuardrailsOrchestrator are working properly with the built-in (regex) detectors.
        1. Deploy an LLM using vLLM as a SR.
        2. Deploy the Guardrails Orchestrator.
        3. Check that the Orchestrator is healthy by querying the health and info endpoints of its /health route.
        4. Check that the built-in regex detectors work as expected:
         4.1. Unsuitable input detection.
         4.2. Unsuitable output detection.
         4.3. No detection.
        5. Check that the /passthrough endpoint forwards the
         query directly to the model without performing any detection.
    """

    def test_guardrails_info_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_health_route,
        guardrails_healthcheck,
    ):
        verify_health_info_response(
            host=guardrails_orchestrator_health_route.host,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
        )

    def test_guardrails_builtin_detectors_unsuitable_input(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        send_and_verify_unsuitable_input_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PII_INPUT_DETECTION_PROMPT,
            model=LLMdInferenceSimConfig.model_name,
        )

    def test_guardrails_builtin_detectors_unsuitable_output(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        send_and_verify_unsuitable_output_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{PII_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            prompt=PII_OUTPUT_DETECTION_PROMPT,
            model=LLMdInferenceSimConfig.model_name,
        )

    @pytest.mark.parametrize(
        "message, url_path",
        [
            pytest.param(
                HARMLESS_PROMPT,
                PII_ENDPOINT,
                id="harmless_input",
            ),
            pytest.param(PII_INPUT_DETECTION_PROMPT.content, "/passthrough", id="passthrough_endpoint"),
        ],
    )
    def test_guardrails_builtin_detectors_negative_detection(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_gateway_route,
        message,
        url_path,
        guardrails_healthcheck,
    ):
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{url_path}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=str(message),
            model=LLMdInferenceSimConfig.model_name,
        )


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace, orchestrator_config, guardrails_gateway_config,guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-huggingface"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "openai": LLM_D_CHAT_GENERATION_CONFIG,
                        "detectors": {
                            PROMPT_INJECTION_DETECTOR: {
                                "type": "text_contents",
                                "service": {
                                    "hostname": f"{PROMPT_INJECTION_DETECTOR}-predictor",
                                    "port": 80,
                                },
                                "chunker_id": "whole_doc_chunker",
                                "default_threshold": 0.5,
                            },
                            HAP_DETECTOR: {
                                "type": "text_contents",
                                "service": {
                                    "hostname": f"{HAP_DETECTOR}-predictor",
                                    "port": 80,
                                },
                                "chunker_id": "whole_doc_chunker",
                                "default_threshold": 0.5,
                            },
                        },
                    })
                },
            },
            {
                "guardrails_gateway_config_data": {
                    "config.yaml": yaml.dump({
                        "orchestrator": {
                            "host": "localhost",
                            "port": 8032,
                        },
                        "detectors": [
                            {
                                "name": "regex",
                                "input": True,
                                "output": True,
                                "detector_params": {"regex": ["email", "ssn"]},
                            },
                        ],
                        "routes": [
                            {"name": "pii", "detectors": ["regex"]},
                            {"name": "passthrough", "detectors": []},
                        ],
                    })
                },
            },
            {
                "orchestrator_config": True,
                "enable_built_in_detectors": False,
                "enable_guardrails_gateway": True,
                "guardrails_gateway_config": True,
                "otel_exporter_config": True,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.usefixtures(
    "patched_dsc_kserve_headed",
    "guardrails_gateway_config",
    "minio_pvc_otel",
    "minio_deployment_otel",
    "minio_service_otel",
    "minio_secret_otel",
    "installed_tempo_operator",
    "installed_opentelemetry_operator",
    "tempo_stack",
    "otel_collector",
)
class TestGuardrailsOrchestratorWithHuggingFaceDetectors:
    """
    These tests verify that the GuardrailsOrchestrator works as expected when using HuggingFace detectors
    Steps:
        - Deploy an LLM (Qwen2.5-0.5B-Instruct) using the vLLM SR.
        - Deploy the GuardrailsOrchestrator.
        - Deploy a prompt injection detector using the HuggingFace SR.
        - Check that the detector works when we have an unsuitable input.
        - Check that the detector works when we have a harmless input (no detection).
         - Check the standalone detections by querying its /text/detection/content endpoint, verifying that input
           detection is correctly performed.
    """

    def test_guardrails_multi_detector_unsuitable_input(
        self,
        current_client_token,
        llm_d_inference_sim_isvc,
        guardrails_orchestrator_route,
        prompt_injection_detector_route,
        hap_detector_route,
        openshift_ca_bundle_file,
        orchestrator_config,
        guardrails_orchestrator,
        otel_collector,
        tempo_stack,
        guardrails_healthcheck,
    ):
        for prompt in [PROMPT_INJECTION_INPUT_DETECTION_PROMPT, HAP_INPUT_DETECTION_PROMPT]:
            send_and_verify_unsuitable_input_detection(
                url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
                token=current_client_token,
                ca_bundle_file=openshift_ca_bundle_file,
                prompt=prompt,
                model=LLMdInferenceSimConfig.model_name,
                detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
            )

    def test_guardrails_multi_detector_negative_detection(
        self,
        current_client_token,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        hap_detector_route,
        prompt_injection_detector_route,
        openshift_ca_bundle_file,
        otel_collector,
        tempo_stack,
        guardrails_healthcheck,
    ):
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=HARMLESS_PROMPT,
            model=LLMdInferenceSimConfig.model_name,
            detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
        )

    def test_guardrails_standalone_detector_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        orchestrator_config,
        guardrails_orchestrator_route,
        hap_detector_route,
        otel_collector,
        tempo_stack,
        guardrails_healthcheck,
    ):
        send_and_verify_standalone_detection(
            url=f"https://{guardrails_orchestrator_route.host}/{STANDALONE_DETECTION_ENDPOINT}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            detector_name=HAP_DETECTOR,
            content=HAP_INPUT_DETECTION_PROMPT.content,
            expected_min_score=0.9,
        )

    def test_guardrails_traces_in_tempo(
        self,
        admin_client,
        model_namespace,
        orchestrator_config,
        guardrails_orchestrator,
        guardrails_gateway_config,
        otel_collector,
        tempo_stack,
        tempo_traces_service_portforward,
        guardrails_healthcheck,
    ):
        """
        Ensure that OpenTelemetry traces from Guardrails Orchestrator are collected in Tempo.
        Equivalent to clicking 'Find Traces' in the Tempo UI.
        """

        @retry(wait_timeout=Timeout.TIMEOUT_1MIN, sleep=5)
        def check_traces():
            services = requests.get(f"{tempo_traces_service_portforward}/api/services").json().get("data") or []
            guardrails_services = [s for s in services if "guardrails" in s]
            if not guardrails_services:
                return False

            svc = guardrails_services[0]

            traces = requests.get(f"{tempo_traces_service_portforward}/api/traces?service={svc}").json()

            if traces.get("data"):
                return traces

        check_traces()


@pytest.mark.parametrize(
    "model_namespace, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-guardrails-autoconfig"},
            {
                "auto_config": {
                    "inferenceServiceToGuardrail": LLMdInferenceSimConfig.isvc_name,
                    "detectorServiceLabelToMatch": AUTOCONFIG_DETECTOR_LABEL,
                },
            },
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
@pytest.mark.tier1
@pytest.mark.rawdeployment
class TestGuardrailsOrchestratorAutoConfig:
    """
    These tests verify that the GuardrailsOrchestrator works as expected when configured through the AutoConfig feature.
    """

    def test_guardrails_gateway_info_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        guardrails_orchestrator_health_route,
        guardrails_healthcheck,
    ):
        verify_health_info_response(
            host=guardrails_orchestrator_health_route.host,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
        )

    def test_guardrails_autoconfig_unsuitable_input(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        guardrails_orchestrator_route,
        guardrails_healthcheck,
    ):
        for prompt in [HAP_INPUT_DETECTION_PROMPT, PROMPT_INJECTION_INPUT_DETECTION_PROMPT]:
            send_and_verify_unsuitable_input_detection(
                url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
                token=current_client_token,
                ca_bundle_file=openshift_ca_bundle_file,
                prompt=prompt,
                model=LLMdInferenceSimConfig.model_name,
                detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
            )

    def test_guardrails_autoconfig_negative_detection(
        self,
        current_client_token,
        llm_d_inference_sim_isvc,
        guardrails_orchestrator_route,
        openshift_ca_bundle_file,
        guardrails_healthcheck,
    ):
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_route.host}/{CHAT_COMPLETIONS_DETECTION_ENDPOINT}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=str(HARMLESS_PROMPT),
            model=LLMdInferenceSimConfig.model_name,
            detectors=create_detector_config(PROMPT_INJECTION_DETECTOR, HAP_DETECTOR),
        )


@pytest.mark.parametrize(
    "model_namespace, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-autoconfig-gateway"},
            {
                "auto_config": {
                    "inferenceServiceToGuardrail": LLMdInferenceSimConfig.isvc_name,
                    "detectorServiceLabelToMatch": AUTOCONFIG_DETECTOR_LABEL,
                },
                "enable_built_in_detectors": True,
                "enable_guardrails_gateway": True,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
@pytest.mark.tier2
@pytest.mark.rawdeployment
class TestGuardrailsOrchestratorAutoConfigWithGateway:
    """
    These tests verify that the GuardrailsOrchestrator works as expected when configured
    through the AutoConfig feature to use the gateway route.
    """

    def test_guardrails_autoconfig_gateway_info_endpoint(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        hap_detector_isvc,
        prompt_injection_detector_isvc,
        guardrails_orchestrator_health_route,
        guardrails_healthcheck,
    ):
        verify_health_info_response(
            host=guardrails_orchestrator_health_route.host,
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
        )

    def test_guardrails_autoconfig_gateway_unsuitable_input(
        self,
        current_client_token,
        openshift_ca_bundle_file,
        llm_d_inference_sim_isvc,
        prompt_injection_detector_isvc,
        hap_detector_isvc,
        guardrails_orchestrator_gateway_route,
        guardrails_healthcheck,
    ):
        for prompt in [HAP_INPUT_DETECTION_PROMPT, PROMPT_INJECTION_INPUT_DETECTION_PROMPT]:
            send_and_verify_unsuitable_input_detection(
                url=f"https://{guardrails_orchestrator_gateway_route.host}"
                f"{AUTOCONFIG_GATEWAY_ENDPOINT}{OpenAIEnpoints.CHAT_COMPLETIONS}",
                token=current_client_token,
                ca_bundle_file=openshift_ca_bundle_file,
                prompt=prompt,
                model=LLMdInferenceSimConfig.model_name,
            )

    @pytest.mark.parametrize(
        "message, url_path",
        [
            pytest.param(
                HARMLESS_PROMPT,
                AUTOCONFIG_GATEWAY_ENDPOINT,
                id="harmless_input",
            ),
            pytest.param(PII_INPUT_DETECTION_PROMPT.content, "/passthrough", id="passthrough_endpoint"),
        ],
    )
    def test_guardrails_autoconfig_gateway_negative_detection(
        self,
        current_client_token,
        llm_d_inference_sim_isvc,
        prompt_injection_detector_isvc,
        hap_detector_isvc,
        guardrails_orchestrator_gateway_route,
        openshift_ca_bundle_file,
        url_path,
        message,
        guardrails_healthcheck,
    ):
        send_and_verify_negative_detection(
            url=f"https://{guardrails_orchestrator_gateway_route.host}{url_path}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            token=current_client_token,
            ca_bundle_file=openshift_ca_bundle_file,
            content=str(message),
            model=LLMdInferenceSimConfig.model_name,
        )


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace, orchestrator_config, guardrails_orchestrator_with_tls",
    [
        pytest.param(
            {"name": "test-guardrails-custom-tls"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "openai": LLM_D_CHAT_GENERATION_CONFIG,
                        "detectors": BUILTIN_DETECTOR_CONFIG,
                    })
                },
            },
            {"tls_secrets": ["custom-tls-cert"]},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
class TestGuardrailsOrchestratorCustomTLS:
    """
    Tests custom TLS certificate mounting for the GuardrailsOrchestrator.
    Verifies that custom TLS secrets specified in the CR are correctly mounted
    to the orchestrator deployment at /etc/tls/$SECRET_NAME.

    Tests are split into dependent steps for better granularity:
    1. Volume mount check
    2. Volume references correct secret
    3. Certificate files exist in the pod
    4. Certificate content matches expected
    """

    @pytest.mark.dependency(name="volume_mount_check")
    def test_volume_mount_exists(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        custom_tls_secret,
        guardrails_orchestrator_with_tls,
        guardrails_orchestrator_pod_with_tls,
    ):
        """Verify the volume mount exists in the pod spec."""
        pod = guardrails_orchestrator_pod_with_tls

        # Verify the volume mount exists in the pod spec (scan all containers to handle sidecars)
        volume_mounts = [
            volume_mount
            for container in pod.instance.spec.containers
            for volume_mount in (container.volumeMounts or [])
            if volume_mount.mountPath == "/etc/tls/custom-tls-cert"
        ]

        assert volume_mounts, "Custom TLS volume mount not found in any container"
        assert volume_mounts[0].name, "Volume mount has no name"

    @pytest.mark.dependency(name="volume_secret_reference", depends=["volume_mount_check"])
    def test_volume_references_correct_secret(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        custom_tls_secret,
        guardrails_orchestrator_with_tls,
        guardrails_orchestrator_pod_with_tls,
    ):
        """Verify the volume references the correct secret."""
        pod = guardrails_orchestrator_pod_with_tls

        volume_mounts = [
            volume_mount
            for container in pod.instance.spec.containers
            for volume_mount in (container.volumeMounts or [])
            if volume_mount.mountPath == "/etc/tls/custom-tls-cert"
        ]

        volumes = [volume for volume in pod.instance.spec.volumes if volume.name == volume_mounts[0].name]
        assert volumes, f"Volume {volume_mounts[0].name} not found in pod volumes"
        assert volumes[0].secret.secretName == "custom-tls-cert", (  # pragma: allowlist secret
            "Volume does not reference the correct secret"
        )

    @pytest.mark.dependency(name="cert_files_exist", depends=["volume_secret_reference"])
    def test_certificate_files_exist(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        custom_tls_secret,
        guardrails_orchestrator_with_tls,
        guardrails_orchestrator_pod_with_tls,
    ):
        """Verify certificate files exist in the pod."""
        pod = guardrails_orchestrator_pod_with_tls

        # Use test -f for authoritative file existence checks
        cert_file_check = "test -f /etc/tls/custom-tls-cert/tls.crt && echo 'cert_exists'"
        cert_result = pod.execute(command=["sh", "-c", cert_file_check])
        assert "cert_exists" in cert_result, "TLS certificate file not found in mounted path"

        key_file_check = "test -f /etc/tls/custom-tls-cert/tls.key && echo 'key_exists'"  # pragma: allowlist secret
        key_result = pod.execute(command=["sh", "-c", key_file_check])
        assert "key_exists" in key_result, "TLS key file not found in mounted path"  # pragma: allowlist secret

    @pytest.mark.dependency(depends=["cert_files_exist"])
    def test_certificate_content_matches(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        custom_tls_secret,
        guardrails_orchestrator_with_tls,
        guardrails_orchestrator_pod_with_tls,
    ):
        """Verify the certificate content matches expected test certificate."""
        pod = guardrails_orchestrator_pod_with_tls

        cert_content_cmd = "cat /etc/tls/custom-tls-cert/tls.crt"
        cert_content = pod.execute(command=["sh", "-c", cert_content_cmd])

        # Normalize whitespace for comparison
        expected_cert = TEST_TLS_CERTIFICATE.strip()
        actual_cert = cert_content.strip()

        assert actual_cert == expected_cert, (
            f"Mounted certificate content does not match expected test certificate. "
            f"Expected length: {len(expected_cert)}, Actual length: {len(actual_cert)}"
        )

        LOGGER.info(
            f"Custom TLS secret successfully mounted and verified at /etc/tls/custom-tls-cert in pod {pod.name}"
        )
