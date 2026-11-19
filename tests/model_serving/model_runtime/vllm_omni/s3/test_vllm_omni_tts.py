"""vLLM-Omni TTS endpoint, security, and RBAC tests using Qwen3-TTS."""

import re
import time
from typing import Any

import pytest
import requests
import structlog
import urllib3
from kubernetes.client.rest import ApiException
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.role_binding import RoleBinding
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config
from urllib3.exceptions import InsecureRequestWarning

from tests.model_serving.model_runtime.vllm_omni.constant import (
    AUDIO_SPEECH_ENDPOINT,
    MODEL_MOUNT_DIR,
    QWEN3_TTS_MODEL_PATH,
)
from tests.model_serving.model_runtime.vllm_omni.utils import get_predictor_pods_or_fail, run_tts_inference
from utilities.constants import Containers, KServeDeploymentType, RuntimeTemplates
from utilities.plugins.constant import OpenAIEnpoints
from utilities.resources.egress_firewall import EgressFirewall
from utilities.resources.pod import Pod as ExtPod

urllib3.disable_warnings(category=InsecureRequestWarning)

LOGGER = structlog.get_logger(name=__name__)


STANDARD_TTS_PROMPT: str = "Welcome to Red Hat OpenShift AI. This is a test of the text to speech capability."
# Fixed 20-word prompt for the timeout-boundary assertion.
TWENTY_WORD_PROMPT: str = (
    "Red Hat OpenShift AI enables data scientists to build train deploy "
    "and serve machine learning models at enterprise scale reliably."
)
# Multi-word prompt used for size comparison in single-word edge case test.
MULTI_WORD_TTS_PROMPT: str = "OpenShift AI enables data scientists to deploy models at scale."


# PII / binary-content detection for log auditing
_BASE64_BLOCK_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")
_SAFE_LOG_LINE_PATTERN: re.Pattern[str] = re.compile(
    r"site-packages/|\.py:\d|\.go:\d|/opt/|/usr/|sha256:|git[-+]|"
    r"WARNING |INFO |DEBUG |ERROR |reflector\.go|version\.py"
)
_WAV_BINARY_MARKER: str = "RIFF"
_BIOMETRIC_LOG_KEYWORDS: list[str] = ["speaker_id", "biometric", "voice_print", "speaker_embedding"]


def _find_base64_leaks(log_text: str) -> str | None:
    """Scan log text line-by-line for base64-encoded audio data.

    Skips lines matching known-safe patterns (file paths, version strings,
    Python tracebacks) to avoid false positives from hex hashes and paths.
    Returns the first suspicious match or None.
    """
    for line in log_text.splitlines():
        if _SAFE_LOG_LINE_PATTERN.search(string=line):
            continue
        match = _BASE64_BLOCK_PATTERN.search(string=line)
        if match:
            text = match.group()
            has_upper = any(c.isupper() for c in text)
            has_lower = any(c.islower() for c in text)
            has_plus = "+" in text
            if has_upper and has_lower and has_plus:
                return text
    return None


# TTS prompts with PII-like content used for the log audit
_TTS_PII_PROMPTS: list[str] = [
    "Hello, my name is John Smith and my phone number is 555-867-5309.",
    "Please call Jane Doe at 123-456-7890 after 3 PM.",
    "Contact Dr. Alice Johnson at alice.johnson@example.com.",
]

# Model file format identifiers for safetensors verification
_SAFETENSORS_KEYWORD: str = "safetensors"
_PICKLE_EXTENSIONS: list[str] = [".bin", ".pkl", ".pickle"]

# External host probed to confirm egress blocking
_EXTERNAL_EGRESS_IP: str = "1.1.1.1"
_EGRESS_PROBE_CONNECT_TIMEOUT_S: int = 10
# Exit code the probe script returns when the outbound connection is blocked by the policy
_EGRESS_PROBE_BLOCKED_RC: int = 2
# Budget for OVN-Kubernetes to enforce the EgressFirewall in the node data plane; enforcement
# is asynchronous and can take several seconds after the resource object is created
_EGRESS_ENFORCEMENT_TIMEOUT_S: int = 60
_EGRESS_ENFORCEMENT_POLL_INTERVAL_S: int = 3
# EgressFirewall rules: allow cluster-internal destinations, deny the public internet. The
# allowed CIDRs are private address space covering the pod network (10/8), the service network
# (172.16/12) and the OVN join subnet (100.64/16); none are routable on the public internet.
_EGRESS_FIREWALL_RULES: list[dict[str, Any]] = [
    {"type": "Allow", "to": {"cidrSelector": "10.0.0.0/8"}},
    {"type": "Allow", "to": {"cidrSelector": "172.16.0.0/12"}},
    {"type": "Allow", "to": {"cidrSelector": "100.64.0.0/16"}},
    {"type": "Deny", "to": {"cidrSelector": "0.0.0.0/0"}},
]


pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


def request_tts(
    base_url: str,
    body: dict[str, Any],
    timeout: int = 60,
) -> requests.Response:
    """POST to /v1/audio/speech and return the raw HTTP response.

    The successful response body is binary audio data; callers should use
    response.content to read the bytes and must NOT call response.json() on
    a 200 reply.

    Args:
        base_url: Base HTTPS URL of the vLLM-Omni InferenceService.
        body: JSON-serialisable TTS request payload.
        timeout: Per-request socket timeout in seconds (default 60).

    Returns:
        Raw requests.Response object.
    """
    return requests.post(
        f"{base_url}/v1/audio/speech",
        json=body,
        headers={"Content-Type": "application/json"},
        verify=False,
        timeout=timeout,
    )


def _wait_for_external_egress_blocked(exec_pod: ExtPod) -> None:
    """Poll the external-egress probe until the egress restriction blocks it.

    OVN-Kubernetes enforces egress restrictions asynchronously (controller -> OVN
    NBDB -> ovn-northd -> SBDB -> the node's ovn-controller -> OVS flows), so a rule
    is not effective in the data plane the moment its object is created.

    Args:
        exec_pod: Predictor pod to run the probe in.

    Raises:
        AssertionError: If the probe fails with an unexpected exit code.
        Failed: If external egress is still allowed after the enforcement timeout.
    """
    probe_command: list[str] = [
        "python3",
        "-c",
        (
            "import socket, sys\n"
            "connection = socket.socket()\n"
            f"connection.settimeout({_EGRESS_PROBE_CONNECT_TIMEOUT_S})\n"
            "try:\n"
            f"    connection.connect(('{_EXTERNAL_EGRESS_IP}', 443))\n"
            "except OSError:\n"
            f"    sys.exit({_EGRESS_PROBE_BLOCKED_RC})\n"
            "sys.exit(0)\n"
        ),
    ]
    enforcement_start = time.monotonic()
    enforcement_deadline = enforcement_start + _EGRESS_ENFORCEMENT_TIMEOUT_S
    while time.monotonic() < enforcement_deadline:
        probe_result = exec_pod.execute(
            command=probe_command,
            container=Containers.KSERVE_CONTAINER_NAME,
            timeout=_EGRESS_PROBE_CONNECT_TIMEOUT_S + 15,
        )
        if probe_result.rc == _EGRESS_PROBE_BLOCKED_RC:
            LOGGER.info(
                event="egress restriction enforced in the data plane",
                external_ip=_EXTERNAL_EGRESS_IP,
                enforced_after_s=round(time.monotonic() - enforcement_start, 1),
            )
            return
        assert probe_result.rc == 0, (
            f"Egress probe returned unexpected exit code {probe_result.rc} "
            f"(stdout={probe_result.stdout!r}, stderr={probe_result.stderr!r}); "
            "expected a successful or a blocked external connection"
        )
        LOGGER.debug(
            event="egress NetworkPolicy not enforced yet, retrying probe",
            external_ip=_EXTERNAL_EGRESS_IP,
            enforcement_timeout_s=_EGRESS_ENFORCEMENT_TIMEOUT_S,
        )
        time.sleep(_EGRESS_ENFORCEMENT_POLL_INTERVAL_S)
    pytest.fail(
        f"Outbound TCP connection to {_EXTERNAL_EGRESS_IP}:443 was still allowed "
        f"{_EGRESS_ENFORCEMENT_TIMEOUT_S}s after the egress restriction was applied; "
        "external internet access must be blocked"
    )


@pytest.mark.vllm_omni_nvidia_single_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, vllm_omni_serving_runtime, vllm_omni_inference_service",
    [
        pytest.param(
            {"name": "vllm-omni-tts-qwen3"},
            {"model-dir": QWEN3_TTS_MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                "name": "vllm-omni-tts-qwen3",
                "model_path": QWEN3_TTS_MODEL_PATH,
                "deployment_mode": KServeDeploymentType.STANDARD,
                "gpu_count": 1,
            },
            id="test_vllm_omni_tts_qwen3",
        ),
    ],
    indirect=True,
)
class TestVllmOmniTtsQwen3:
    """TTS and security tests using the Qwen3-TTS InferenceService.

    Covers multi-format TTS synthesis, single-word edge case, invalid payload
    handling, and consolidated security/RBAC tests: no PII in logs, safetensors-only
    model format, external egress isolation via EgressFirewall, Cluster Admin CRUD,
    Namespace-Scoped User read-only access, and Monitoring User metrics access.
    """

    @pytest.mark.parametrize(
        "response_format",
        [
            pytest.param("wav", id="test_response_format_wav"),
            pytest.param("mp3", id="test_response_format_mp3"),
            pytest.param("flac", id="test_response_format_flac"),
        ],
    )
    def test_vllm_omni_tts_qwen3_basic(
        self,
        vllm_omni_inference_service: InferenceService,
        response_format: str,
    ) -> None:
        """Basic /v1/audio/speech inference returns valid audio per format.

        Given a vLLM-Omni InferenceService serving Qwen3-TTS is deployed and ready,
        When POST /v1/audio/speech is called with a standard text input and the
        specified audio format (parametrized over wav, mp3, flac),
        Then the response has HTTP 200, a matching Content-Type header, a body
        exceeding 44 bytes, and magic bytes match the format (wav/flac).
        """
        response = run_tts_inference(
            isvc=vllm_omni_inference_service,
            text=STANDARD_TTS_PROMPT,
            response_format=response_format,
        )
        LOGGER.info(
            event="standard TTS response validated",
            status=response.status_code,
            content_type=response.headers.get("Content-Type"),
            body_bytes=len(response.content),
            response_format=response_format,
        )

    def test_vllm_omni_tts_qwen3_20_word_no_timeout(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
    ) -> None:
        """A 20-word prompt completes without HTTP 504 (timeout boundary).

        Given a Qwen3-TTS ISVC is deployed and Ready,
        When a /v1/audio/speech request is sent with a ~20-word prompt,
        Then the response is HTTP 200 (not 504), confirming the model handles
        longer prompts within the serving timeout window.
        """
        response = requests.post(
            url=f"{vllm_omni_isvc_url}{AUDIO_SPEECH_ENDPOINT}",
            json={
                "model": vllm_omni_inference_service.instance.metadata.name,
                "input": TWENTY_WORD_PROMPT,
                "voice": "vivian",
                "response_format": "wav",
            },
            verify=False,
            timeout=120,
        )
        assert response.status_code != 504, (
            "POST /v1/audio/speech with a 20-word prompt returned HTTP 504 Gateway Timeout; "
            "request must complete within the OpenShift Route default timeout"
        )
        assert response.status_code == 200, (
            f"POST /v1/audio/speech returned HTTP {response.status_code}; expected 200. Body: {response.text[:200]}"
        )

    def test_vllm_omni_tts_qwen3_single_word_prompt(
        self,
        vllm_omni_inference_service: InferenceService,
    ) -> None:
        """Single-word prompt does not inflate response with white-noise.

        Given a vLLM-Omni InferenceService serving Qwen3-TTS is ready,
        When POST /v1/audio/speech is called with the single-word input "Hello",
        Then the response has HTTP 200 with a valid WAV body exceeding 44 bytes,
        and the body is proportionally smaller than a multi-word synthesis
        (size confirms no white-noise artefact per known vllm-omni edge case).
        """
        single_word_response = run_tts_inference(isvc=vllm_omni_inference_service, text="Hello")
        single_word_size = len(single_word_response.content)

        multi_word_response = run_tts_inference(isvc=vllm_omni_inference_service, text=MULTI_WORD_TTS_PROMPT)
        multi_word_size = len(multi_word_response.content)

        assert single_word_size < multi_word_size, (
            f"Single-word response ({single_word_size} bytes) is not smaller than "
            f"multi-word response ({multi_word_size} bytes); possible white-noise artefact"
        )
        LOGGER.info(
            event="single-word vs multi-word TTS size comparison",
            single_word_bytes=single_word_size,
            multi_word_bytes=multi_word_size,
        )

    @pytest.mark.parametrize(
        "body_factory, description",
        [
            pytest.param(
                lambda model: {"model": model, "input": "", "voice": "vivian", "response_format": "wav"},
                "empty input field",
                id="test_invalid_empty_input",
            ),
            pytest.param(
                lambda model: {"model": model, "voice": "vivian", "response_format": "wav"},
                "missing input field",
                id="test_invalid_missing_input",
            ),
            pytest.param(
                lambda _model: {},
                "empty JSON body",
                id="test_invalid_empty_body",
            ),
        ],
    )
    def test_vllm_omni_tts_invalid_payload_errors(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
        body_factory: Any,
        description: str,
    ) -> None:
        """Invalid or incomplete payloads return 400/422 with error messages.

        Given a vLLM-Omni InferenceService is ready,
        When POST /v1/audio/speech is called with an invalid payload (parametrized
        over empty input, missing input field, and empty body),
        Then the server returns HTTP 400 or 422 with a non-empty descriptive error
        body and never returns HTTP 500.
        """
        base_url = vllm_omni_isvc_url
        invalid_body = body_factory(vllm_omni_inference_service.instance.metadata.name)  # noqa: FCN001
        response = request_tts(base_url=base_url, body=invalid_body)

        LOGGER.info(
            event="invalid TTS payload response",
            description=description,
            status=response.status_code,
            body=response.text[:500],
        )
        assert response.status_code in {400, 422}, (
            f"Invalid payload ({description!r}) returned HTTP {response.status_code}; "
            f"expected 400 or 422. Body: {response.text[:500]}"
        )
        assert response.text, f"Error response body is empty for invalid payload ({description!r})"

    def test_vllm_omni_no_pii_in_logs(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
    ) -> None:
        """Verify that audio content and PII do not appear in vLLM-Omni pod logs.

        Given a deployed vLLM-Omni InferenceService at the default INFO log level
        When TTS requests with PII-like input prompts are sent to /v1/audio/speech
        Then pod logs contain no base64-encoded audio blocks, no WAV binary markers,
             no biometric label keywords, and no verbatim copies of the input prompts.
        """
        isvc_url = vllm_omni_isvc_url
        model_name = vllm_omni_inference_service.name

        for prompt in _TTS_PII_PROMPTS:
            payload: dict[str, str] = {"model": model_name, "input": prompt, "voice": "vivian"}
            response = requests.post(
                url=f"{isvc_url}{AUDIO_SPEECH_ENDPOINT}",
                json=payload,
                verify=False,
                timeout=60,
            )
            assert response.status_code == 200, (
                f"TTS request failed for prompt {prompt!r}: HTTP {response.status_code}. Body: {response.text[:200]}"
            )

        predictor_pods = get_predictor_pods_or_fail(client=admin_client, isvc=vllm_omni_inference_service)

        for pod in predictor_pods:
            pod_logs: str = pod.log(container=Containers.KSERVE_CONTAINER_NAME)

            base64_leak = _find_base64_leaks(log_text=pod_logs)
            assert base64_leak is None, (
                f"Pod {pod.name!r} logs contain a base64-encoded block (>=120 chars, "
                f"mixed case + '+') that may indicate audio data leakage: "
                f"{base64_leak[:80]}..."
            )
            assert _WAV_BINARY_MARKER not in pod_logs, (
                f"Pod {pod.name!r} logs contain WAV binary marker {_WAV_BINARY_MARKER!r} — "
                "raw audio content must not appear in pod logs"
            )
            for biometric_keyword in _BIOMETRIC_LOG_KEYWORDS:
                assert biometric_keyword not in pod_logs, (
                    f"Pod {pod.name!r} logs contain biometric PII keyword {biometric_keyword!r}"
                )
            for prompt in _TTS_PII_PROMPTS:
                assert prompt not in pod_logs, (
                    f"Pod {pod.name!r} logs contain verbatim input prompt — PII may be exposed: {prompt!r}"
                )

    def test_vllm_omni_safetensors_only_model_format(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        vllm_omni_inference_service: InferenceService,
    ) -> None:
        """Verify safetensors files are present and pickle-format files are absent.

        Given a deployed vLLM-Omni InferenceService with weights downloaded from S3
        When pod logs and the /mnt/models directory are inspected via exec
        Then pod logs reference safetensors loading, .safetensors files exist in /mnt/models,
             and no .bin pickle-format files are present.
        """
        predictor_pods = get_predictor_pods_or_fail(client=admin_client, isvc=vllm_omni_inference_service)
        predictor_pod = predictor_pods[0]

        pod_logs: str = predictor_pod.log(container=Containers.KSERVE_CONTAINER_NAME)
        assert _SAFETENSORS_KEYWORD in pod_logs.lower(), (
            f"Pod {predictor_pod.name!r} logs do not reference 'safetensors' loading — "
            "the runtime may not be using the expected weight format"
        )
        for pickle_ext in _PICKLE_EXTENSIONS:
            assert pickle_ext not in pod_logs, (
                f"Pod {predictor_pod.name!r} logs reference pickle-format file {pickle_ext!r} — "
                "deserialization risk: only safetensors weights should be loaded"
            )

        exec_pod = ExtPod(
            client=admin_client,
            name=predictor_pod.name,
            namespace=predictor_pod.namespace,
        )

        find_st_result = exec_pod.execute(
            command=["find", MODEL_MOUNT_DIR, "-name", "*.safetensors"],
            container=Containers.KSERVE_CONTAINER_NAME,
            timeout=30,
        )
        assert find_st_result.rc == 0 and find_st_result.stdout.strip(), (
            f"No .safetensors files found in {MODEL_MOUNT_DIR}: "
            f"rc={find_st_result.rc}, stdout={find_st_result.stdout!r}, "
            f"stderr={find_st_result.stderr!r}"
        )

        for ext in _PICKLE_EXTENSIONS:
            find_result = exec_pod.execute(
                command=["find", MODEL_MOUNT_DIR, "-name", f"*{ext}"],
                container=Containers.KSERVE_CONTAINER_NAME,
                timeout=30,
            )
            assert find_result.rc == 0, (
                f"'find *{ext}' command failed: rc={find_result.rc}, stderr={find_result.stderr!r}"
            )
            assert not find_result.stdout.strip(), (
                f"Pickle-format {ext} files found in {MODEL_MOUNT_DIR}: "
                f"{find_result.stdout.strip()!r}. Only safetensors weights are permitted."
            )

        LOGGER.info(
            event="safetensors files present, no pickle files found",
            safetensors_files=find_st_result.stdout.strip(),
        )

    def test_vllm_omni_egress_firewall_blocks_external(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
    ) -> None:
        """Verify an EgressFirewall blocks external traffic while TTS inference continues.

        Given a deployed vLLM-Omni InferenceService with weights already loaded from S3
        When an OVN EgressFirewall denying public-internet egress is applied to the model
             namespace and OVN-Kubernetes has enforced it in the data plane
        Then a direct TCP connection from the predictor pod to an external IP (1.1.1.1:443) is
             blocked, and TTS inference via /v1/audio/speech returns HTTP 200 (weights are in
             pod memory, served through the OpenShift Route).
        """
        # An EgressFirewall is used instead of a NetworkPolicy because odh-model-controller
        # creates a per-InferenceService NetworkPolicy that allows all egress, and
        # NetworkPolicy rules are additive — a restrictive NetworkPolicy can never block a
        # predictor pod that is also selected by the allow-all policy.
        isvc_url = vllm_omni_isvc_url
        model_name = vllm_omni_inference_service.name

        with EgressFirewall(
            client=admin_client,
            name="default",
            namespace=model_namespace.name,
            egress=_EGRESS_FIREWALL_RULES,
        ):
            predictor_pods = get_predictor_pods_or_fail(client=admin_client, isvc=vllm_omni_inference_service)
            exec_pod = ExtPod(
                client=admin_client,
                name=predictor_pods[0].name,
                namespace=predictor_pods[0].namespace,
            )
            _wait_for_external_egress_blocked(exec_pod=exec_pod)

            tts_payload: dict[str, str] = {
                "model": model_name,
                "input": "Egress policy verification.",
                "voice": "vivian",
            }
            inference_response = requests.post(
                url=f"{isvc_url}{AUDIO_SPEECH_ENDPOINT}",
                json=tts_payload,
                verify=False,
                timeout=60,
            )
            assert inference_response.status_code == 200, (
                f"TTS inference failed after EgressFirewall applied: "
                f"HTTP {inference_response.status_code}. "
                "Weights are already in pod memory — inference must not require external access."
            )

            LOGGER.info(
                event="egress firewall blocks external traffic; inference continues",
            )

    def test_vllm_omni_cluster_admin_crud_serving_runtime(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        s3_models_storage_uri: str,
        vllm_omni_serving_runtime: ServingRuntime,
        vllm_omni_inference_service: InferenceService,
    ) -> None:
        """Verify Cluster Admin can read and patch the vLLM-Omni ServingRuntime.

        Given a vLLM-Omni ServingRuntime provisioned by the admin in a namespace
        When the admin retrieves and patches an annotation on the ServingRuntime
        Then both operations succeed without HTTP 403 Forbidden responses.
        """
        runtime = ServingRuntime(
            client=admin_client,
            name=vllm_omni_serving_runtime.name,
            namespace=model_namespace.name,
        )
        assert runtime.exists, (
            f"ServingRuntime {vllm_omni_serving_runtime.name!r} not found in "
            f"{model_namespace.name!r} — Cluster Admin GET must succeed"
        )

        with ResourceEditor(patches={runtime: {"metadata": {"annotations": {"test.rbac/verified": "true"}}}}):
            updated_annotations: dict[str, str] = runtime.instance.metadata.annotations or {}
            assert updated_annotations.get("test.rbac/verified") == "true", (
                "Cluster Admin PATCH on ServingRuntime did not persist; "
                "verify the cluster-admin ClusterRoleBinding is in place"
            )

        LOGGER.info(
            event="ServingRuntime CRUD verified for Cluster Admin",
            runtime_name=vllm_omni_serving_runtime.name,
        )

    def test_vllm_omni_cluster_admin_crud_inference_service(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        vllm_omni_inference_service: InferenceService,
    ) -> None:
        """Verify Cluster Admin can read and patch the vLLM-Omni InferenceService.

        Given a vLLM-Omni InferenceService provisioned by the admin in a namespace
        When the admin retrieves and patches an annotation on the InferenceService
        Then both operations succeed without HTTP 403 Forbidden responses.
        """
        with ResourceEditor(
            patches={vllm_omni_inference_service: {"metadata": {"annotations": {"test.rbac/verified": "true"}}}}
        ):
            updated_annotations: dict[str, str] = vllm_omni_inference_service.instance.metadata.annotations or {}
            assert updated_annotations.get("test.rbac/verified") == "true", (
                "Cluster Admin PATCH on InferenceService did not persist; "
                "verify the cluster-admin ClusterRoleBinding is in place"
            )

        LOGGER.info(
            event="InferenceService CRUD verified for Cluster Admin",
            isvc_name=vllm_omni_inference_service.name,
        )

    def test_vllm_omni_namespace_scoped_user_read_only_access(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        model_namespace: Namespace,
        vllm_omni_inference_service: InferenceService,
        non_admin_user_password: tuple[str, str] | None,
    ) -> None:
        """Verify Namespace-Scoped User has read-only access to the InferenceService.

        Given a vLLM-Omni InferenceService deployed in a namespace the user can view
        When the user retrieves the ISVC, then tries to create a new ISVC, then tries to delete
        Then GET returns InferenceService data, CREATE returns HTTP 403, DELETE returns HTTP 403.
        """
        assert non_admin_user_password, (
            "non_admin_user_password fixture is not configured. "
            "Set --non-admin-user-password CLI option or NON_ADMIN_USER_PASSWORD env var."
        )
        username = non_admin_user_password[0]
        with RoleBinding(
            client=admin_client,
            name="vllm-omni-viewer",
            namespace=model_namespace.name,
            role_ref_name="view",
            role_ref_kind="ClusterRole",
            subjects_kind="User",
            subjects_name=username,
        ):
            readable_isvc = InferenceService(
                client=unprivileged_client,
                name=vllm_omni_inference_service.name,
                namespace=model_namespace.name,
            )
            assert readable_isvc.exists, (
                f"Namespace-Scoped User should be able to GET InferenceService "
                f"{vllm_omni_inference_service.name!r} in {model_namespace.name!r}. "
                "Verify the project viewer RoleBinding is applied to the test user."
            )

            isvc_resource = unprivileged_client.resources.get(
                api_version="serving.kserve.io/v1beta1", kind="InferenceService"
            )
            with pytest.raises(ApiException) as create_exc:
                isvc_resource.create(
                    body={
                        "apiVersion": "serving.kserve.io/v1beta1",
                        "kind": "InferenceService",
                        "metadata": {
                            "name": "vllm-omni-unauthorized-create",
                            "namespace": model_namespace.name,
                        },
                        "spec": {"predictor": {"model": {"modelFormat": {"name": "vLLM"}}}},
                    },
                    namespace=model_namespace.name,
                )

            assert create_exc.value.status == 403, (
                f"Expected HTTP 403 for Namespace-Scoped User creating InferenceService, "
                f"got HTTP {create_exc.value.status}"
            )

            with pytest.raises(ApiException) as delete_exc:
                InferenceService(
                    client=unprivileged_client,
                    name=vllm_omni_inference_service.name,
                    namespace=model_namespace.name,
                ).delete()

            assert delete_exc.value.status == 403, (
                f"Expected HTTP 403 for Namespace-Scoped User deleting InferenceService, "
                f"got HTTP {delete_exc.value.status}"
            )

            LOGGER.info(
                event="Namespace-Scoped User read-only access validated",
                isvc_name=vllm_omni_inference_service.name,
            )

    def test_vllm_omni_metrics_endpoint_reachable(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
    ) -> None:
        """Verify the /metrics endpoint is reachable and returns vLLM metric families.

        Given a vLLM-Omni InferenceService with metrics enabled (default)
        When the /metrics endpoint is queried via unauthenticated HTTP GET on the Route
        Then the response is HTTP 200 and the body contains vllm metric family names.

        Note: Role-based access control is enforced at the pod level by the
        kube-rbac-proxy sidecar, not by the OpenShift Route. This test validates
        Route-level reachability only.
        """
        isvc_url = vllm_omni_isvc_url
        metrics_url = f"{isvc_url}{OpenAIEnpoints.METRICS}"

        # Unauthenticated: kube-rbac-proxy handles auth at the pod level;
        # the Route exposes metrics without additional auth.
        metrics_response = requests.get(url=metrics_url, verify=False, timeout=30)
        assert metrics_response.status_code == 200, (
            f"Metrics endpoint returned HTTP {metrics_response.status_code} — expected 200. URL: {metrics_url}"
        )
        metrics_text = metrics_response.text
        assert "vllm" in metrics_text.lower(), (
            f"Metrics response does not contain 'vllm' metric families. "
            f"Response body (first 500 chars): {metrics_text[:500]!r}"
        )

        LOGGER.info(
            event="/metrics endpoint accessible and returns vllm metrics",
            metrics_bytes=len(metrics_text),
        )

    def test_vllm_omni_monitoring_user_cannot_access_cluster_serving_runtime(
        self,
        model_namespace: Namespace,
        s3_models_storage_uri: str,
        vllm_omni_serving_runtime: ServingRuntime,
        vllm_omni_inference_service: InferenceService,
        unprivileged_client: DynamicClient,
    ) -> None:
        """Verify an unprivileged Monitoring User cannot GET the cluster-level ServingRuntime.

        Given a vLLM-Omni ServingRuntime template in the RHOAI operator namespace
        When a Monitoring User (project-viewer role) attempts to GET it
        Then the operation returns HTTP 403 Forbidden.
        """
        cluster_runtime = ServingRuntime(
            client=unprivileged_client,
            name=RuntimeTemplates.VLLM_OMNI_CUDA,
            namespace=py_config["applications_namespace"],
        )
        with pytest.raises(ApiException) as exc_info:
            _ = cluster_runtime.instance  # triggers the API call

        assert exc_info.value.status == 403, (
            f"Expected HTTP 403 for Monitoring User accessing cluster ServingRuntime, got HTTP {exc_info.value.status}"
        )
        LOGGER.info(event="Monitoring User correctly denied cluster ServingRuntime GET")
