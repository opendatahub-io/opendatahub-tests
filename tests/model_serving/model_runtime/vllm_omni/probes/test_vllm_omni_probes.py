"""Tests for vLLM-Omni initialization sequence and probe behavior."""

import time
from collections.abc import Generator
from copy import deepcopy
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_runtime.utils import pod_is_ready
from tests.model_serving.model_runtime.vllm.utils import dedupe_vllm_cli_args
from tests.model_serving.model_runtime.vllm_omni.constant import (
    OMNI_MULTI_GPU_RESOURCES,
    OMNI_SERVING_ARGS,
    OMNI_VOLUME_MOUNTS,
    OMNI_VOLUMES,
    QWEN3_TTS_MODEL_PATH,
)
from tests.model_serving.model_runtime.vllm_omni.probes.utils import (
    HEALTH_POLL_INTERVAL_SECONDS,
    has_crash_loop_backoff,
    safe_exec_health_probe,
    wait_for_pod_running,
)
from utilities.constants import KServeDeploymentType, Labels, Ports
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label

LOGGER = structlog.get_logger(name=__name__)

pytestmark = pytest.mark.usefixtures("valid_aws_config")


# Source: RHAISTRAT-2493 (Readiness and Health Probes section)
# Probe specs imported from constant.py (single source of truth).
_OMNI_HEALTH_HTTP_GET: dict[str, Any] = {
    "path": "/health",
    "port": Ports.REST_PORT,
    "scheme": "HTTP",
}

# Timing constants
HEALTH_STABILITY_WINDOW_SECONDS: int = 60
HEALTH_TRANSITION_TIMEOUT_SECONDS: int = 600  # 10 min; max wait for /health → 200 during init


def _safe_exec_health_probe(pod: Pod) -> str:
    """Local wrapper passing the module-level httpGet config to the shared helper."""
    return safe_exec_health_probe(pod=pod, http_get=_OMNI_HEALTH_HTTP_GET)


@pytest.fixture(scope="class")
def vllm_omni_unready_isvc(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    vllm_omni_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    vllm_omni_model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    """vLLM-Omni InferenceService started but not yet waited to be pod-ready.

    Yields the ISVC as soon as the Kubernetes resource is created so that
    tests can observe the initialization phase before all three stages (LLM,
    TTS, Code2Wav) complete.  Used to capture the non-200 → 200 /health
    transition.
    """
    from tests.model_serving.model_runtime.vllm_omni.constant import (
        OMNI_SINGLE_GPU_RESOURCES,
    )

    gpu_count = request.param.get("gpu_count", 1)
    base_resources = OMNI_MULTI_GPU_RESOURCES if gpu_count > 1 else OMNI_SINGLE_GPU_RESOURCES
    resources = deepcopy(x=base_resources["resources"])
    resources["requests"][Labels.Nvidia.NVIDIA_COM_GPU] = gpu_count
    resources["limits"][Labels.Nvidia.NVIDIA_COM_GPU] = gpu_count

    serving_args = list(OMNI_SERVING_ARGS)

    with create_isvc(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        runtime=vllm_omni_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=vllm_omni_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=vllm_omni_model_service_account.name,
        resources=resources,
        volumes=OMNI_VOLUMES,
        volumes_mounts=OMNI_VOLUME_MOUNTS,
        argument=dedupe_vllm_cli_args(arguments=serving_args),
        wait=False,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.mark.vllm_omni_nvidia_single_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, vllm_omni_serving_runtime, vllm_omni_unready_isvc",
    [
        pytest.param(
            {"name": "vllm-omni-probes-health"},
            {"model-dir": QWEN3_TTS_MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                "name": "vllm-omni-health-probe",
                "model_path": QWEN3_TTS_MODEL_PATH,
                "deployment_mode": KServeDeploymentType.STANDARD,
                "gpu_count": 1,
            },
            id="test_vllm_omni_qwen3_tts_health_transition",
        ),
    ],
    indirect=True,
)
class TestVllmOmniHealthEndpointTransition:
    """Validate that /health reflects initialization state and is stable post-ready.

    Steps:
        1. Deploy vLLM-Omni ISVC without waiting for pod readiness (vllm_omni_unready_isvc).
        2. Wait for the predictor pod to enter Running phase.
        3. Poll /health at 5-second intervals; assert at least one non-200 is observed
           while initialization is in progress.
        4. Assert /health transitions to HTTP 200 only after all three stages complete.
        5. Continue polling for 60 seconds; assert no flapping back to non-200.

    """

    def test_vllm_omni_health_endpoint_transitions(
        self,
        admin_client: DynamicClient,
        vllm_omni_unready_isvc: InferenceService,
    ) -> None:
        """Given a vLLM-Omni predictor pod that has just started,
        When /health is polled at 5-second intervals from the moment the
        container is Running,
        Then at least one non-200 response is observed while stages are loading,
        the endpoint transitions to HTTP 200 only after all stages finish,
        and /health remains stable at 200 for at least 60 seconds with no flapping.

        If the model loads faster than the first poll interval (5 s), the test
        may never observe a non-200 response and will fail the ``saw_non_200``
        assertion.  This is expected on clusters with pre-pulled images and very
        small models — the race is inherent to poll-based transition detection.
        """
        pod = wait_for_pod_running(admin_client=admin_client, isvc=vllm_omni_unready_isvc)

        saw_non_200: bool = False
        transitioned_to_200: bool = False

        try:
            for health_code in TimeoutSampler(
                wait_timeout=HEALTH_TRANSITION_TIMEOUT_SECONDS,
                sleep=HEALTH_POLL_INTERVAL_SECONDS,
                func=_safe_exec_health_probe,
                pod=pod,
            ):
                LOGGER.info(
                    event="health probe during initialization",
                    pod=pod.name,
                    status_code=health_code,
                )
                if health_code != "200":
                    saw_non_200 = True
                elif saw_non_200:
                    transitioned_to_200 = True
                    break
        except TimeoutExpiredError:
            pytest.fail(
                f"Pod '{pod.name}' /health did not transition to HTTP 200 within "
                f"{HEALTH_TRANSITION_TIMEOUT_SECONDS} s. saw_non_200={saw_non_200}"
            )

        assert saw_non_200, (
            f"Pod '{pod.name}' /health never returned a non-200 response before becoming "
            f"ready. Expected at least one non-200 during the three-stage initialization. "
            f"Verify the test started before all stages completed."
        )
        assert transitioned_to_200, (
            f"Pod '{pod.name}' /health did not transition from non-200 to HTTP 200 within "
            f"{HEALTH_TRANSITION_TIMEOUT_SECONDS} s"
        )

        # Stability check: /health must remain at 200 for HEALTH_STABILITY_WINDOW_SECONDS
        stability_start = time.monotonic()
        poll_count: int = 0
        while time.monotonic() - stability_start < HEALTH_STABILITY_WINDOW_SECONDS:
            stability_code = _safe_exec_health_probe(pod=pod)
            poll_count += 1
            elapsed = time.monotonic() - stability_start
            assert stability_code == "200", (
                f"Pod '{pod.name}' /health flapped to HTTP {stability_code} after transition "
                f"to 200 (poll {poll_count}, {elapsed:.0f} s into {HEALTH_STABILITY_WINDOW_SECONDS} s "
                f"stability window)"
            )
            time.sleep(HEALTH_POLL_INTERVAL_SECONDS)

        LOGGER.info(
            event="health endpoint stability confirmed",
            pod=pod.name,
            stability_seconds=HEALTH_STABILITY_WINDOW_SECONDS,
            total_polls=poll_count,
        )


_TEXT_ONLY_MODEL_DIR: str = "opt-125m"
FAILED_INIT_TIMEOUT_S: int = 300


@pytest.mark.vllm_omni_nvidia_single_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, vllm_omni_serving_runtime, vllm_omni_unready_isvc",
    [
        pytest.param(
            {"name": "vllm-omni-failed-init"},
            {"model-dir": _TEXT_ONLY_MODEL_DIR},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                "name": "vllm-omni-failed-init",
                "model_path": _TEXT_ONLY_MODEL_DIR,
                "deployment_mode": KServeDeploymentType.STANDARD,
                "gpu_count": 1,
            },
            id="test_vllm_omni_failed_stage_init",
        ),
    ],
    indirect=True,
)
class TestVllmOmniFailedStageInit:
    """Failed multi-stage init keeps pod NotReady and /health non-200.

    Deploys a text-only model (opt-125m) with the vLLM-Omni runtime and --omni flag.
    The engine downloads the model successfully but fails during multi-stage pipeline
    initialization because the model has no TTS/Code2Wav components. Validates that
    the pod never reaches Ready and /health correctly reflects the failure state.
    """

    def test_vllm_omni_failed_stage_init_stays_not_ready(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        vllm_omni_unready_isvc: InferenceService,
    ) -> None:
        """Pod remains NotReady and /health returns non-200 when a stage fails.

        Given a vLLM-Omni ISVC pointing to a text-only model (opt-125m, no TTS components),
        When the engine attempts multi-stage init with --omni,
        Then the pod enters CrashLoopBackOff or remains NotReady,
        /health never returns 200, and the ISVC status reflects the failure.
        """
        pod_appeared = False
        pod_crashed = False
        health_ever_200 = False

        deadline = time.monotonic() + FAILED_INIT_TIMEOUT_S

        while time.monotonic() < deadline:
            try:
                pods = get_pods_by_isvc_label(client=admin_client, isvc=vllm_omni_unready_isvc)
            except Exception:  # noqa: BLE001
                time.sleep(HEALTH_POLL_INTERVAL_SECONDS)
                continue

            if not pods:
                time.sleep(HEALTH_POLL_INTERVAL_SECONDS)
                continue

            pod = pods[0]
            pod_appeared = True

            container_statuses = pod.instance.status.containerStatuses or []
            kserve_cs = next((cs for cs in container_statuses if cs.name == "kserve-container"), None)

            if kserve_cs and kserve_cs.restartCount >= 1:
                pod_crashed = True
                LOGGER.info(
                    event="kserve-container crashed and restarted",
                    pod=pod.name,
                    restart_count=kserve_cs.restartCount,
                    reason=getattr(getattr(getattr(kserve_cs, "state", None), "waiting", None), "reason", "N/A"),
                )
                break

            if has_crash_loop_backoff(pod=pod):
                pod_crashed = True
                LOGGER.info(event="pod entered CrashLoopBackOff as expected", pod=pod.name)
                break

            health_code = _safe_exec_health_probe(pod=pod)
            if health_code == "200":
                health_ever_200 = True
                break

            if pod_is_ready(pod=pod):
                pytest.fail(f"Failed-stage init pod '{pod.name}' became Ready unexpectedly")
                break

            LOGGER.debug(
                event="failed-init: pod not ready, waiting for crash",
                pod=pod.name,
                phase=getattr(pod.instance.status, "phase", "Unknown"),
                restart_count=kserve_cs.restartCount if kserve_cs else 0,
            )
            time.sleep(HEALTH_POLL_INTERVAL_SECONDS)

        assert pod_appeared, (
            f"No predictor pod appeared for ISVC '{vllm_omni_unready_isvc.name}' within {FAILED_INIT_TIMEOUT_S}s"
        )
        assert not health_ever_200, (
            "Failed-stage init: /health returned 200 for a text-only model with --omni. "
            "Expected /health to remain non-200 when multi-stage init fails."
        )
        assert pod_crashed, (
            f"Failed-stage init: pod did not enter CrashLoopBackOff within {FAILED_INIT_TIMEOUT_S}s. "
            f"Expected the engine to crash when loading a text-only model with --omni."
        )

        health_code = _safe_exec_health_probe(pod=pods[0])
        assert health_code != "200", (
            f"Failed-stage init: /health returned {health_code} on a crashed pod. "
            f"Expected non-200 after failed multi-stage init."
        )

        LOGGER.info(
            event="failed-stage init validated",
            pod=pods[0].name,
            pod_crashed=pod_crashed,
            health_code=health_code,
        )
