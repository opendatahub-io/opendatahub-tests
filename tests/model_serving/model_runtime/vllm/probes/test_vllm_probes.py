import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.utils import get_restart_counts, pod_is_ready
from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG
from tests.model_serving.model_runtime.vllm.cpu.cpu_x86.constant import (
    CPU_X86_ENV_VARIABLES,
    CPU_X86_SERVING_ARGUMENT,
    OPT_125M_MODEL_PATH,
)
from tests.model_serving.model_runtime.vllm.probes.utils import (
    exec_vllm_health_check,
    get_probe,
    resolve_http_get,
)
from utilities.constants import KServeDeploymentType

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_cpu_x86_accelerator_type", "valid_aws_config")


@pytest.mark.smoke
@pytest.mark.vllm_cpu_x86
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, probes_serving_runtime, vllm_probes_inference_service",
    [
        pytest.param(
            {"name": "opt-125m-probes"},
            {"model-dir": OPT_125M_MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "name": "opt-125m-probes",
                "runtime_argument": CPU_X86_SERVING_ARGUMENT,
                "model_env_variables": CPU_X86_ENV_VARIABLES,
            },
            id="test_vllm_opt_125m_standard_cpu_probes",
        ),
    ],
    indirect=True,
)
class TestVllmProbeHealth:
    """Validate vLLM CPU predictor readiness and liveness probes for S3-backed OPT-125M.

    Steps:
        1. Deploy a vLLM CPU ServingRuntime with readiness/liveness probes and OPT-125M from S3.
        2. Verify pod Ready, readinessProbe httpGet, and health endpoint HTTP 200.
        3. Verify livenessProbe httpGet, no premature restarts, and health endpoint HTTP 200.
    """

    def test_vllm_readiness_probe(
        self,
        vllm_probes_inference_service: InferenceService,
        skip_if_not_probes_raw_deployment: None,
        vllm_probes_pod_resource: Pod,
    ) -> None:
        """Given a deployed vLLM OPT-125M ISVC with probe-enabled CPU runtime,
        When the predictor pod is inspected,
        Then the pod is Ready, readinessProbe defines httpGet, and the endpoint returns HTTP 200.
        """
        assert pod_is_ready(pod=vllm_probes_pod_resource), f"Pod {vllm_probes_pod_resource.name} is not Ready"

        readiness_probe = get_probe(pod=vllm_probes_pod_resource, probe_type="readinessProbe")
        http_get = readiness_probe.get("httpGet")
        assert http_get, "readinessProbe must define httpGet"

        status_code = exec_vllm_health_check(
            pod=vllm_probes_pod_resource, http_get=resolve_http_get(probe=readiness_probe)
        )
        assert status_code == "200", (
            f"Readiness probe on {vllm_probes_pod_resource.name} returned HTTP {status_code}, expected 200"
        )

    def test_vllm_liveness_probe(
        self,
        vllm_probes_inference_service: InferenceService,
        skip_if_not_probes_raw_deployment: None,
        vllm_probes_pod_resource: Pod,
    ) -> None:
        """Given a deployed vLLM OPT-125M ISVC with probe-enabled CPU runtime,
        When the predictor pod container status is checked,
        Then livenessProbe defines httpGet, no containers restarted, and the endpoint returns HTTP 200.
        """
        restart_counts = get_restart_counts(pod=vllm_probes_pod_resource)
        restarted_containers = [name for name, count in restart_counts.items() if count > 0]
        assert not restarted_containers, (
            f"Containers {restarted_containers} restarted during startup; restart counts: {restart_counts}"
        )

        liveness_probe = get_probe(pod=vllm_probes_pod_resource, probe_type="livenessProbe")
        http_get = liveness_probe.get("httpGet")
        assert http_get, "livenessProbe must define httpGet"

        status_code = exec_vllm_health_check(
            pod=vllm_probes_pod_resource, http_get=resolve_http_get(probe=liveness_probe)
        )
        assert status_code == "200", (
            f"Liveness probe on {vllm_probes_pod_resource.name} returned HTTP {status_code}, expected 200"
        )
