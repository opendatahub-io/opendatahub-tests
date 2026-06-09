import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.vllm.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    GRANITE_SERVING_ARGUMENT,
)
from tests.model_serving.model_runtime.vllm.probes.utils import (
    exec_vllm_health_check,
    get_probe,
    get_restart_counts,
    pod_is_ready,
    resolve_http_get,
)
from utilities.constants import KServeDeploymentType

MODEL_PATH: str = "granite-7b-starter"

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.tier1
@pytest.mark.vllm_nvidia_single_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, probes_serving_runtime, vllm_probes_inference_service",
    [
        pytest.param(
            {"name": "granite-starter-probes"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "runtime_argument": GRANITE_SERVING_ARGUMENT,
                "gpu_count": 1,
                "name": "granite-starter-probes",
            },
            id="test_vllm_granite_raw_single_gpu_probes",
        ),
    ],
    indirect=True,
)
class TestVllmProbeHealth:
    """Validate vLLM predictor readiness and liveness probes for S3-backed Granite.

    Steps:
        1. Deploy a vLLM ServingRuntime with readiness/liveness probes and Granite from S3.
        2. Verify pod Ready, readinessProbe httpGet, and health endpoint HTTP 200.
        3. Verify livenessProbe httpGet, no premature restarts, and health endpoint HTTP 200.
    """

    def test_vllm_readiness_probe(
        self,
        vllm_probes_inference_service: InferenceService,
        skip_if_not_probes_raw_deployment: None,
        vllm_probes_pod_resource: Pod,
    ) -> None:
        """Given a deployed vLLM Granite ISVC with probe-enabled runtime,
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
        """Given a deployed vLLM Granite ISVC with probe-enabled runtime,
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
