import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.triton.constant import BASE_RAW_DEPLOYMENT_CONFIG, MODEL_PATH_PREFIX
from tests.model_serving.model_runtime.triton.probes.utils import (
    exec_triton_health_check,
    get_probe,
    resolve_http_get,
)
from tests.model_serving.model_runtime.utils import get_restart_counts, pod_is_ready
from utilities.constants import KServeDeploymentType

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, triton_probes_serving_runtime, triton_probes_inference_service",
    [
        pytest.param(
            {"name": "triton-onnx-probes"},
            {"model-dir": MODEL_PATH_PREFIX},
            {"deployment_type": KServeDeploymentType.STANDARD},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "name": "triton-onnx-probes",
                "deployment_mode": KServeDeploymentType.STANDARD,
            },
            id="test_triton_onnx_standard_rest_probes",
        ),
    ],
    indirect=True,
)
class TestTritonProbeHealth:
    """Validate Triton predictor readiness and liveness probes for S3-backed ONNX model.

    Given a Triton REST ServingRuntime with readiness/liveness probes and ONNX model from S3,
    When the InferenceService is deployed,
    Then the pod is Ready, probes define httpGet, and health endpoints return HTTP 200.
    """

    def test_triton_readiness_probe(
        self,
        triton_probes_inference_service: InferenceService,
        triton_probes_pod_resource: Pod,
    ) -> None:
        """Given a deployed Triton ONNX ISVC with probe-enabled REST runtime,
        When the predictor pod is inspected,
        Then the pod is Ready, readinessProbe defines httpGet, and the endpoint returns HTTP 200.
        """
        assert pod_is_ready(pod=triton_probes_pod_resource), f"Pod {triton_probes_pod_resource.name} is not Ready"

        readiness_probe = get_probe(pod=triton_probes_pod_resource, probe_type="readinessProbe")
        http_get = readiness_probe.get("httpGet")
        assert http_get, "readinessProbe must define httpGet"

        status_code = exec_triton_health_check(
            pod=triton_probes_pod_resource, http_get=resolve_http_get(probe=readiness_probe)
        )
        assert status_code == "200", (
            f"Readiness probe on {triton_probes_pod_resource.name} returned HTTP {status_code}, expected 200"
        )

    def test_triton_liveness_probe(
        self,
        triton_probes_inference_service: InferenceService,
        triton_probes_pod_resource: Pod,
    ) -> None:
        """Given a deployed Triton ONNX ISVC with probe-enabled REST runtime,
        When the predictor pod container status is checked,
        Then livenessProbe defines httpGet, no containers restarted, and the endpoint returns HTTP 200.
        """
        restart_counts = get_restart_counts(pod=triton_probes_pod_resource)
        restarted_containers = [name for name, count in restart_counts.items() if count > 0]
        assert not restarted_containers, (
            f"Containers {restarted_containers} restarted during startup; restart counts: {restart_counts}"
        )

        liveness_probe = get_probe(pod=triton_probes_pod_resource, probe_type="livenessProbe")
        http_get = liveness_probe.get("httpGet")
        assert http_get, "livenessProbe must define httpGet"

        status_code = exec_triton_health_check(
            pod=triton_probes_pod_resource, http_get=resolve_http_get(probe=liveness_probe)
        )
        assert status_code == "200", (
            f"Liveness probe on {triton_probes_pod_resource.name} returned HTTP {status_code}, expected 200"
        )
