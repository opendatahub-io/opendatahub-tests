import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.openvino.constant import BASE_RAW_DEPLOYMENT_CONFIG, MODEL_PATH_PREFIX
from tests.model_serving.model_runtime.openvino.probes.utils import (
    exec_ovms_health_check,
    get_probe,
    resolve_http_get,
)
from tests.model_serving.model_runtime.utils import get_restart_counts, pod_is_ready
from utilities.constants import KServeDeploymentType, ModelFormat

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, ovms_probes_serving_runtime, ovms_probes_inference_service",
    [
        pytest.param(
            {"name": "ovms-onnx-probes"},
            {"model-dir": f"{MODEL_PATH_PREFIX}/{ModelFormat.ONNX}"},
            {"deployment_type": KServeDeploymentType.STANDARD},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "name": "ovms-onnx-probes",
                "deployment_mode": KServeDeploymentType.STANDARD,
            },
            id="test_ovms_onnx_standard_rest_probes",
        ),
    ],
    indirect=True,
)
class TestOVMSProbeHealth:
    """Validate OVMS predictor readiness and liveness probes for S3-backed ONNX model.

    Steps:
        1. Deploy an OVMS ServingRuntime with readiness/liveness probes and ONNX model from S3.
        2. Verify pod Ready, readinessProbe httpGet, and health endpoint HTTP 200.
        3. Verify livenessProbe httpGet, no premature restarts, and health endpoint HTTP 200.
    """

    def test_ovms_readiness_probe(
        self,
        ovms_probes_inference_service: InferenceService,
        ovms_probes_pod_resource: Pod,
    ) -> None:
        """Given a deployed OVMS ONNX ISVC with probe-enabled runtime,
        When the predictor pod is inspected,
        Then the pod is Ready, readinessProbe defines httpGet, and the endpoint returns HTTP 200.
        """
        assert pod_is_ready(pod=ovms_probes_pod_resource), f"Pod {ovms_probes_pod_resource.name} is not Ready"

        readiness_probe = get_probe(pod=ovms_probes_pod_resource, probe_type="readinessProbe")
        http_get = readiness_probe.get("httpGet")
        assert http_get, "readinessProbe must define httpGet"

        status_code = exec_ovms_health_check(
            pod=ovms_probes_pod_resource, http_get=resolve_http_get(probe=readiness_probe)
        )
        assert status_code == "200", (
            f"Readiness probe on {ovms_probes_pod_resource.name} returned HTTP {status_code}, expected 200"
        )

    def test_ovms_liveness_probe(
        self,
        ovms_probes_inference_service: InferenceService,
        ovms_probes_pod_resource: Pod,
    ) -> None:
        """Given a deployed OVMS ONNX ISVC with probe-enabled runtime,
        When the predictor pod container status is checked,
        Then livenessProbe defines httpGet, no containers restarted, and the endpoint returns HTTP 200.
        """
        restart_counts = get_restart_counts(pod=ovms_probes_pod_resource)
        restarted_containers = [name for name, count in restart_counts.items() if count > 0]
        assert not restarted_containers, (
            f"Containers {restarted_containers} restarted during startup; restart counts: {restart_counts}"
        )

        liveness_probe = get_probe(pod=ovms_probes_pod_resource, probe_type="livenessProbe")
        http_get = liveness_probe.get("httpGet")
        assert http_get, "livenessProbe must define httpGet"

        status_code = exec_ovms_health_check(
            pod=ovms_probes_pod_resource, http_get=resolve_http_get(probe=liveness_probe)
        )
        assert status_code == "200", (
            f"Liveness probe on {ovms_probes_pod_resource.name} returned HTTP {status_code}, expected 200"
        )
