import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.mlserver.constant import BASE_RAW_DEPLOYMENT_CONFIG, MODEL_PATH_PREFIX
from tests.model_serving.model_runtime.mlserver.probes.utils import (
    exec_mlserver_health_check,
    get_probe,
    resolve_http_get,
)
from tests.model_serving.model_runtime.utils import get_restart_counts, pod_is_ready
from utilities.constants import KServeDeploymentType, ModelFormat

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, mlserver_probes_serving_runtime, mlserver_probes_inference_service",
    [
        pytest.param(
            {"name": "sklearn-probes"},
            {"model-dir": f"{MODEL_PATH_PREFIX}/{ModelFormat.SKLEARN}"},
            {"deployment_type": KServeDeploymentType.STANDARD},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "name": "sklearn-probes",
                "deployment_mode": KServeDeploymentType.STANDARD,
            },
            id="test_mlserver_sklearn_standard_rest_probes",
        ),
    ],
    indirect=True,
)
class TestMLServerProbeHealth:
    """Validate MLServer predictor readiness and liveness probes for S3-backed sklearn.

    Steps:
        1. Deploy an MLServer ServingRuntime with readiness/liveness probes and sklearn from S3.
        2. Verify pod Ready, readinessProbe httpGet, and health endpoint HTTP 200.
        3. Verify livenessProbe httpGet, no premature restarts, and health endpoint HTTP 200.
    """

    def test_mlserver_readiness_probe(
        self,
        mlserver_probes_inference_service: InferenceService,
        mlserver_probes_pod_resource: Pod,
    ) -> None:
        """Given a deployed MLServer sklearn ISVC with probe-enabled runtime,
        When the predictor pod is inspected,
        Then the pod is Ready, readinessProbe defines httpGet, and the endpoint returns HTTP 200.
        """
        assert pod_is_ready(pod=mlserver_probes_pod_resource), f"Pod {mlserver_probes_pod_resource.name} is not Ready"

        readiness_probe = get_probe(pod=mlserver_probes_pod_resource, probe_type="readinessProbe")
        http_get = readiness_probe.get("httpGet")
        assert http_get, "readinessProbe must define httpGet"

        status_code = exec_mlserver_health_check(
            pod=mlserver_probes_pod_resource, http_get=resolve_http_get(probe=readiness_probe)
        )
        assert status_code == "200", (
            f"Readiness probe on {mlserver_probes_pod_resource.name} returned HTTP {status_code}, expected 200"
        )

    def test_mlserver_liveness_probe(
        self,
        mlserver_probes_inference_service: InferenceService,
        mlserver_probes_pod_resource: Pod,
    ) -> None:
        """Given a deployed MLServer sklearn ISVC with probe-enabled runtime,
        When the predictor pod container status is checked,
        Then livenessProbe defines httpGet, no containers restarted, and the endpoint returns HTTP 200.
        """
        restart_counts = get_restart_counts(pod=mlserver_probes_pod_resource)
        restarted_containers = [name for name, count in restart_counts.items() if count > 0]
        assert not restarted_containers, (
            f"Containers {restarted_containers} restarted during startup; restart counts: {restart_counts}"
        )

        liveness_probe = get_probe(pod=mlserver_probes_pod_resource, probe_type="livenessProbe")
        http_get = liveness_probe.get("httpGet")
        assert http_get, "livenessProbe must define httpGet"

        status_code = exec_mlserver_health_check(
            pod=mlserver_probes_pod_resource, http_get=resolve_http_get(probe=liveness_probe)
        )
        assert status_code == "200", (
            f"Liveness probe on {mlserver_probes_pod_resource.name} returned HTTP {status_code}, expected 200"
        )
