"""MLServer readiness and liveness probe validation."""

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.mlserver.constant import BASE_RAW_DEPLOYMENT_CONFIG
from tests.model_serving.model_runtime.mlserver.probes.utils import (
    exec_mlserver_health_check,
    get_probe,
    get_restart_counts,
    pod_is_ready,
    resolve_http_get,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.tier2
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, mlserver_probes_serving_runtime, mlserver_probes_inference_service",
    [
        pytest.param(
            {"name": "mlserver-probes"},
            {"model-dir": "mlserver/model_repository/sklearn"},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                "name": "mlserver-probes-sklearn",
                **BASE_RAW_DEPLOYMENT_CONFIG,
            },
            id="mlserver-sklearn-probes-standard",
        ),
    ],
    indirect=True,
)
class TestMLServerProbeHealth:
    """Validate MLServer predictor readiness and liveness probes for S3-backed sklearn model."""

    def test_mlserver_readiness_probe(
        self,
        mlserver_probes_inference_service: InferenceService,
        mlserver_probes_pod_resource: Pod,
    ) -> None:
        """Verify readinessProbe is configured and health endpoint returns HTTP 200."""
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
        """Verify livenessProbe is configured, no restarts occurred, and health endpoint returns HTTP 200."""
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
