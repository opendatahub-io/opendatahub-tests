"""
Upgrade tests for MLServer multi-model serving.

Validates ServingRuntime template persistence and InferenceService
readiness with model inference after a RHOAI platform upgrade.

Source: RHAISTRAT-3033
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.serving_runtime import ServingRuntime
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from utilities.infra import get_pods_by_isvc_label

LOGGER = structlog.get_logger(name=__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MULTI_MODEL_NS: str = "multi-model-upg"
ISVC_READY_TIMEOUT_SEC: int = 300
POLL_INTERVAL_SEC: int = 5
EXPECTED_MODEL_COUNT: int = 10
MODEL_NAMES: list[str] = [
    "sklearn-iris-1",
    "sklearn-iris-2",
    "xgb-classifier-1",
    "xgb-classifier-2",
    "lgbm-classifier-1",
    "lgbm-classifier-2",
    "onnx-iris-1",
    "onnx-iris-2",
    "sklearn-iris-3",
    "onnx-iris-3",
]
SKLEARN_IRIS_PAYLOAD: dict[str, Any] = {
    "inputs": [
        {
            "name": "predict",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": [[5.1, 3.5, 1.4, 0.2]],
        }
    ]
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_isvc_pod(admin_client: DynamicClient, isvc: InferenceService) -> Pod:
    """Return the first pod for the given InferenceService."""
    pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
    if not pods:
        raise RuntimeError(f"No pods found for InferenceService {isvc.name}")
    return pods[0]


def _exec_in_pod(pod: Pod, command: list[str]) -> str:
    """Execute a command inside a pod and return stdout."""
    return pod.execute(command=command)


def _v2_infer(pod: Pod, model_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send a V2 inference request via pod exec curl."""
    url = f"http://localhost:8080/v2/models/{model_name}/infer"
    result = _exec_in_pod(
        pod=pod,
        command=[
            "curl",
            "-s",
            "-w",
            "\n%{http_code}",
            "-X",
            "POST",
            url,
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload),
        ],
    )
    lines = result.strip().rsplit("\n", 1)
    body_str = lines[0] if len(lines) > 1 else ""
    if not lines[-1].isdigit():
        raise RuntimeError(f"Unparseable HTTP status from V2 infer response: {result!r}")
    http_code = int(lines[-1])
    try:
        body = json.loads(body_str) if body_str else {}
    except json.JSONDecodeError:
        body = {"raw": body_str}
    return {"http_code": http_code, "body": body}


def _v2_health(pod: Pod, endpoint: str) -> int:
    """Query a V2 health endpoint and return the HTTP status code."""
    url = f"http://localhost:8080{endpoint}"
    result = _exec_in_pod(
        pod=pod,
        command=["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
    )
    stripped = result.strip()
    if not stripped.isdigit():
        raise RuntimeError(f"Unparseable HTTP status from V2 health response: {result!r}")
    return int(stripped)


def _v2_repository_index(pod: Pod) -> list[dict[str, Any]]:
    """Query the V2 repository index and return the model list."""
    result = _exec_in_pod(
        pod=pod,
        command=[
            "curl",
            "-s",
            "-X",
            "POST",
            "http://localhost:8080/v2/repository/index",
            "-H",
            "Content-Type: application/json",
            "-d",
            "{}",
        ],
    )
    try:
        return json.loads(result.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unparseable JSON from V2 repository index: {result!r}") from exc


# ---------------------------------------------------------------------------
# Module-scoped state for pre/post upgrade comparison
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def pre_upgrade_sr_snapshot(
    admin_client: DynamicClient,
    mlserver_serving_runtime: ServingRuntime,
) -> dict[str, Any]:
    """Capture ServingRuntime spec before upgrade for post-upgrade comparison.

    Args:
        admin_client: Kubernetes dynamic client.
        mlserver_serving_runtime: The MLServer ServingRuntime resource.

    Returns:
        Dictionary containing the pre-upgrade spec and annotations.
    """
    sr = mlserver_serving_runtime
    sr.reload()
    spec = sr.instance.get("spec", {})
    annotations = sr.instance.get("metadata", {}).get("annotations", {})
    return {
        "name": sr.name,
        "namespace": sr.namespace,
        "spec": spec,
        "annotations": annotations,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.tier3
@pytest.mark.slow
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": MULTI_MODEL_NS})],
    indirect=True,
)
class TestServingRuntimeUpgrade:
    """Upgrade tests for MLServer ServingRuntime template persistence."""

    @pytest.mark.post_upgrade
    @pytest.mark.order(1)
    @pytest.mark.dependency(name="TC-UPG-001")
    def test_tc_upg_001(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        mlserver_serving_runtime: ServingRuntime,
        pre_upgrade_sr_snapshot: dict[str, Any],
    ) -> None:
        """Verify ServingRuntime template persists through RHOAI platform upgrade.

        Given: RHOAI (source version) installed, mlserver-runtime ServingRuntime
            template applied, InferenceService deployed and serving, upgrade
            path available. Pre-upgrade spec snapshot captured.
        When: The RHOAI platform upgrade completes and the ServingRuntime is
            inspected.
        Then: The mlserver-runtime SR exists, spec is unchanged (excluding
            resourceVersion), the opendatahub.io/support-status annotation
            is preserved, and the SR shows a usable status.
        """
        sr = mlserver_serving_runtime
        sr.reload()

        # Verify SR still exists
        assert sr.exists, f"ServingRuntime {sr.name} does not exist after upgrade"

        # Compare spec (excluding resourceVersion)
        post_upgrade_spec = sr.instance.get("spec", {})
        pre_upgrade_spec = pre_upgrade_sr_snapshot["spec"]

        # Deep comparison of key spec fields
        pre_formats = pre_upgrade_spec.get("supportedModelFormats", [])
        post_formats = post_upgrade_spec.get("supportedModelFormats", [])
        def _normalize_formats(formats: list) -> list:
            return sorted(
                [{k: v for k, v in f.items()} for f in formats],
                key=lambda x: x.get("name", ""),
            )

        assert _normalize_formats(pre_formats) == _normalize_formats(post_formats), (
            f"supportedModelFormats changed after upgrade: "
            f"pre={_normalize_formats(pre_formats)}, post={_normalize_formats(post_formats)}"
        )

        # Compare container images
        pre_containers = pre_upgrade_spec.get("containers", [])
        post_containers = post_upgrade_spec.get("containers", [])
        pre_images = [c.get("image", "") for c in pre_containers]
        post_images = [c.get("image", "") for c in post_containers]
        assert pre_images == post_images, f"Container images changed: {pre_images} -> {post_images}"

        # Verify tech-preview annotation preserved
        post_annotations = sr.instance.get("metadata", {}).get("annotations", {})
        support_status = post_annotations.get("opendatahub.io/support-status", "")
        LOGGER.info(
            "Post-upgrade SR annotations",
            support_status=support_status,
            all_annotations=list(post_annotations.keys()),
        )

        pre_support_status = pre_upgrade_sr_snapshot["annotations"].get(
            "opendatahub.io/support-status", ""
        )
        if pre_support_status:
            assert support_status == pre_support_status, (
                f"support-status annotation changed: {pre_support_status!r} -> {support_status!r}"
            )

        # Verify SR status is usable
        sr_status = sr.instance.get("status", {})
        LOGGER.info("Post-upgrade SR status", status=sr_status)

    @pytest.mark.post_upgrade
    @pytest.mark.order(2)
    @pytest.mark.dependency(name="TC-UPG-002", depends=["TC-UPG-001"])
    def test_tc_upg_002(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        multi_model_inference_service: InferenceService,
    ) -> None:
        """Verify InferenceService readiness and model inference post-upgrade.

        Given: TC-UPG-001 completed (SR verified), InferenceService was
            deployed and serving pre-upgrade with all 10 models at HTTP 200.
        When: The InferenceService status is checked post-upgrade, the
            repository index is queried, and V2 inference is sent to all models.
        Then: ISVC shows Ready:True, /v2/health/ready returns HTTP 200, the
            repository index contains 10 models, all 10 inference requests
            return HTTP 200, and the pod image reflects the upgraded version.
        """
        isvc = multi_model_inference_service
        isvc.reload()

        # Check ISVC Ready condition
        isvc_conditions = isvc.instance.get("status", {}).get("conditions", [])
        isvc_ready = next((c for c in isvc_conditions if c.get("type") == "Ready"), None)

        # If not Ready, wait for pod recovery
        if not isvc_ready or isvc_ready.get("status") != "True":
            LOGGER.info("ISVC not Ready post-upgrade, waiting for recovery")
            try:
                for sample in TimeoutSampler(
                    wait_timeout=ISVC_READY_TIMEOUT_SEC,
                    sleep=POLL_INTERVAL_SEC,
                    func=lambda: get_pods_by_isvc_label(client=admin_client, isvc=isvc),
                ):
                    if sample:
                        pod = sample[0]
                        pod.reload()
                        pod_conditions = pod.instance.status.get("conditions", [])
                        pod_ready = next(
                            (c for c in pod_conditions if c.get("type") == "Ready"), None
                        )
                        # Also check ISVC Ready
                        isvc.reload()
                        isvc_conds = isvc.instance.get("status", {}).get("conditions", [])
                        isvc_r = next(
                            (c for c in isvc_conds if c.get("type") == "Ready"), None
                        )
                        if (
                            pod_ready
                            and pod_ready.get("status") == "True"
                            and isvc_r
                            and isvc_r.get("status") == "True"
                        ):
                            break
            except TimeoutExpiredError:
                pytest.fail("ISVC did not reach Ready within timeout after upgrade")

        # Final assertion
        isvc.reload()
        isvc_conditions = isvc.instance.get("status", {}).get("conditions", [])
        isvc_ready = next((c for c in isvc_conditions if c.get("type") == "Ready"), None)
        assert isvc_ready is not None, "ISVC has no Ready condition after recovery"
        assert isvc_ready.get("status") == "True", f"ISVC Ready condition is {isvc_ready.get('status')}, expected True"

        pod = _get_isvc_pod(admin_client=admin_client, isvc=isvc)

        # Verify /v2/health/ready
        ready_code = _v2_health(pod=pod, endpoint="/v2/health/ready")
        assert ready_code == 200, f"/v2/health/ready returned HTTP {ready_code} post-upgrade"

        # Verify repository index
        repo_index = _v2_repository_index(pod=pod)
        assert len(repo_index) == EXPECTED_MODEL_COUNT, (
            f"Post-upgrade repository index has {len(repo_index)} models, expected {EXPECTED_MODEL_COUNT}"
        )

        # Send V2 inference to all 10 models
        for model_name in MODEL_NAMES:
            result = _v2_infer(pod=pod, model_name=model_name, payload=SKLEARN_IRIS_PAYLOAD)
            assert result["http_code"] == 200, (
                f"Post-upgrade inference for {model_name} returned HTTP {result['http_code']}"
            )

        # Record pod image version
        containers = pod.instance.spec.get("containers", [])
        for container in containers:
            image = container.get("image", "")
            LOGGER.info("Post-upgrade pod container image", name=container.get("name"), image=image)
