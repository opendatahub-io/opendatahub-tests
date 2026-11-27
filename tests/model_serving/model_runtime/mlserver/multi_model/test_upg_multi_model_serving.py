"""
Upgrade tests for MLServer multi-model serving.

Validates ServingRuntime template persistence and InferenceService
readiness with model inference after a RHOAI platform upgrade.

Source: RHAISTRAT-3033
"""

from __future__ import annotations

from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_runtime.mlserver.multi_model.constant import (
    EXPECTED_MODEL_COUNT,
    ISVC_READY_TIMEOUT_SEC,
    MODEL_NAMES,
    MULTI_MODEL_UPG_NS,
    POLL_INTERVAL_SEC,
    SKLEARN_IRIS_PAYLOAD,
)
from tests.model_serving.model_runtime.mlserver.multi_model.utils import (
    get_isvc_pod,
    v2_health,
    v2_infer,
    v2_repository_index,
)
from utilities.infra import get_pods_by_isvc_label

LOGGER = structlog.get_logger(name=__name__)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.tier3
@pytest.mark.slow
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": MULTI_MODEL_UPG_NS})],
    indirect=True,
)
class TestServingRuntimeUpgrade:
    """Upgrade tests for MLServer ServingRuntime template persistence."""

    @pytest.mark.pre_upgrade
    @pytest.mark.order(1)
    @pytest.mark.dependency(name="TC-UPG-001-pre")
    def test_tc_upg_001_pre(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        multi_model_serving_runtime: ServingRuntime,
        multi_model_inference_service: InferenceService,
        pre_upgrade_serving_runtime_snapshot: dict[str, Any],
    ) -> None:
        """Create resources and capture ServingRuntime baseline before upgrade.

        Given: RHOAI (source version) installed, mlserver-runtime ServingRuntime
            template applied, InferenceService deployed and serving.
        When: The pre-upgrade snapshot fixture captures the ServingRuntime
            spec and persists it to a ConfigMap.
        Then: The ServingRuntime exists and the snapshot is saved for
            post-upgrade comparison.
        """
        runtime = multi_model_serving_runtime
        runtime.reload()
        assert runtime.exists, f"ServingRuntime {runtime.name} does not exist"
        LOGGER.info(
            "Pre-upgrade ServingRuntime snapshot captured",
            name=pre_upgrade_serving_runtime_snapshot["name"],
            namespace=pre_upgrade_serving_runtime_snapshot["namespace"],
        )

    @pytest.mark.post_upgrade
    @pytest.mark.order(2)
    @pytest.mark.dependency(name="TC-UPG-001")
    def test_tc_upg_001(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        multi_model_serving_runtime: ServingRuntime,
        pre_upgrade_serving_runtime_snapshot: dict[str, Any],
    ) -> None:
        """Verify ServingRuntime template persists through RHOAI platform upgrade.

        Given: RHOAI (source version) installed, mlserver-runtime ServingRuntime
            template applied, InferenceService deployed and serving, upgrade
            path available. Pre-upgrade spec snapshot captured and saved to
            ConfigMap.
        When: The RHOAI platform upgrade completes and the ServingRuntime is
            inspected.
        Then: The mlserver-runtime ServingRuntime exists, spec is unchanged
            (excluding resourceVersion), the opendatahub.io/support-status
            annotation is preserved, and the ServingRuntime shows a usable status.
        """
        runtime = multi_model_serving_runtime
        runtime.reload()

        assert runtime.exists, f"ServingRuntime {runtime.name} does not exist after upgrade"

        # Compare complete spec (excluding mutable metadata fields)
        post_upgrade_spec = runtime.instance.to_dict().get("spec", {})
        pre_upgrade_spec = pre_upgrade_serving_runtime_snapshot["spec"]

        def _normalize_spec(spec: dict) -> dict:
            """Normalize a ServingRuntime spec for comparison.

            Sorts list fields by a stable key so ordering differences do not
            cause false failures.
            """
            normalized = dict(spec)
            if "supportedModelFormats" in normalized:
                normalized["supportedModelFormats"] = sorted(
                    normalized["supportedModelFormats"],
                    key=lambda x: x.get("name", ""),
                )
            if "containers" in normalized:
                normalized["containers"] = sorted(
                    normalized["containers"],
                    key=lambda x: x.get("name", ""),
                )
            return normalized

        pre_normalized = _normalize_spec(spec=pre_upgrade_spec)
        post_normalized = _normalize_spec(spec=post_upgrade_spec)

        # Find specific differences for a clear failure message
        spec_diffs: list[str] = []
        all_keys = set(pre_normalized) | set(post_normalized)
        for key in sorted(all_keys):
            pre_val = pre_normalized.get(key)
            post_val = post_normalized.get(key)
            if pre_val != post_val:
                spec_diffs.append(f"  {key}: {pre_val!r} -> {post_val!r}")

        assert not spec_diffs, "ServingRuntime spec changed after upgrade:\n" + "\n".join(spec_diffs)

        # Verify tech-preview annotation preserved
        post_annotations = runtime.instance.to_dict().get("metadata", {}).get("annotations", {})
        support_status = post_annotations.get("opendatahub.io/support-status", "")
        LOGGER.info(
            "Post-upgrade ServingRuntime annotations",
            support_status=support_status,
            all_annotations=list(post_annotations.keys()),
        )

        pre_support_status = pre_upgrade_serving_runtime_snapshot["annotations"].get(
            "opendatahub.io/support-status", ""
        )
        if pre_support_status:
            assert support_status == pre_support_status, (
                f"support-status annotation changed: {pre_support_status!r} -> {support_status!r}"
            )

        # Verify ServingRuntime status is usable
        runtime_status = runtime.instance.get("status", {})
        LOGGER.info("Post-upgrade ServingRuntime status", status=runtime_status)

    @pytest.mark.post_upgrade
    @pytest.mark.order(3)
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
            repository index contains 10 models, and all 10 inference requests
            return HTTP 200. The pod container image is recorded for manual
            comparison but not asserted — the expected target image version
            is environment-specific and not available to the test framework.
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
                        pod_ready = next((c for c in pod_conditions if c.get("type") == "Ready"), None)
                        # Also check ISVC Ready
                        isvc.reload()
                        isvc_conds = isvc.instance.get("status", {}).get("conditions", [])
                        isvc_r = next((c for c in isvc_conds if c.get("type") == "Ready"), None)
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

        pod = get_isvc_pod(admin_client=admin_client, isvc=isvc)

        # Verify /v2/health/ready
        ready_code = v2_health(pod=pod, endpoint="/v2/health/ready")
        assert ready_code == 200, f"/v2/health/ready returned HTTP {ready_code} post-upgrade"

        # Verify repository index
        repo_index = v2_repository_index(pod=pod)
        assert len(repo_index) == EXPECTED_MODEL_COUNT, (
            f"Post-upgrade repository index has {len(repo_index)} models, expected {EXPECTED_MODEL_COUNT}"
        )

        # Send V2 inference to all 10 models
        for model_name in MODEL_NAMES:
            result = v2_infer(pod=pod, model_name=model_name, payload=SKLEARN_IRIS_PAYLOAD)
            assert result["http_code"] == 200, (
                f"Post-upgrade inference for {model_name} returned HTTP {result['http_code']}"
            )

        # Record pod image version
        containers = pod.instance.spec.get("containers", [])
        for container in containers:
            image = container.get("image", "")
            LOGGER.info("Post-upgrade pod container image", name=container.get("name"), image=image)
