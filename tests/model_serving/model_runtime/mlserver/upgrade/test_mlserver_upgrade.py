"""MLServer CPU ISVC upgrade scenario tests.

Validates that a CPU ISVC using mlserver-runtime is functional before and after
RHOAI upgrade — inference passes and no additional container restarts occur.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_runtime.mlserver.constant import MODEL_CONFIGS
from tests.model_serving.model_runtime.mlserver.upgrade.constant import UPGRADE_RESTART_KEY
from tests.model_serving.model_runtime.mlserver.utils import (
    run_mlserver_inference,
    validate_deterministic_snapshot,
)
from tests.model_serving.model_runtime.utils import get_restart_counts
from utilities.constants import ModelFormat, Protocols, Timeout
from utilities.infra import get_pods_by_isvc_label

LOGGER = structlog.get_logger(name=__name__)


class TestMLServerCPUUpgrade:
    """Validates CPU MLServer ISVC survives RHOAI upgrade."""

    @pytest.mark.pre_upgrade
    @pytest.mark.tier1
    def test_pre_upgrade_inference(
        self,
        mlserver_upgrade_inference_service: InferenceService,
        mlserver_upgrade_baseline: ConfigMap,
    ) -> None:
        """Deploy CPU ISVC and validate inference before upgrade.

        Given OCP with RHOAI pre-upgrade
        When a CPU ISVC is deployed with sklearn model
        Then inference passes and baseline restart counts are recorded.
        """
        assert mlserver_upgrade_inference_service.exists, (
            f"ISVC '{mlserver_upgrade_inference_service.name}' not created "
            f"in '{mlserver_upgrade_inference_service.namespace}'"
        )
        assert mlserver_upgrade_baseline.exists, f"Baseline ConfigMap '{mlserver_upgrade_baseline.name}' not created"
        LOGGER.info("Pre-upgrade validation complete", isvc=mlserver_upgrade_inference_service.name)

    @pytest.mark.post_upgrade
    @pytest.mark.tier1
    def test_post_upgrade_inference(
        self,
        admin_client: DynamicClient,
        mlserver_upgrade_inference_service: InferenceService,
        mlserver_upgrade_baseline: ConfigMap,
    ) -> None:
        """Verify CPU MLServer ISVC recovers to Ready after RHOAI upgrade.

        Given a CPU ISVC deployed before upgrade
        When the upgrade completes
        Then the ISVC returns to Ready, inference passes, and no additional restarts occurred.
        """
        sklearn_config: dict[str, Any] = MODEL_CONFIGS[ModelFormat.SKLEARN]

        isvc = mlserver_upgrade_inference_service
        assert isvc.exists, f"ISVC '{isvc.name}' not found in '{isvc.namespace}' after upgrade"

        isvc_ready: bool = False
        for isvc_status in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_5MIN,
            sleep=15,
            func=lambda: isvc.instance.status,
        ):
            if not isvc_status:
                continue
            conditions = getattr(isvc_status, "conditions", None) or []
            for condition in conditions:
                if condition.type == "Ready" and condition.status == "True":
                    isvc_ready = True
                    break
            if isvc_ready:
                break

        assert isvc_ready, f"ISVC '{isvc.name}' did not reach Ready within {Timeout.TIMEOUT_5MIN}s"

        predictor_pods = get_pods_by_isvc_label(client=admin_client, isvc=isvc)
        assert predictor_pods, f"No predictor pods found for ISVC '{isvc.name}' after upgrade"

        post_response = run_mlserver_inference(
            isvc=isvc,
            input_data=sklearn_config["rest_query"],
            model_version=sklearn_config["model_version"],
            protocol=Protocols.REST,
        )
        validate_deterministic_snapshot(response=post_response)

        baseline_restart_counts: dict[str, int] = json.loads(
            mlserver_upgrade_baseline.instance.data[UPGRADE_RESTART_KEY]
        )
        current_restart_counts: dict[str, int] = get_restart_counts(pod=predictor_pods[0])

        additional_restarts: dict[str, int] = {
            container: current_restart_counts.get(container, 0) - baseline_restart_counts.get(container, 0)
            for container in set(list(baseline_restart_counts) + list(current_restart_counts))
            if current_restart_counts.get(container, 0) > baseline_restart_counts.get(container, 0)
        }
        assert not additional_restarts, f"Predictor pod had additional restarts after upgrade: {additional_restarts}"
