"""
Tests for neighbor ISVC isolation when one InferenceService fails to load.

A failing ISVC in the same namespace must not destabilize a healthy neighbor:
the healthy ISVC should continue serving HTTP 200, its pods should accumulate
zero new restarts, and the KServe control plane should remain Available.
"""

import json

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.kserve.negative.utils import (
    VALID_OVMS_INFERENCE_BODY,
    assert_kserve_control_plane_stable,
    send_inference_request,
    wait_for_isvc_model_status_states,
)
from utilities.infra import get_pods_by_isvc_label

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]

VALID_BODY_RAW: str = json.dumps(VALID_OVMS_INFERENCE_BODY)


@pytest.mark.tier3
class TestModelLoadIsolation:
    """A failed ISVC must not impact a healthy neighbor in the same namespace."""

    def test_failing_isvc_reaches_failed_to_load(
        self,
        neighbor_failing_ovms_isvc: InferenceService,
    ) -> None:
        """Given a failing ISVC with invalid S3 keys, it must report FailedToLoad.

        When:
            A second ISVC with intentionally wrong S3 credentials is deployed
            into the same namespace as a healthy ISVC.

        Then:
            ``status.modelStatus.states.targetModelState`` reaches ``FailedToLoad``
            with ``transitionStatus == BlockedByFailedLoad``.
        """
        wait_for_isvc_model_status_states(
            isvc=neighbor_failing_ovms_isvc,
            target_model_state="FailedToLoad",
            transition_status="BlockedByFailedLoad",
        )

    def test_healthy_neighbor_still_serves_after_failed_isvc(
        self,
        negative_test_ovms_isvc: InferenceService,
        neighbor_failing_ovms_isvc: InferenceService,
    ) -> None:
        """Given a failed neighbor ISVC, the healthy ISVC must still return HTTP 200.

        When:
            The failing ISVC has been deployed and reached ``FailedToLoad``.

        Then:
            An inference request to the healthy ISVC returns HTTP 200 with valid output.
        """
        wait_for_isvc_model_status_states(
            isvc=neighbor_failing_ovms_isvc,
            target_model_state="FailedToLoad",
            transition_status="BlockedByFailedLoad",
        )

        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=VALID_BODY_RAW,
        )
        assert status_code == 200, (
            f"Healthy neighbor ISVC returned {status_code} after failed ISVC deployed. Response: {response_body}"
        )
        parsed = json.loads(response_body)
        assert parsed.get("outputs"), f"Healthy neighbor ISVC returned invalid payload: {response_body}"

    def test_healthy_neighbor_pods_have_zero_new_restarts(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        neighbor_failing_ovms_isvc: InferenceService,
        healthy_isvc_pod_restart_baseline: dict[str, int],
    ) -> None:
        """Given a failed neighbor ISVC, the healthy ISVC pods must not restart.

        When:
            The failing ISVC has been deployed and reached ``FailedToLoad``.

        Then:
            The healthy ISVC pods show zero new restarts compared to baseline.
        """
        wait_for_isvc_model_status_states(
            isvc=neighbor_failing_ovms_isvc,
            target_model_state="FailedToLoad",
            transition_status="BlockedByFailedLoad",
        )

        pods = get_pods_by_isvc_label(client=admin_client, isvc=negative_test_ovms_isvc)
        current_uids = {pod.instance.metadata.uid for pod in pods}
        baseline_uids = set(healthy_isvc_pod_restart_baseline)
        assert current_uids == baseline_uids, (
            f"Healthy ISVC pod set changed. Baseline UIDs: {baseline_uids}, Current UIDs: {current_uids}"
        )
        for pod in pods:
            uid = pod.instance.metadata.uid
            current_total = 0
            for cs in pod.instance.status.containerStatuses or []:
                current_total += cs.restartCount
            for ics in pod.instance.status.initContainerStatuses or []:
                current_total += ics.restartCount

            baseline = healthy_isvc_pod_restart_baseline[uid]
            assert current_total == baseline, (
                f"Healthy ISVC pod {pod.name} restarted after failing neighbor deployed. "
                f"Baseline: {baseline}, Current: {current_total}"
            )

    def test_control_plane_stable_after_isolation_scenario(
        self,
        admin_client: DynamicClient,
        neighbor_failing_ovms_isvc: InferenceService,
        kserve_control_plane_restart_baseline: dict[str, int],
    ) -> None:
        """Given a failed neighbor ISVC, the control plane must remain stable.

        When:
            The failing ISVC has been deployed and reached ``FailedToLoad``.

        Then:
            ``kserve-controller-manager`` and ``odh-model-controller`` remain Available,
            show no CrashLoopBackOff, and do not accumulate new container restarts.
        """
        applications_namespace: str = py_config["applications_namespace"]

        wait_for_isvc_model_status_states(
            isvc=neighbor_failing_ovms_isvc,
            target_model_state="FailedToLoad",
            transition_status="BlockedByFailedLoad",
        )

        assert_kserve_control_plane_stable(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
            prior_restart_totals=kserve_control_plane_restart_baseline,
        )
