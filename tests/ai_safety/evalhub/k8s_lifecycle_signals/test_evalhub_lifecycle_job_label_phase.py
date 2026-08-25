"""TC-LBL: Kubernetes Job label lifecycle phase tracking.

Covers RHAISTRAT-1923 — verifies that evaluation batch Jobs carry the
trustyai.opendatahub.io/evaluation-phase label and that it transitions
through the correct values.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.k8s_lifecycle_signals.constants import (
    LIFECYCLE_PHASE_FAILED,
    LIFECYCLE_PHASE_LABEL,
    LIFECYCLE_PHASE_RUNNING,
    LIFECYCLE_PHASE_SUCCEEDED,
    LIFECYCLE_PHASE_THRESHOLD_VIOLATED,
)
from tests.ai_safety.evalhub.k8s_lifecycle_signals.utils import (
    assert_failure_phase_label,
    build_lifecycle_success_payload,
    build_nonexistent_adapter_payload,
    build_threshold_violation_payload,
    read_job_label,
    submit_evalhub_job_and_capture_runtime_job,
    wait_for_evaluation_job_name,
    wait_for_failure_phase_signals,
    wait_for_job_label,
    wait_for_success_phase_signals,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    submit_evalhub_job,
    wait_for_evalhub_job,
)


@pytest.mark.ai_safety
class TestLblJobLabelPhase:
    """TC-LBL: trustyai.opendatahub.io/evaluation-phase label lifecycle verification.

    Verifies that evaluation batch Jobs carry the correct phase label value at each
    lifecycle stage. All tests share the session-scoped EvalHub.
    """

    @pytest.mark.smoke
    def test_lbl_001_job_label_set_to_running_on_start(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given a deployed EvalHub,
        when an evaluation request is submitted and the adapter container starts,
        then the batch Job carries the label evaluation-phase=Running."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-lbl-001",
        )
        job_id = submit_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=payload,
        )["resource"]["id"]
        job_name = wait_for_evaluation_job_name(
            admin_client=admin_client,
            namespace=ns,
            evalhub_job_id=job_id,
        )

        label_value = wait_for_job_label(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_PHASE_LABEL,
            expected_value=LIFECYCLE_PHASE_RUNNING,
            timeout=60,
        )

        assert label_value == LIFECYCLE_PHASE_RUNNING, (
            f"Expected {LIFECYCLE_PHASE_LABEL}={LIFECYCLE_PHASE_RUNNING}, got {label_value!r}"
        )

    @pytest.mark.smoke
    def test_lbl_002_job_label_set_to_succeeded_on_completion(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given a deployed EvalHub and a known-good model and dataset,
        when the evaluation completes all benchmarks successfully,
        then the batch Job label evaluation-phase=Succeeded."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_lifecycle_success_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-lbl-002",
        )
        job_id = submit_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=payload,
        )["resource"]["id"]
        job_name = wait_for_evaluation_job_name(
            admin_client=admin_client,
            namespace=ns,
            evalhub_job_id=job_id,
        )
        wait_for_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            job_id=job_id,
        )

        wait_for_success_phase_signals(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
        )
        label_value = read_job_label(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_PHASE_LABEL,
        )

        assert label_value == LIFECYCLE_PHASE_SUCCEEDED, (
            f"Expected {LIFECYCLE_PHASE_LABEL}={LIFECYCLE_PHASE_SUCCEEDED}, got {label_value!r}"
        )

    @pytest.mark.smoke
    def test_lbl_003_job_label_set_to_failed_on_failure(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given an evaluation submitted with a non-existent adapter (server-detected failure),
        when the server detects the failure,
        then the batch Job label evaluation-phase=Failed."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_nonexistent_adapter_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-lbl-003",
        )
        job_id, job_name = submit_evalhub_job_and_capture_runtime_job(
            admin_client=admin_client,
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=payload,
        )
        wait_for_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            job_id=job_id,
        )

        wait_for_failure_phase_signals(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
        )
        assert_failure_phase_label(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
        )

    @pytest.mark.smoke
    def test_lbl_004_job_label_set_to_threshold_violated(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given an evaluation submitted with a threshold the emulator cannot meet,
        when the threshold check runs after benchmark completion,
        then the batch Job label evaluation-phase=ThresholdViolated."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_threshold_violation_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-lbl-004",
        )
        job_id, job_name = submit_evalhub_job_and_capture_runtime_job(
            admin_client=admin_client,
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=payload,
        )
        wait_for_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            job_id=job_id,
        )

        label_value = wait_for_job_label(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_PHASE_LABEL,
            expected_value=LIFECYCLE_PHASE_THRESHOLD_VIOLATED,
        )

        assert label_value == LIFECYCLE_PHASE_THRESHOLD_VIOLATED, (
            f"Expected {LIFECYCLE_PHASE_LABEL}={LIFECYCLE_PHASE_THRESHOLD_VIOLATED}, got {label_value!r}"
        )

    @pytest.mark.tier1
    def test_lbl_005_job_label_queryable_via_label_selector(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given a successful evaluation and a failed evaluation have both completed,
        when batch Jobs are queried via kubectl label selector for each phase,
        then each selector returns only the expected Job and there is no cross-contamination."""
        from ocp_resources.job import Job

        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name

        success_payload = build_lifecycle_success_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-lbl-005-success",
        )
        fail_payload = build_nonexistent_adapter_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-lbl-005-fail",
        )

        success_id, success_job_name = submit_evalhub_job_and_capture_runtime_job(
            admin_client=admin_client,
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=success_payload,
        )
        fail_id, fail_job_name = submit_evalhub_job_and_capture_runtime_job(
            admin_client=admin_client,
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=fail_payload,
        )
        wait_for_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            job_id=success_id,
        )
        wait_for_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            job_id=fail_id,
        )

        wait_for_success_phase_signals(
            admin_client=admin_client,
            job_name=success_job_name,
            namespace=ns,
        )
        wait_for_failure_phase_signals(
            admin_client=admin_client,
            job_name=fail_job_name,
            namespace=ns,
        )

        success_label = read_job_label(
            admin_client=admin_client,
            job_name=success_job_name,
            namespace=ns,
            key=LIFECYCLE_PHASE_LABEL,
        )
        assert success_label == LIFECYCLE_PHASE_SUCCEEDED, (
            f"Expected success job label {LIFECYCLE_PHASE_SUCCEEDED!r}, got {success_label!r}"
        )
        assert_failure_phase_label(
            admin_client=admin_client,
            job_name=fail_job_name,
            namespace=ns,
        )

        succeeded_jobs = list(
            Job.get(
                client=admin_client,
                namespace=ns,
                label_selector=f"{LIFECYCLE_PHASE_LABEL}={LIFECYCLE_PHASE_SUCCEEDED}",
            )
        )
        failed_jobs = list(
            Job.get(
                client=admin_client,
                namespace=ns,
                label_selector=f"{LIFECYCLE_PHASE_LABEL}={LIFECYCLE_PHASE_FAILED}",
            )
        )

        succeeded_names = {j.name for j in succeeded_jobs}
        failed_names = {j.name for j in failed_jobs}

        assert success_job_name in succeeded_names, (
            f"Succeeded job {success_job_name!r} not found in Succeeded selector results"
        )
        assert fail_job_name not in succeeded_names, (
            f"Failed job {fail_job_name!r} incorrectly appeared in Succeeded selector results"
        )
        assert fail_job_name in failed_names, f"Failed job {fail_job_name!r} not found in Failed selector results"
        assert success_job_name not in failed_names, (
            f"Succeeded job {success_job_name!r} incorrectly appeared in Failed selector results"
        )
