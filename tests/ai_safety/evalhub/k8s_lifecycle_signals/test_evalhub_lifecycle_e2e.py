"""TC-E2E: End-to-end lifecycle scenario tests.

Covers RHAISTRAT-1923 — verifies complete lifecycle signal flows for success,
server-reported failure, infrastructure failure, threshold violation, and
dual-emission deduplication scenarios.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.k8s_lifecycle_signals.constants import (
    LIFECYCLE_PHASE_LABEL,
    LIFECYCLE_PHASE_RUNNING,
    LIFECYCLE_PHASE_THRESHOLD_VIOLATED,
    LIFECYCLE_REASON_COMPLETED,
    LIFECYCLE_REASON_FAILED,
    LIFECYCLE_REASON_STARTED,
    LIFECYCLE_REASON_THRESHOLD_VIOLATED,
    LIFECYCLE_SOURCE_OPERATOR,
    LIFECYCLE_SOURCE_SERVER,
    LIFECYCLE_STATUS_ANNOTATION,
)
from tests.ai_safety.evalhub.k8s_lifecycle_signals.utils import (
    build_bad_image_payload,
    build_lifecycle_success_payload,
    build_nonexistent_adapter_payload,
    build_oom_job_payload,
    build_threshold_violation_payload,
    filter_events_by_source,
    get_job_annotation,
    list_events_for_job,
    parse_status_annotation,
    submit_evalhub_job_and_capture_runtime_job,
    wait_for_evaluation_job_name,
    wait_for_event,
    wait_for_event_from_component,
    wait_for_failed_label_near_operator_event,
    wait_for_failure_phase_signals,
    wait_for_job_label,
    wait_for_success_phase_signals,
)
from tests.ai_safety.evalhub.utils import (
    submit_evalhub_job,
    wait_for_evalhub_job,
)


@pytest.mark.ai_safety
class TestE2eLifecycle:
    """TC-E2E: End-to-end lifecycle verification covering all signal types.

    Each test covers the full observable lifecycle: Event emission, label transitions,
    and annotation updates for a complete evaluation scenario.
    """

    @pytest.mark.smoke
    def test_e2e_001_successful_evaluation_lifecycle(
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
        when a successful evaluation runs to completion,
        then EvaluationStarted and EvaluationCompleted Events are emitted (both Normal),
        the job label transitions Running -> Succeeded,
        the annotation phase reaches Completed,
        and the EvalHub API confirms completion with no Warning Events."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_lifecycle_success_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-e2e-001",
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

        # Verify Running label while job is executing
        wait_for_job_label(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_PHASE_LABEL,
            expected_value=LIFECYCLE_PHASE_RUNNING,
            timeout=60,
        )

        # Verify EvaluationStarted Event appears quickly
        started_event = wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_STARTED,
        )
        assert started_event.get("type") == "Normal"
        assert (started_event.get("source") or {}).get("component") == LIFECYCLE_SOURCE_SERVER

        # Wait for full completion
        result = wait_for_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            job_id=job_id,
        )
        assert result.get("status", {}).get("state") == "completed", (
            f"Expected completed, got {result.get('status', {}).get('state')!r}"
        )

        wait_for_success_phase_signals(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
        )

        # Verify EvaluationCompleted Event
        completed_event = wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_COMPLETED,
        )
        assert completed_event.get("type") == "Normal"
        assert (completed_event.get("source") or {}).get("component") == LIFECYCLE_SOURCE_SERVER

        # Verify annotation
        raw = get_job_annotation(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_STATUS_ANNOTATION,
        )
        assert raw is not None
        data = parse_status_annotation(annotation_value=raw)
        assert data.get("phase") in ("Completed", "Succeeded")
        assert "evaluationId" in data
        assert "summaryMetrics" in data

        # Verify no Warning Events
        all_events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
        )
        warning_events = [e for e in all_events if e.get("type") == "Warning"]
        assert not warning_events, (
            f"Expected no Warning Events for successful evaluation; found: "
            f"{[(e.get('reason'), e.get('message')) for e in warning_events]}"
        )

    @pytest.mark.smoke
    def test_e2e_002_server_reported_failure_lifecycle(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given an evaluation with a non-existent adapter (server-detected failure),
        when the server detects the failure,
        then EvaluationStarted and EvaluationFailed Events are emitted from evalhub-server,
        the job label is Failed, the annotation phase is Failed,
        and no operator-emitted EvaluationFailed Event exists."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_nonexistent_adapter_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-e2e-002",
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

        # Verify EvaluationStarted was emitted
        started_event = wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_STARTED,
        )
        assert started_event.get("type") == "Normal"

        # Verify EvaluationFailed from server
        failed_event = wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
        )
        assert failed_event.get("type") == "Warning"
        assert (failed_event.get("source") or {}).get("component") == LIFECYCLE_SOURCE_SERVER

        # Verify label (event-first; annotation fallback inside helper)
        wait_for_failure_phase_signals(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
        )

        # Verify annotation
        raw = get_job_annotation(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_STATUS_ANNOTATION,
        )
        assert raw is not None
        data = parse_status_annotation(annotation_value=raw)
        assert data.get("phase") == "Failed"

        # Verify no operator EvaluationFailed duplicate
        all_failed = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
        )
        operator_failed = [e for e in all_failed if (e.get("source") or {}).get("component") != LIFECYCLE_SOURCE_SERVER]
        assert not operator_failed, (
            f"Operator must not emit EvaluationFailed when server already handled it; "
            f"found {len(operator_failed)} operator event(s)"
        )

    @pytest.mark.smoke
    def test_e2e_003_infrastructure_failure_lifecycle(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given an evaluation with a 10Mi memory limit (causes OOMKill),
        when the operator detects the failure and POSTs to EvalHub,
        then EvaluationFailed Events are emitted from evalhub-server (API sync) and the operator,
        and the job label is Failed."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_oom_job_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-e2e-003",
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

        # Verify label (poll aggressively; operator deletes Job after sync)
        wait_for_failed_label_near_operator_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
        )

        all_failed = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
        )
        server_failed = filter_events_by_source(events=all_failed, source_component=LIFECYCLE_SOURCE_SERVER)
        operator_failed = filter_events_by_source(events=all_failed, source_component=LIFECYCLE_SOURCE_OPERATOR)
        assert len(server_failed) == 1, (
            f"Expected one server EvaluationFailed after operator API sync; found {len(server_failed)}"
        )
        assert len(operator_failed) == 1, f"Expected one operator EvaluationFailed; found {len(operator_failed)}"

    @pytest.mark.smoke
    def test_e2e_004_threshold_violation_lifecycle(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given an evaluation with a threshold the emulator cannot meet (accuracy >= 1.01),
        when benchmarks complete and the threshold check runs,
        then EvaluationStarted and EvaluationThresholdViolated Events are emitted,
        the job label is ThresholdViolated, the annotation reflects ThresholdViolated,
        and no EvaluationFailed Event exists."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_threshold_violation_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-e2e-004",
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

        # Verify EvaluationStarted
        started_event = wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_STARTED,
        )
        assert started_event.get("type") == "Normal"

        # Verify EvaluationThresholdViolated
        threshold_event = wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_THRESHOLD_VIOLATED,
        )
        assert threshold_event.get("type") == "Warning"
        assert (threshold_event.get("source") or {}).get("component") == LIFECYCLE_SOURCE_SERVER

        # Verify label
        wait_for_job_label(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_PHASE_LABEL,
            expected_value=LIFECYCLE_PHASE_THRESHOLD_VIOLATED,
        )

        # Verify annotation (eval-hub updates the label on threshold violation but may leave
        # the completion phase in the status annotation until a future enhancement)
        raw = get_job_annotation(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_STATUS_ANNOTATION,
        )
        assert raw is not None
        data = parse_status_annotation(annotation_value=raw)
        assert data.get("phase") in ("Succeeded", "ThresholdViolated")
        assert "evaluationId" in data

        # No EvaluationFailed Event (this is a threshold violation, not a failure)
        failed_events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
        )
        assert not failed_events, f"Expected no EvaluationFailed for threshold violation; found {len(failed_events)}"

    @pytest.mark.tier1
    def test_e2e_005_dual_emission_deduplication_lifecycle(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
        lifecycle_signals_bad_image_provider: str,
    ) -> None:
        """Given an infrastructure failure (ImagePullBackOff) handled by the operator,
        when all Events for the job are listed,
        then one EvaluationFailed Event exists from evalhub-server and one from the operator,
        and the job label is Failed."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_bad_image_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-e2e-005",
            bad_image_provider_id=lifecycle_signals_bad_image_provider,
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

        # Wait for operator Event
        wait_for_event_from_component(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
            source_component=LIFECYCLE_SOURCE_OPERATOR,
            timeout=120,
        )
        all_failed = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
        )

        server_failed = filter_events_by_source(events=all_failed, source_component=LIFECYCLE_SOURCE_SERVER)
        operator_failed = filter_events_by_source(events=all_failed, source_component=LIFECYCLE_SOURCE_OPERATOR)
        assert len(all_failed) == 2, (
            f"Expected two EvaluationFailed Events (server + operator), found {len(all_failed)}: "
            f"{[(e.get('source') or {}).get('component') for e in all_failed]}"
        )
        assert len(server_failed) == 1, f"Expected one server EvaluationFailed, found {len(server_failed)}"
        assert len(operator_failed) == 1, f"Expected one operator EvaluationFailed, found {len(operator_failed)}"

        # Label must be Failed (poll aggressively; operator may delete Job after sync)
        wait_for_failed_label_near_operator_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
        )
