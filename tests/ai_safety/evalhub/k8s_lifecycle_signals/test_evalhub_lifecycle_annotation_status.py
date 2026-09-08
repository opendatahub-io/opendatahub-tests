"""TC-ANN: Job annotation evaluation-status payload verification.

Covers RHAISTRAT-1923 — verifies that the trustyai.opendatahub.io/evaluation-status
annotation is present on batch Jobs, contains valid JSON with required fields, is
updated at each lifecycle transition, and stays within the 256 KB size limit.
"""

from datetime import datetime

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.k8s_lifecycle_signals.constants import (
    LIFECYCLE_PHASE_LABEL,
    LIFECYCLE_PHASE_RUNNING,
    LIFECYCLE_STATUS_ANNOTATION,
    LIFECYCLE_STATUS_ANNOTATION_MAX_BYTES,
)
from tests.ai_safety.evalhub.k8s_lifecycle_signals.utils import (
    get_job_annotation,
    parse_status_annotation,
    submit_evalhub_job_and_capture_runtime_job,
    wait_for_evaluation_job_name,
    wait_for_job_label,
    wait_for_success_phase_signals,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    submit_evalhub_job,
    wait_for_evalhub_job,
)


@pytest.mark.ai_safety
class TestAnnAnnotationStatus:
    """TC-ANN: trustyai.opendatahub.io/evaluation-status annotation verification.

    Verifies that the status annotation is present on evaluation batch Jobs, contains
    valid JSON with required fields, is updated at each lifecycle transition, and stays
    within the 256 KB Kubernetes limit. All tests share the session-scoped EvalHub
    deployment and lifecycle signals fixtures.

    Given the session-scoped EvalHub deployment with lifecycle signals fixtures ready
    (tenant namespace, Route, vLLM emulator, and RBAC),
    When an evaluation job is submitted through the EvalHub API in the tenant namespace
    and the trustyai.opendatahub.io/evaluation-status annotation is read from the
    runtime batch Job,
    Then the annotation is valid JSON with required fields (phase, timestamp,
    evaluationId, summaryMetrics), reflects lifecycle transitions, and remains below
    the 262144-byte Kubernetes annotation size limit.
    """

    @pytest.mark.tier1
    def test_ann_001_status_annotation_contains_valid_json(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given a successful evaluation has completed,
        when the evaluation-status annotation is read from the batch Job,
        then the annotation value is valid JSON (parseable without error)."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-ann-001",
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

        raw = get_job_annotation(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_STATUS_ANNOTATION,
        )

        assert raw is not None, f"Expected annotation {LIFECYCLE_STATUS_ANNOTATION} on Job {job_name!r} in {ns}"
        parsed = parse_status_annotation(annotation_value=raw)
        assert isinstance(parsed, dict), f"Expected JSON object, got {type(parsed)}"

    @pytest.mark.tier1
    def test_ann_002_status_annotation_contains_required_fields(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given a successful evaluation has completed,
        when the evaluation-status annotation is parsed,
        then it contains phase (str), timestamp (RFC 3339 UTC str), evaluationId (str),
        and summaryMetrics (object) as required fields."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-ann-002",
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

        raw = get_job_annotation(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_STATUS_ANNOTATION,
        )
        assert raw is not None, f"Missing annotation {LIFECYCLE_STATUS_ANNOTATION}"
        data = parse_status_annotation(annotation_value=raw)

        assert "phase" in data, f"Missing 'phase' field in annotation: {data}"
        assert isinstance(data["phase"], str) and data["phase"], "phase must be a non-empty string"

        assert "timestamp" in data, f"Missing 'timestamp' field in annotation: {data}"
        ts = data["timestamp"]
        assert isinstance(ts, str), f"timestamp must be a string, got {type(ts)}"
        try:
            datetime.fromisoformat(date_string=ts.replace(old="Z", new="+00:00"))
        except ValueError as exc:
            raise AssertionError(f"timestamp is not a valid RFC 3339 UTC value: {ts!r}") from exc

        assert "evaluationId" in data, f"Missing 'evaluationId' field in annotation: {data}"
        assert isinstance(data["evaluationId"], str) and data["evaluationId"], "evaluationId must be a non-empty string"

        assert "summaryMetrics" in data, f"Missing 'summaryMetrics' field in annotation: {data}"
        assert isinstance(data["summaryMetrics"], dict), (
            f"summaryMetrics must be an object, got {type(data['summaryMetrics'])}"
        )

    @pytest.mark.tier1
    def test_ann_003_status_annotation_updated_at_each_transition(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given an evaluation job is submitted,
        when the annotation is read while Running and again after Succeeded,
        then the phase field transitions from Running to Completed and
        the timestamp field reflects a more recent value after completion."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-ann-003",
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

        # Read annotation while job is Running
        wait_for_job_label(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_PHASE_LABEL,
            expected_value=LIFECYCLE_PHASE_RUNNING,
            timeout=60,
        )
        running_raw = get_job_annotation(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_STATUS_ANNOTATION,
        )
        assert running_raw is not None, "Missing annotation while job is Running"
        running_data = parse_status_annotation(annotation_value=running_raw)
        running_phase = running_data.get("phase", "")

        # Wait for completion and read again
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
        completed_raw = get_job_annotation(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_STATUS_ANNOTATION,
        )

        assert completed_raw is not None, "Missing annotation after job completion"
        completed_data = parse_status_annotation(annotation_value=completed_raw)

        assert completed_data.get("phase") in ("Completed", "Succeeded"), (
            f"Expected phase=Completed or Succeeded after job completion, got {completed_data.get('phase')!r}"
        )
        assert running_phase == "Running", f"Expected phase=Running during execution, got {running_phase!r}"
        if running_raw != completed_raw:
            running_ts = parse_status_annotation(annotation_value=running_raw).get("timestamp", "")
            completed_ts = completed_data.get("timestamp", "")
            assert completed_ts >= running_ts, (
                f"Completed timestamp {completed_ts!r} should be >= running timestamp {running_ts!r}"
            )

    @pytest.mark.tier2
    def test_ann_004_status_annotation_within_256kb_limit(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given a successful evaluation that produces summary metrics,
        when the evaluation-status annotation size is measured,
        then it is less than 262144 bytes (256 KB) and the Job was accepted by the API server."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-ann-004",
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

        raw = get_job_annotation(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_STATUS_ANNOTATION,
        )
        assert raw is not None, f"Missing annotation {LIFECYCLE_STATUS_ANNOTATION}"

        annotation_bytes = len(raw.encode("utf-8"))

        assert annotation_bytes < LIFECYCLE_STATUS_ANNOTATION_MAX_BYTES, (
            f"Annotation size {annotation_bytes} bytes exceeds 256 KB limit "
            f"({LIFECYCLE_STATUS_ANNOTATION_MAX_BYTES} bytes)"
        )
