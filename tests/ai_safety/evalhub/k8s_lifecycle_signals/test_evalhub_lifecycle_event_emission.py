"""TC-EVT: Kubernetes Event emission from the EvalHub server.

Covers RHAISTRAT-1923 / RHAIRFE-2042 — verifies that EvalHub emits correctly-structured
Kubernetes Events for all four evaluation lifecycle transitions.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.k8s_lifecycle_signals.constants import (
    LIFECYCLE_EXPECTED_REASONS,
    LIFECYCLE_PHASE_LABEL,
    LIFECYCLE_REASON_COMPLETED,
    LIFECYCLE_REASON_FAILED,
    LIFECYCLE_REASON_STARTED,
    LIFECYCLE_REASON_THRESHOLD_VIOLATED,
    LIFECYCLE_SOURCE_SERVER,
)
from tests.ai_safety.evalhub.k8s_lifecycle_signals.utils import (
    build_nonexistent_adapter_payload,
    build_threshold_violation_payload,
    get_batch_job_uid,
    is_valid_camel_case,
    list_events_for_job,
    read_job_label,
    submit_evalhub_job_and_capture_runtime_job,
    wait_for_evaluation_job_name,
    wait_for_event,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    submit_evalhub_job,
    wait_for_evalhub_job,
)


@pytest.mark.ai_safety
class TestEvtEventEmission:
    """TC-EVT: Kubernetes Event emission from the EvalHub server.

    Verifies that the EvalHub server emits Kubernetes Events with correct type,
    reason codes, source, and involvedObject for all four lifecycle transitions.
    All tests share the session-scoped EvalHub in the lifecycle signals namespace.
    """

    @pytest.mark.smoke
    def test_evt_001_evaluation_started_event_emitted(
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
        when a valid evaluation request is submitted,
        then exactly one EvaluationStarted Event with type=Normal and
        source.component=evalhub-server is emitted for the created Job."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-evt-001",
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

        event = wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_STARTED,
        )
        events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_STARTED,
        )

        assert len(events) == 1, f"Expected 1 EvaluationStarted event, got {len(events)}"
        assert event.get("type") == "Normal", f"Expected type=Normal, got {event.get('type')}"
        assert (event.get("involvedObject") or {}).get("kind") == "Job"
        assert (event.get("source") or {}).get("component") == LIFECYCLE_SOURCE_SERVER, (
            f"Expected source.component={LIFECYCLE_SOURCE_SERVER}, got {event.get('source')}"
        )
        assert event.get("message"), "Expected non-empty message on EvaluationStarted event"

    @pytest.mark.smoke
    def test_evt_002_evaluation_completed_event_emitted(
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
        when an evaluation completes all benchmarks successfully,
        then exactly one EvaluationCompleted Event with type=Normal and
        source.component=evalhub-server is emitted for the Job."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-evt-002",
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

        event = wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_COMPLETED,
        )
        events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_COMPLETED,
        )

        assert len(events) == 1, f"Expected 1 EvaluationCompleted event, got {len(events)}"
        assert event.get("type") == "Normal", f"Expected type=Normal, got {event.get('type')}"
        assert (event.get("involvedObject") or {}).get("kind") == "Job"
        assert (event.get("source") or {}).get("component") == LIFECYCLE_SOURCE_SERVER
        assert event.get("message"), "Expected non-empty message on EvaluationCompleted event"

    @pytest.mark.smoke
    def test_evt_003_evaluation_failed_event_emitted_on_adapter_error(
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
        when an evaluation is submitted with a non-existent adapter name,
        then at least one EvaluationFailed Event with type=Warning and
        source.component=evalhub-server is emitted for the Job."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_nonexistent_adapter_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-evt-003",
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

        event = wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
        )
        events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
            source_component=LIFECYCLE_SOURCE_SERVER,
        )

        assert len(events) >= 1, f"Expected at least 1 EvaluationFailed event from server, got {len(events)}"
        assert event.get("type") == "Warning", f"Expected type=Warning, got {event.get('type')}"
        assert (event.get("source") or {}).get("component") == LIFECYCLE_SOURCE_SERVER
        assert event.get("message"), "Expected non-empty message describing adapter error"

    @pytest.mark.smoke
    def test_evt_004_evaluation_threshold_violated_event_emitted(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given a deployed EvalHub and a threshold of accuracy >= 1.01 (always fails),
        when an evaluation is submitted with this threshold,
        then exactly one EvaluationThresholdViolated Event with type=Warning and
        source.component=evalhub-server is emitted for the Job."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_threshold_violation_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-evt-004",
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

        event = wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_THRESHOLD_VIOLATED,
        )
        events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_THRESHOLD_VIOLATED,
        )

        assert len(events) == 1, f"Expected 1 EvaluationThresholdViolated event, got {len(events)}"
        assert event.get("type") == "Warning", f"Expected type=Warning, got {event.get('type')}"
        assert (event.get("source") or {}).get("component") == LIFECYCLE_SOURCE_SERVER
        assert event.get("message"), "Expected non-empty message referencing threshold violation"

    @pytest.mark.tier1
    def test_evt_005_event_type_field_correctness_across_transitions(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given all four lifecycle transitions have been triggered,
        when the type field of each resulting Event is inspected,
        then EvaluationStarted and EvaluationCompleted have type=Normal and
        EvaluationFailed and EvaluationThresholdViolated have type=Warning."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name

        success_payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-evt-005-success",
        )
        fail_payload = build_nonexistent_adapter_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-evt-005-fail",
        )
        threshold_payload = build_threshold_violation_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-evt-005-threshold",
        )

        # Collect job names for the three submitted jobs before API terminal wait
        success_job_id, success_job = submit_evalhub_job_and_capture_runtime_job(
            admin_client=admin_client,
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=success_payload,
        )
        fail_job_id, fail_job = submit_evalhub_job_and_capture_runtime_job(
            admin_client=admin_client,
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=fail_payload,
        )
        threshold_job_id, threshold_job = submit_evalhub_job_and_capture_runtime_job(
            admin_client=admin_client,
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=threshold_payload,
        )
        for job_id in (success_job_id, fail_job_id, threshold_job_id):
            wait_for_evalhub_job(
                host=host,
                token=lifecycle_signals_token,
                ca_bundle_file=lifecycle_signals_ca_bundle_file,
                tenant=ns,
                job_id=job_id,
            )

        for job_name, reason, expected_type in [
            (success_job, LIFECYCLE_REASON_STARTED, "Normal"),
            (success_job, LIFECYCLE_REASON_COMPLETED, "Normal"),
            (fail_job, LIFECYCLE_REASON_FAILED, "Warning"),
            (threshold_job, LIFECYCLE_REASON_THRESHOLD_VIOLATED, "Warning"),
        ]:
            wait_for_event(
                admin_client=admin_client,
                job_name=job_name,
                namespace=ns,
                reason=reason,
            )
            events = list_events_for_job(
                admin_client=admin_client,
                job_name=job_name,
                namespace=ns,
                reason=reason,
            )
            assert events, f"Expected at least one event for reason={reason} on job {job_name}"
            for event in events:
                assert event.get("type") == expected_type, (
                    f"Reason={reason} should have type={expected_type}, got {event.get('type')}"
                )

    @pytest.mark.tier1
    def test_evt_006_event_reason_codes_are_camel_case(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given a completed and a failed evaluation have both run,
        when all Events from evalhub-server are collected,
        then the set of unique reason codes is exactly the expected four CamelCase values."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name

        payloads = [
            build_evalhub_job_payload(
                model_service_name=lifecycle_signals_vllm_service.name,
                tenant_namespace=ns,
                job_name="tc-evt-006-success",
            ),
            build_nonexistent_adapter_payload(
                vllm_service_name=lifecycle_signals_vllm_service.name,
                namespace=ns,
                job_name="tc-evt-006-fail",
            ),
            build_threshold_violation_payload(
                vllm_service_name=lifecycle_signals_vllm_service.name,
                namespace=ns,
                job_name="tc-evt-006-threshold",
            ),
        ]
        job_names: list[str] = []
        for payload in payloads:
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
            job_names.append(job_name)

        all_events: list[dict] = []
        for job_name in job_names:
            all_events.extend(
                list_events_for_job(
                    admin_client=admin_client,
                    job_name=job_name,
                    namespace=ns,
                )
            )
        server_reasons = {
            e.get("reason") for e in all_events if (e.get("source") or {}).get("component") == LIFECYCLE_SOURCE_SERVER
        }

        unexpected = server_reasons - LIFECYCLE_EXPECTED_REASONS
        assert not unexpected, f"Unexpected reason codes from evalhub-server: {unexpected}"

        missing = LIFECYCLE_EXPECTED_REASONS - server_reasons
        assert not missing, f"Missing expected reason codes from evalhub-server: {missing}"

        for reason in server_reasons:
            assert is_valid_camel_case(reason), f"Reason {reason!r} is not CamelCase"

    @pytest.mark.tier1
    def test_evt_007_event_source_component_identifies_server(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given a successful evaluation has run,
        when all Events for the resulting Job are retrieved,
        then every Event emitted by the EvalHub server has
        source.component=evalhub-server and no other source.component value."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-evt-007",
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

        events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
        )
        server_events = [e for e in events if (e.get("source") or {}).get("component") == LIFECYCLE_SOURCE_SERVER]

        assert server_events, f"Expected at least one event from {LIFECYCLE_SOURCE_SERVER}"
        for event in server_events:
            component = (event.get("source") or {}).get("component")
            assert component == LIFECYCLE_SOURCE_SERVER, (
                f"Expected source.component={LIFECYCLE_SOURCE_SERVER}, got {component}"
            )

    @pytest.mark.tier1
    def test_evt_008_event_involved_object_references_correct_job(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given a successful evaluation has run,
        when the involvedObject fields of the EvaluationStarted Event are inspected,
        then kind=Job, apiVersion=batch/v1, name and namespace match the actual batch Job,
        and uid matches the batch Job's uid."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-evt-008",
        )
        _, job_name = submit_evalhub_job_and_capture_runtime_job(
            admin_client=admin_client,
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=payload,
        )
        wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_STARTED,
        )

        job_uid = get_batch_job_uid(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
        )
        assert job_uid, f"Expected batch Job {job_name} to exist in {ns}"

        event = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_STARTED,
        )[0]
        involved = event.get("involvedObject") or {}

        assert involved.get("kind") == "Job", f"Expected involvedObject.kind=Job, got {involved.get('kind')}"
        assert involved.get("apiVersion") in ("batch/v1", "batch"), (
            f"Expected involvedObject.apiVersion=batch/v1, got {involved.get('apiVersion')}"
        )
        assert involved.get("name") == job_name, f"Expected involvedObject.name={job_name}, got {involved.get('name')}"
        assert involved.get("namespace") == ns, (
            f"Expected involvedObject.namespace={ns}, got {involved.get('namespace')}"
        )
        assert involved.get("uid") == job_uid, f"Expected involvedObject.uid={job_uid}, got {involved.get('uid')}"

        # Confirm the lifecycle phase label is present (cross-checks with TC-LBL)
        phase_label = read_job_label(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_PHASE_LABEL,
        )
        assert phase_label is not None, (
            f"Expected label key {LIFECYCLE_PHASE_LABEL} on Job {job_name}, got {phase_label!r}"
        )
