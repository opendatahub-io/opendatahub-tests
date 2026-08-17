"""TC-DUP: Dual-emission deduplication between EvalHub server and operator.

Covers RHAISTRAT-1923 — verifies that only one EvaluationFailed Event is emitted
per failure, that the operator checks the server-set label before emitting, and
that both emitters use consistent reason codes.
"""

import time

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.k8s_lifecycle_signals.constants import (
    LIFECYCLE_EVENT_EMISSION_TIMEOUT,
    LIFECYCLE_REASON_FAILED,
    LIFECYCLE_SOURCE_OPERATOR,
    LIFECYCLE_SOURCE_SERVER,
)
from tests.ai_safety.evalhub.k8s_lifecycle_signals.utils import (
    build_bad_image_payload,
    build_nonexistent_adapter_payload,
    build_oom_job_payload,
    filter_events_by_source,
    list_events_for_job,
    submit_evalhub_job_and_capture_runtime_job,
    wait_for_event,
    wait_for_event_from_component,
)
from tests.ai_safety.evalhub.utils import wait_for_evalhub_job


@pytest.mark.ai_safety
class TestDupDualEmission:
    """TC-DUP: Deduplication between server-side and operator-side Event emission.

    Application failures are reported by the EvalHub server; the operator skips when the
    server-set Failed label is present. Infrastructure failures are POSTed to EvalHub by
    the operator, which triggers a server lifecycle Event plus an operator Event.
    """

    @pytest.mark.tier1
    def test_dup_001_operator_checks_server_label_before_emitting(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given an application-level failure detected by the EvalHub server (non-existent adapter),
        when the server sets evaluation-phase=Failed and emits EvaluationFailed,
        then the operator does NOT emit a duplicate EvaluationFailed Event
        because it sees the server-set label and skips emission."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_nonexistent_adapter_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-dup-001",
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

        wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
            timeout=60,
        )
        time.sleep(LIFECYCLE_EVENT_EMISSION_TIMEOUT)
        all_failed_events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
        )
        server_events = [
            e for e in all_failed_events if (e.get("source") or {}).get("component") == LIFECYCLE_SOURCE_SERVER
        ]
        operator_events = [
            e for e in all_failed_events if (e.get("source") or {}).get("component") != LIFECYCLE_SOURCE_SERVER
        ]

        assert server_events, "Expected at least one EvaluationFailed from evalhub-server"
        assert not operator_events, (
            f"Operator should not emit EvaluationFailed when server already set the Failed label; "
            f"found {len(operator_events)} operator event(s)"
        )

    @pytest.mark.tier1
    def test_dup_002_no_duplicate_events_for_infrastructure_failure(
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
        """Given an infrastructure failure (ImagePullBackOff) detected by the operator,
        when the operator POSTs failed status to EvalHub and emits EvaluationFailed,
        then one EvaluationFailed Event exists from each source (server sync + operator),
        with no duplicate Events from the same source.component."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_bad_image_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-dup-002",
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

        wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
            timeout=120,
        )
        all_failed_events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
        )
        operator_events = filter_events_by_source(events=all_failed_events, source_component=LIFECYCLE_SOURCE_OPERATOR)
        server_events = filter_events_by_source(events=all_failed_events, source_component=LIFECYCLE_SOURCE_SERVER)

        assert operator_events, "Expected at least one EvaluationFailed from operator for ImagePullBackOff"
        assert server_events, (
            "Expected one EvaluationFailed from evalhub-server after operator API sync; "
            f"found {len(server_events)} server event(s)"
        )
        assert len(operator_events) == 1, (
            f"Expected exactly one operator EvaluationFailed, found {len(operator_events)}"
        )
        assert len(server_events) == 1, f"Expected exactly one server EvaluationFailed, found {len(server_events)}"

    @pytest.mark.tier1
    def test_dup_003_consistent_reason_codes_between_server_and_operator(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given an application-level failure (server-emitted) and an infrastructure failure
        (operator POST + operator Event),
        when their Events are compared,
        then both use reason=EvaluationFailed and type=Warning,
        differing only in source.component and message content."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name

        app_fail_payload = build_nonexistent_adapter_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-dup-003-app-fail",
        )
        infra_fail_payload = build_oom_job_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-dup-003-infra-fail",
        )

        app_id, app_job_name = submit_evalhub_job_and_capture_runtime_job(
            admin_client=admin_client,
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=app_fail_payload,
        )
        infra_id, infra_job_name = submit_evalhub_job_and_capture_runtime_job(
            admin_client=admin_client,
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=infra_fail_payload,
        )
        wait_for_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            job_id=app_id,
        )
        wait_for_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            job_id=infra_id,
        )

        server_event = wait_for_event_from_component(
            admin_client=admin_client,
            job_name=app_job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
            source_component=LIFECYCLE_SOURCE_SERVER,
            timeout=60,
        )
        operator_event = wait_for_event_from_component(
            admin_client=admin_client,
            job_name=infra_job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
            source_component=LIFECYCLE_SOURCE_OPERATOR,
            timeout=120,
        )

        assert server_event.get("reason") == LIFECYCLE_REASON_FAILED, (
            f"Server event reason mismatch: {server_event.get('reason')!r}"
        )
        assert operator_event.get("reason") == LIFECYCLE_REASON_FAILED, (
            f"Operator event reason mismatch: {operator_event.get('reason')!r}"
        )
        assert server_event.get("type") == "Warning", (
            f"Server EvaluationFailed should be type=Warning, got {server_event.get('type')!r}"
        )
        assert operator_event.get("type") == "Warning", (
            f"Operator EvaluationFailed should be type=Warning, got {operator_event.get('type')!r}"
        )
        assert server_event.get("message"), "Server EvaluationFailed message must be non-empty"
        assert operator_event.get("message"), "Operator EvaluationFailed message must be non-empty"

        server_component = (server_event.get("source") or {}).get("component")
        operator_component = (operator_event.get("source") or {}).get("component")
        assert server_component == LIFECYCLE_SOURCE_SERVER, (
            f"Server event source.component should be {LIFECYCLE_SOURCE_SERVER!r}, got {server_component!r}"
        )
        assert operator_component == LIFECYCLE_SOURCE_OPERATOR, (
            f"Operator event source.component should be {LIFECYCLE_SOURCE_OPERATOR!r}, got {operator_component!r}"
        )
