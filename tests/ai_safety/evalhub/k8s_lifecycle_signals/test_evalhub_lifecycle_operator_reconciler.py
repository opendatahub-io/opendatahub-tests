"""TC-OPR: TrustyAI Operator failure reconciler Event emission.

Covers RHAISTRAT-1923 — verifies that the TrustyAI Service Operator emits
EvaluationFailed Events for infrastructure-level failures that the EvalHub
server cannot detect on its own.
"""

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
    KUEUE_LOCAL_QUEUE_NAME,
    build_bad_image_payload,
    build_oom_job_payload,
    build_oom_kueue_job_payload,
    filter_events_by_source,
    is_kueue_installed,
    kueue_local_queue_exists,
    list_events_for_job,
    submit_evalhub_job_and_capture_runtime_job,
    wait_for_event,
    wait_for_failure_phase_signals,
)
from tests.ai_safety.evalhub.utils import (
    wait_for_evalhub_job,
    wait_for_evalhub_job_workload_admitted,
)


@pytest.mark.ai_safety
class TestOprOperatorReconciler:
    """TC-OPR: TrustyAI Operator failure reconciler emits EvaluationFailed Events.

    The operator monitors batch Jobs for infrastructure failures (OOM, ImagePullBackOff,
    Kueue eviction) and emits Events when the EvalHub server cannot self-report them.
    """

    @pytest.mark.smoke
    def test_opr_001_operator_emits_failed_event_for_oom(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given an evaluation job submitted with an artificially low memory limit (10Mi),
        when the adapter container is OOMKilled,
        then the TrustyAI Operator emits at least one EvaluationFailed Warning Event
        with a source.component that is not evalhub-server."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_oom_job_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-opr-001",
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
            source_component=LIFECYCLE_SOURCE_OPERATOR,
            event_timeout=LIFECYCLE_EVENT_EMISSION_TIMEOUT,
        )
        events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
        )
        operator_events = filter_events_by_source(events=events, source_component=LIFECYCLE_SOURCE_OPERATOR)
        assert operator_events, "Expected at least one EvaluationFailed Event from the operator (not evalhub-server)"
        event = operator_events[0]
        assert event.get("type") == "Warning", (
            f"Expected type=Warning for OOM EvaluationFailed, got {event.get('type')}"
        )
        operator_component = (event.get("source") or {}).get("component")
        assert operator_component, "Expected non-empty source.component on operator Event"
        assert operator_component != LIFECYCLE_SOURCE_SERVER, (
            f"Operator Event should not have source.component={LIFECYCLE_SOURCE_SERVER}"
        )
        assert event.get("message"), "Expected non-empty message referencing OOM failure"

    @pytest.mark.smoke
    def test_opr_002_operator_emits_failed_event_for_image_pull_failure(
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
        """Given an evaluation job submitted with a non-existent adapter image,
        when the pod enters ImagePullBackOff,
        then the TrustyAI Operator emits at least one EvaluationFailed Warning Event
        with a source.component that is not evalhub-server."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_bad_image_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-opr-002",
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

        event = wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
            timeout=120,
        )
        events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
        )
        operator_events = [e for e in events if (e.get("source") or {}).get("component") != LIFECYCLE_SOURCE_SERVER]

        assert operator_events, "Expected at least one EvaluationFailed Event from the operator for ImagePullBackOff"
        assert event.get("type") == "Warning", f"Expected type=Warning, got {event.get('type')}"
        assert operator_events[0].get("message"), "Expected message describing image pull failure"

    @pytest.mark.smoke
    @pytest.mark.kueue
    def test_opr_003_operator_emits_failed_event_for_kueue_eviction(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given a Kueue-managed evaluation job that has been admitted,
        when the Kueue Workload is evicted (deleted by a cluster admin),
        then the TrustyAI Operator emits at least one EvaluationFailed Warning Event
        referencing the Kueue eviction and with a source.component that is not evalhub-server."""
        from utilities.kueue_utils import Workload

        if not is_kueue_installed(admin_client):
            pytest.skip("Kueue not installed")

        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        if not kueue_local_queue_exists(admin_client, ns, KUEUE_LOCAL_QUEUE_NAME):
            pytest.skip(f"Kueue LocalQueue {KUEUE_LOCAL_QUEUE_NAME!r} not found in {ns}")

        payload = build_oom_kueue_job_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-opr-003",
            queue_name=KUEUE_LOCAL_QUEUE_NAME,
        )

        job_id, job_name = submit_evalhub_job_and_capture_runtime_job(
            admin_client=admin_client,
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=payload,
        )
        workload = wait_for_evalhub_job_workload_admitted(
            admin_client=admin_client,
            namespace=ns,
            evalhub_job_id=job_id,
            timeout=180,
        )

        # Evict the Kueue Workload for this job
        Workload(client=admin_client, name=workload.name, namespace=ns).delete(wait=False)

        event = wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
            timeout=120,
        )
        operator_events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
            source_component=None,
        )
        operator_events = [
            e for e in operator_events if (e.get("source") or {}).get("component") != LIFECYCLE_SOURCE_SERVER
        ]

        assert operator_events, "Expected EvaluationFailed from operator after Kueue eviction"
        assert event.get("type") == "Warning"
        message = operator_events[0].get("message") or ""
        assert "evict" in message.lower() or "kueue" in message.lower(), (
            f"Expected message referencing eviction, got: {message!r}"
        )

    @pytest.mark.tier1
    def test_opr_004_operator_event_source_component_differs_from_server(
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
        """Given an evaluation fails due to ImagePullBackOff,
        when the operator POSTs failure to EvalHub and emits EvaluationFailed,
        then its source.component is trustyai-service-operator, differs from evalhub-server,
        and filtering by source.component separates server sync vs operator Events."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_bad_image_payload(
            vllm_service_name=lifecycle_signals_vllm_service.name,
            namespace=ns,
            job_name="tc-opr-004",
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
        all_events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_FAILED,
        )

        server_events = filter_events_by_source(events=all_events, source_component=LIFECYCLE_SOURCE_SERVER)
        operator_events = filter_events_by_source(events=all_events, source_component=LIFECYCLE_SOURCE_OPERATOR)

        assert operator_events, "Expected at least one operator EvaluationFailed Event"
        operator_component = (operator_events[0].get("source") or {}).get("component")
        assert operator_component == LIFECYCLE_SOURCE_OPERATOR, (
            f"Operator source.component should be {LIFECYCLE_SOURCE_OPERATOR!r}, got {operator_component!r}"
        )
        assert len(server_events) == 1, (
            f"Expected one server EvaluationFailed after operator API sync; found {len(server_events)}"
        )
        assert len(operator_events) == 1, (
            f"Expected exactly one operator EvaluationFailed; found {len(operator_events)}"
        )
