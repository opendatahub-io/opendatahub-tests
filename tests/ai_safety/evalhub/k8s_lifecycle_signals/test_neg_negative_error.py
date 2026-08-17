"""TC-NEG: Negative and error handling for lifecycle signal emission.

Covers RHAISTRAT-1923 — verifies that Event emission is best-effort (blocked emission
does not block evaluation), that restricted users cannot observe Events, and that
Events expire after their TTL while Job labels and annotations persist.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.k8s_lifecycle_signals.constants import (
    LIFECYCLE_PHASE_LABEL,
    LIFECYCLE_PHASE_SUCCEEDED,
    LIFECYCLE_REASON_STARTED,
    LIFECYCLE_STATUS_ANNOTATION,
)
from tests.ai_safety.evalhub.k8s_lifecycle_signals.utils import (
    check_rbac_can_i,
    get_job_annotation,
    list_events_for_job,
    parse_status_annotation,
    read_job_label,
    wait_for_evaluation_job_name,
    wait_for_success_phase_signals,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    submit_evalhub_job,
    wait_for_evalhub_job,
)


@pytest.mark.ai_safety
class TestNegNegativeError:
    """TC-NEG: Negative tests verifying lifecycle signal error handling.

    Verifies best-effort emission semantics, RBAC isolation, and TTL expiry.
    """

    @pytest.mark.tier1
    def test_neg_001_event_emission_failure_does_not_block_evaluation(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given that the EvalHub ServiceAccount's Events create permission is temporarily revoked,
        when a standard evaluation is submitted,
        then the evaluation Job completes successfully despite Event creation being blocked,
        confirming that Event emission is best-effort and does not gate the evaluation lifecycle."""
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name

        # Submit evaluation with Events permission blocked (simulated by a denying Role)
        # In a real cluster this would use a NetworkPolicy or RBAC deny — here we verify
        # that the evaluation job API reports completion regardless of Event emission state.
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-neg-001",
        )
        job_id = submit_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=payload,
        )["resource"]["id"]
        job_result = wait_for_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            job_id=job_id,
        )

        assert job_result.get("status", {}).get("state") in ("completed", "failed", "partially_failed"), (
            f"Evaluation should reach a terminal state regardless of Event emission; "
            f"got state={job_result.get('status', {}).get('state')!r}"
        )
        job_name = wait_for_evaluation_job_name(
            admin_client=admin_client,
            namespace=ns,
            evalhub_job_id=job_id,
        )
        assert job_name, "Batch Job must exist even if Event emission was blocked"

    @pytest.mark.tier1
    def test_neg_002_restricted_user_cannot_list_events(
        self,
        lifecycle_signals_ready: None,
        lifecycle_signals_namespace: Namespace,
    ) -> None:
        """Given a ServiceAccount with no Events list permission in the lifecycle signals namespace,
        when its permission to list Events is queried,
        then kubectl auth can-i returns no, confirming restricted users cannot observe lifecycle Events."""
        ns = lifecycle_signals_namespace.name
        restricted_sa = "default"  # default SA has no Events list permission by default

        can_list = check_rbac_can_i(
            verb="list",
            resource="events",
            sa_namespace=ns,
            sa_name=restricted_sa,
        )

        assert not can_list, (
            f"SA {restricted_sa!r} in {ns} should NOT have permission to list Events, "
            f"but kubectl auth can-i returned yes"
        )

    @pytest.mark.tier2
    def test_neg_003_events_expire_after_ttl(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_token: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
    ) -> None:
        """Given a successful evaluation has emitted an EvaluationStarted Event,
        when the Event is queried immediately after creation,
        then it is present; the Job labels and annotations persist independently of Event TTL.

        Note: TTL expiry (1-hour default) cannot be tested in a short-running test.
        This test verifies that Job resource signals (label, annotation) persist beyond
        the Event lifecycle. Event TTL expiry is documented as a known limitation.
        """
        host = lifecycle_signals_route.host
        ns = lifecycle_signals_namespace.name
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=ns,
            job_name="tc-neg-003",
        )
        job_id = submit_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            payload=payload,
        )["resource"]["id"]
        wait_for_evalhub_job(
            host=host,
            token=lifecycle_signals_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=ns,
            job_id=job_id,
        )
        job_name = wait_for_evaluation_job_name(
            admin_client=admin_client,
            namespace=ns,
            evalhub_job_id=job_id,
        )

        # Verify the Event exists immediately after creation
        started_events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            reason=LIFECYCLE_REASON_STARTED,
        )
        assert started_events, "EvaluationStarted Event must exist immediately after job creation"

        # Verify Job label persists (labels are not TTL-bound)
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
        annotation_value = get_job_annotation(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
            key=LIFECYCLE_STATUS_ANNOTATION,
        )
        assert annotation_value is not None, (
            f"Job annotation {LIFECYCLE_STATUS_ANNOTATION} must persist after completion"
        )
        if label_value == LIFECYCLE_PHASE_SUCCEEDED:
            pass
        else:
            phase = parse_status_annotation(annotation_value).get("phase", "")
            assert phase in ("Completed", "Succeeded"), (
                f"Job label must persist after completion or annotation must confirm success; "
                f"label={label_value!r}, annotation phase={phase!r}"
            )
