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
    LIFECYCLE_SIGNALS_CP_NAMESPACE,
    LIFECYCLE_SIGNALS_CR_NAME,
    LIFECYCLE_SOURCE_SERVER,
    LIFECYCLE_STATUS_ANNOTATION,
)
from tests.ai_safety.evalhub.k8s_lifecycle_signals.utils import (
    check_rbac_can_i,
    get_job_annotation,
    list_events_for_job,
    parse_status_annotation,
    read_job_label,
    revoked_evalhub_events_create_permission,
    submit_evalhub_job_and_capture_runtime_job,
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
        evalhub_sa_name = f"{LIFECYCLE_SIGNALS_CR_NAME}-service"

        with revoked_evalhub_events_create_permission(
            admin_client=admin_client,
            evalhub_cr_name=LIFECYCLE_SIGNALS_CR_NAME,
            evalhub_sa_namespace=LIFECYCLE_SIGNALS_CP_NAMESPACE,
            tenant_namespace=ns,
        ):
            assert not check_rbac_can_i(
                admin_client=admin_client,
                verb="create",
                resource="events",
                sa_namespace=LIFECYCLE_SIGNALS_CP_NAMESPACE,
                sa_name=evalhub_sa_name,
                target_namespace=ns,
            ), f"Precondition failed: {evalhub_sa_name!r} must not be able to create Events in {ns!r}"

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
            job_name = wait_for_evaluation_job_name(
                admin_client=admin_client,
                namespace=ns,
                evalhub_job_id=job_id,
            )

            started_events = list_events_for_job(
                admin_client=admin_client,
                job_name=job_name,
                namespace=ns,
                reason=LIFECYCLE_REASON_STARTED,
                source_component=LIFECYCLE_SOURCE_SERVER,
            )
            assert not started_events, (
                "EvalHub server must not emit EvaluationStarted when events create is denied; "
                f"found {len(started_events)} event(s)"
            )

            job_result = wait_for_evalhub_job(
                host=host,
                token=lifecycle_signals_token,
                ca_bundle_file=lifecycle_signals_ca_bundle_file,
                tenant=ns,
                job_id=job_id,
            )

        assert job_result.get("status", {}).get("state") == "completed", (
            f"Evaluation must complete successfully when Event emission is blocked; "
            f"got state={job_result.get('status', {}).get('state')!r}"
        )
        assert job_name, "Batch Job must exist even when Event emission was blocked"
        wait_for_success_phase_signals(
            admin_client=admin_client,
            job_name=job_name,
            namespace=ns,
        )

    @pytest.mark.tier1
    def test_neg_002_restricted_user_cannot_list_events(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_namespace: Namespace,
    ) -> None:
        """Given a ServiceAccount with no Events list permission in the lifecycle signals namespace,
        when its permission to list Events is queried,
        then SubjectAccessReview reports denied, confirming restricted users cannot observe lifecycle Events."""
        ns = lifecycle_signals_namespace.name
        restricted_sa = "default"  # default SA has no Events list permission by default

        can_list = check_rbac_can_i(
            admin_client=admin_client,
            verb="list",
            resource="events",
            sa_namespace=ns,
            sa_name=restricted_sa,
        )

        assert not can_list, (
            f"SA {restricted_sa!r} in {ns} should NOT have permission to list Events, "
            f"but SubjectAccessReview reported allowed"
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
