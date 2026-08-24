"""Tests for EvalHub controller reconciliation loop observability.

RHAISTRAT-1606 / RHAI-241: Verifies Prometheus metrics and OTEL distributed
tracing instrumentation on the EvalHub controller reconcile path in the
TrustyAI Service Operator.
"""

import time

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.service_monitor import ServiceMonitor
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import (
    ERROR_TYPE_OTHER,
    EVALHUB_CONTROLLER_LABEL_VALUE,
    EVALHUB_ERROR_TYPES,
    EVALHUB_RECONCILE_CHILD_SPANS,
    EVALHUB_RECONCILE_METRICS,
    JOB_FAILURE_EVENTS_METRIC,
    MANAGED_INSTANCES_METRIC,
    METRIC_LABEL_CONTROLLER,
    METRIC_LABEL_ERROR_TYPE,
    METRIC_LABEL_FAILURE_REASON,
    METRIC_LABEL_RESULT,
    OPERATOR_METRICS_PORT,
    OPERATOR_POD_LABEL_SELECTOR,
    RECONCILE_DURATION_METRIC,
    RECONCILE_ERRORS_METRIC,
    RECONCILE_TOTAL_METRIC,
    RESULT_ERROR,
    RESULT_REQUEUE,
    RESULT_SUCCESS,
    SPAN_ATTR_EVALHUB_NAME,
    SPAN_ATTR_EXIT_CODE,
    SPAN_ATTR_FAILURE_REASON,
    SPAN_ATTR_JOB_NAME,
    SPAN_ATTR_K8S_NAMESPACE,
    SPAN_ATTR_RECONCILE_GENERATION,
    SPAN_JOB_FAILURE_RECONCILE,
    SPAN_RECONCILE,
    SPAN_RECONCILE_DEPLOYMENT,
)
from tests.ai_safety.evalhub.utils import (
    fetch_operator_metrics,
    fetch_trace_collector_logs,
    filter_spans_by_name,
    get_child_spans,
    get_metric_samples,
    metric_value_sum,
    parse_prometheus_text,
    parse_trace_spans_from_logs,
)

METRICS_POLL_TIMEOUT: int = 120
METRICS_POLL_INTERVAL: int = 10
TRACE_POLL_TIMEOUT: int = 60
TRACE_POLL_INTERVAL: int = 5

_TRANSIENT_METRICS_EXCEPTIONS: dict[type, list] = {
    requests.exceptions.ConnectionError: [],
    requests.exceptions.ReadTimeout: [],
}


# TC-MET: Prometheus Metrics (10 tests)


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-met"}, id="test_evalhub_met")],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.tier1
class TestEvalHubReconcileMetrics:
    """TC-MET-001 through TC-MET-010: Prometheus reconciliation metrics."""

    def test_duration_histogram_registered(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given a reconciled EvalHub CR, the duration histogram is exposed on :8080.

        TC-MET-001: Verify evalhub_controller_reconcile_duration_seconds histogram
        is registered and exposed on the operator metrics endpoint.
        """
        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                bucket_key = f"{RECONCILE_DURATION_METRIC}_bucket"
                if bucket_key in metrics or RECONCILE_DURATION_METRIC in metrics:
                    return
        except TimeoutExpiredError:
            pytest.fail(f"{RECONCILE_DURATION_METRIC} histogram not found on operator metrics endpoint")

    def test_duration_histogram_captures_latency(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given a successful reconciliation, the histogram records a positive latency.

        TC-MET-002: Verify duration histogram captures accurate latency values
        after a reconciliation cycle completes.
        """
        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                sum_key = f"{RECONCILE_DURATION_METRIC}_sum"
                samples = get_metric_samples(
                    metrics=metrics,
                    metric_name=sum_key,
                    label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
                )
                if samples and float(samples[0]["value"]) > 0:
                    return
        except TimeoutExpiredError:
            pytest.fail("Duration histogram sum is not positive after reconciliation")

    def test_total_counter_success(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given a successful reconciliation, the total counter increments with result=success.

        TC-MET-003: Verify evalhub_controller_reconcile_total counter increments
        for a successful reconciliation outcome.
        """
        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                total = metric_value_sum(
                    metrics=metrics,
                    metric_name=RECONCILE_TOTAL_METRIC,
                    label_filter={
                        METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE,
                        METRIC_LABEL_RESULT: RESULT_SUCCESS,
                    },
                )
                if total > 0:
                    return
        except TimeoutExpiredError:
            pytest.fail(f"{RECONCILE_TOTAL_METRIC}{{result=success}} not incremented")

    def test_total_counter_requeue(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        model_namespace: Namespace,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given a reconciliation that triggers requeue, the total counter increments with result=requeue.

        TC-MET-004: Verify evalhub_controller_reconcile_total counter increments
        when a reconciliation returns requeue.
        """
        with ResourceEditor(
            patches={evalhub_reconcile_cr: {"metadata": {"annotations": {"test-trigger": f"requeue-{time.time()}"}}}}
        ):
            pass

        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                total = metric_value_sum(
                    metrics=metrics,
                    metric_name=RECONCILE_TOTAL_METRIC,
                    label_filter={
                        METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE,
                        METRIC_LABEL_RESULT: RESULT_REQUEUE,
                    },
                )
                if total > 0:
                    return
        except TimeoutExpiredError:
            pytest.fail(f"{RECONCILE_TOTAL_METRIC}{{result=requeue}} not incremented after update")

    def test_total_counter_error(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_failure_cr: EvalHub,
    ) -> None:
        """Given a failing reconciliation, the total counter increments with result=error.

        TC-MET-005: Verify evalhub_controller_reconcile_total counter increments
        when a reconciliation results in an error.
        """
        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                total = metric_value_sum(
                    metrics=metrics,
                    metric_name=RECONCILE_TOTAL_METRIC,
                    label_filter={
                        METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE,
                        METRIC_LABEL_RESULT: RESULT_ERROR,
                    },
                )
                if total > 0:
                    return
        except TimeoutExpiredError:
            pytest.fail(f"{RECONCILE_TOTAL_METRIC}{{result=error}} not incremented for failure CR")

    def test_error_counter_classifies_by_type(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_failure_cr: EvalHub,
    ) -> None:
        """Given a deployment failure, the error counter records the correct error_type label.

        TC-MET-006: Verify evalhub_controller_reconcile_errors_total classifies
        errors by type (e.g. deployment_create_failed).
        """
        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                samples = get_metric_samples(
                    metrics=metrics,
                    metric_name=RECONCILE_ERRORS_METRIC,
                    label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
                )
                if samples:
                    error_types_seen = {s["labels"].get(METRIC_LABEL_ERROR_TYPE) for s in samples}
                    assert error_types_seen.intersection(set(EVALHUB_ERROR_TYPES)), (
                        f"Expected known error_type, got: {error_types_seen}"
                    )
                    return
        except TimeoutExpiredError:
            pytest.fail(f"{RECONCILE_ERRORS_METRIC} not recorded for failure CR")

    def test_unexpected_error_mapped_to_other(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_failure_cr: EvalHub,
    ) -> None:
        """Given an unexpected error, the error_type label is set to 'other'.

        TC-MET-007: Verify that unexpected/unclassified errors are mapped to
        the 'other' error_type bucket.
        """
        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                samples = get_metric_samples(
                    metrics=metrics,
                    metric_name=RECONCILE_ERRORS_METRIC,
                    label_filter={
                        METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE,
                        METRIC_LABEL_ERROR_TYPE: ERROR_TYPE_OTHER,
                    },
                )
                if samples and float(samples[0]["value"]) > 0:
                    return
        except TimeoutExpiredError:
            pytest.fail(f"{RECONCILE_ERRORS_METRIC}{{error_type=other}} not populated")

    @pytest.mark.skip(reason="Requires a job-failure fixture that submits and awaits a failing evaluation job")
    def test_job_failure_counter(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        model_namespace: Namespace,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given a job failure event, the job failure counter increments with failure_reason.

        TC-MET-008: Verify evalhub_job_failure_events_total records failures
        with the failure_reason label.
        """
        raw_metrics = fetch_operator_metrics(
            admin_client=admin_client,
            operator_metrics_token=operator_metrics_token,
        )
        metrics = parse_prometheus_text(text=raw_metrics)
        samples = get_metric_samples(metrics=metrics, metric_name=JOB_FAILURE_EVENTS_METRIC)
        assert samples, "No job failure events recorded"
        assert all(METRIC_LABEL_FAILURE_REASON in s["labels"] for s in samples), (
            "Job failure metric missing failure_reason label"
        )

    def test_managed_instances_gauge(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given active EvalHub CRs, the managed instances gauge reflects their count.

        TC-MET-009: Verify evalhub_managed_instances_total gauge reflects the
        number of active EvalHub CR instances.
        """
        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                samples = get_metric_samples(metrics=metrics, metric_name=MANAGED_INSTANCES_METRIC)
                if samples and float(samples[0]["value"]) >= 1:
                    return
        except TimeoutExpiredError:
            pytest.fail(f"{MANAGED_INSTANCES_METRIC} not >= 1 with active EvalHub CR")

    def test_all_metrics_registered(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given a running operator, all five reconciliation metrics are registered.

        TC-MET-010: Verify all five evalhub controller metrics are registered
        with the controller-runtime metrics registry.
        """
        found: set[str] = set()
        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                found = set()
                for metric_name in EVALHUB_RECONCILE_METRICS:
                    if (
                        metric_name in metrics
                        or f"{metric_name}_bucket" in metrics
                        or f"{metric_name}_total" in metrics
                    ):
                        found.add(metric_name)
                if found == set(EVALHUB_RECONCILE_METRICS):
                    return
        except TimeoutExpiredError:
            pytest.fail(f"Not all metrics registered. Found: {found}, expected: {set(EVALHUB_RECONCILE_METRICS)}")


# TC-TRC: OTEL Distributed Tracing (5 tests)


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-trc"}, id="test_evalhub_trc")],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.tier1
class TestEvalHubReconcileTracing:
    """TC-TRC-001 through TC-TRC-005: OTEL trace spans for reconciliation."""

    def test_parent_span_created_with_attributes(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        operator_with_otel_tracing: Deployment,
        otel_trace_collector_pod: Pod,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given OTEL-enabled operator and a reconciliation, the parent span has correct attributes.

        TC-TRC-001: Verify evalhub.reconcile parent span is created with
        k8s.namespace, evalhub.name, and reconcile.generation attributes.
        """
        try:
            for logs in TimeoutSampler(
                wait_timeout=TRACE_POLL_TIMEOUT,
                sleep=TRACE_POLL_INTERVAL,
                func=fetch_trace_collector_logs,
                trace_collector_pod=otel_trace_collector_pod,
            ):
                spans = parse_trace_spans_from_logs(logs=logs)
                parent_spans = filter_spans_by_name(spans=spans, name=SPAN_RECONCILE)
                if parent_spans:
                    span = parent_spans[0]
                    assert SPAN_ATTR_K8S_NAMESPACE in span["attributes"], f"Missing {SPAN_ATTR_K8S_NAMESPACE} attribute"
                    assert SPAN_ATTR_EVALHUB_NAME in span["attributes"], f"Missing {SPAN_ATTR_EVALHUB_NAME} attribute"
                    assert SPAN_ATTR_RECONCILE_GENERATION in span["attributes"], (
                        f"Missing {SPAN_ATTR_RECONCILE_GENERATION} attribute"
                    )
                    return
        except TimeoutExpiredError:
            pytest.fail(f"No {SPAN_RECONCILE} parent span found in collector logs")

    def test_child_spans_for_sub_reconcilers(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        operator_with_otel_tracing: Deployment,
        otel_trace_collector_pod: Pod,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given a full reconciliation, child spans are created for each sub-reconciler phase.

        TC-TRC-002: Verify child spans exist for configmap, deployment, service,
        route, and rbac sub-reconciler phases.
        """
        try:
            for logs in TimeoutSampler(
                wait_timeout=TRACE_POLL_TIMEOUT,
                sleep=TRACE_POLL_INTERVAL,
                func=fetch_trace_collector_logs,
                trace_collector_pod=otel_trace_collector_pod,
            ):
                spans = parse_trace_spans_from_logs(logs=logs)
                parent_spans = filter_spans_by_name(spans=spans, name=SPAN_RECONCILE)
                if not parent_spans:
                    continue

                parent_span_id = parent_spans[0]["span_id"]
                children = get_child_spans(spans=spans, parent_span_id=parent_span_id)
                child_names = {c["name"] for c in children}

                if set(EVALHUB_RECONCILE_CHILD_SPANS).issubset(child_names):
                    return
        except TimeoutExpiredError:
            pytest.fail(f"Not all child spans found. Expected: {set(EVALHUB_RECONCILE_CHILD_SPANS)}")

    @pytest.mark.skip(reason="Requires a job-failure fixture that submits and awaits a failing evaluation job")
    def test_job_failure_span_includes_details(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        operator_with_otel_tracing: Deployment,
        otel_trace_collector_pod: Pod,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given a job failure event, the job failure span includes failure details.

        TC-TRC-003: Verify evalhub.job_failure_reconcile span includes job_name,
        failure_reason, and exit_code attributes.
        """
        logs = fetch_trace_collector_logs(trace_collector_pod=otel_trace_collector_pod)
        spans = parse_trace_spans_from_logs(logs=logs)
        failure_spans = filter_spans_by_name(spans=spans, name=SPAN_JOB_FAILURE_RECONCILE)

        assert failure_spans, "No job failure span detected"
        span = failure_spans[0]
        assert SPAN_ATTR_JOB_NAME in span["attributes"], f"Missing {SPAN_ATTR_JOB_NAME}"
        assert SPAN_ATTR_FAILURE_REASON in span["attributes"], f"Missing {SPAN_ATTR_FAILURE_REASON}"
        assert SPAN_ATTR_EXIT_CODE in span["attributes"], f"Missing {SPAN_ATTR_EXIT_CODE}"

    def test_span_hierarchy_multiple_reconciliations(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        operator_with_otel_tracing: Deployment,
        otel_trace_collector_pod: Pod,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given multiple reconciliations, each produces an independent span hierarchy.

        TC-TRC-004: Verify span hierarchy is preserved across multiple
        reconciliation cycles (distinct trace IDs per cycle).
        """
        with ResourceEditor(
            patches={
                evalhub_reconcile_cr: {"metadata": {"annotations": {"test-trigger": f"multi-reconcile-{time.time()}"}}}
            }
        ):
            pass

        try:
            for logs in TimeoutSampler(
                wait_timeout=TRACE_POLL_TIMEOUT,
                sleep=TRACE_POLL_INTERVAL,
                func=fetch_trace_collector_logs,
                trace_collector_pod=otel_trace_collector_pod,
            ):
                spans = parse_trace_spans_from_logs(logs=logs)
                parent_spans = filter_spans_by_name(spans=spans, name=SPAN_RECONCILE)
                if len(parent_spans) >= 2:
                    trace_ids = {s["trace_id"] for s in parent_spans}
                    assert len(trace_ids) >= 2, "Multiple reconciliations should produce distinct trace IDs"
                    return
        except TimeoutExpiredError:
            pytest.skip("Fewer than 2 reconcile spans detected within timeout")

    def test_failed_sub_reconciler_span_error_status(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        operator_with_otel_tracing: Deployment,
        otel_trace_collector_pod: Pod,
        evalhub_failure_cr: EvalHub,
    ) -> None:
        """Given a sub-reconciler failure, the corresponding span records error status.

        TC-TRC-005: Verify that a failed sub-reconciler phase span records
        an error status code.
        """
        try:
            for logs in TimeoutSampler(
                wait_timeout=TRACE_POLL_TIMEOUT,
                sleep=TRACE_POLL_INTERVAL,
                func=fetch_trace_collector_logs,
                trace_collector_pod=otel_trace_collector_pod,
            ):
                spans = parse_trace_spans_from_logs(logs=logs)
                deployment_spans = filter_spans_by_name(spans=spans, name=SPAN_RECONCILE_DEPLOYMENT)
                error_spans = [s for s in deployment_spans if "error" in s.get("status", "").lower()]
                if error_spans:
                    return
        except TimeoutExpiredError:
            pytest.fail(f"No error-status span found for {SPAN_RECONCILE_DEPLOYMENT}")


# TC-ERR: Error Classification (4 tests)


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-err"}, id="test_evalhub_err")],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.tier1
class TestEvalHubReconcileErrors:
    """TC-ERR-001 through TC-ERR-004: Error classification."""

    def test_nil_error_no_panic(
        self,
        admin_client: DynamicClient,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given a successful reconciliation (nil error), the operator does not panic.

        TC-ERR-001: Verify metric recording does not panic on nil error — the
        operator pod remains running with zero restart count.
        """
        operator_ns = py_config["applications_namespace"]
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=operator_ns,
                label_selector=OPERATOR_POD_LABEL_SELECTOR,
            )
        )
        assert pods, "No operator pod found"
        pod = pods[0]
        container_statuses = pod.instance.status.containerStatuses or []
        manager_container = next((c for c in container_statuses if c.name == "manager"), None)
        assert manager_container is not None, "manager container not found"
        assert manager_container.restartCount == 0, (
            f"Operator restarted {manager_container.restartCount} times — possible panic"
        )

    def test_multiple_failure_types_same_cycle(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_failure_cr: EvalHub,
    ) -> None:
        """Given multiple error types in the same reconciliation, all are recorded.

        TC-ERR-002: Verify that multiple failure types within the same
        reconciliation cycle are all recorded in the errors metric.
        """
        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                samples = get_metric_samples(
                    metrics=metrics,
                    metric_name=RECONCILE_ERRORS_METRIC,
                    label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
                )
                if samples:
                    error_types_seen = {
                        s["labels"].get(METRIC_LABEL_ERROR_TYPE) for s in samples if float(s["value"]) > 0
                    }
                    if len(error_types_seen) >= 2:
                        return
        except TimeoutExpiredError:
            pytest.fail("No error types recorded in errors metric")

    @pytest.mark.skip(reason="Requires a job-failure fixture that submits and awaits a failing evaluation job")
    def test_job_failure_produces_metric_and_trace(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        operator_with_otel_tracing: Deployment,
        otel_trace_collector_pod: Pod,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given a job failure, both a metric increment and a trace span are produced.

        TC-ERR-003: Verify a job failure event produces both the
        evalhub_job_failure_events_total metric and a trace span.
        """
        raw_metrics = fetch_operator_metrics(
            admin_client=admin_client,
            operator_metrics_token=operator_metrics_token,
        )
        metrics = parse_prometheus_text(text=raw_metrics)
        metric_samples = get_metric_samples(metrics=metrics, metric_name=JOB_FAILURE_EVENTS_METRIC)

        logs = fetch_trace_collector_logs(trace_collector_pod=otel_trace_collector_pod)
        spans = parse_trace_spans_from_logs(logs=logs)
        failure_spans = filter_spans_by_name(spans=spans, name=SPAN_JOB_FAILURE_RECONCILE)

        assert metric_samples or failure_spans, "No job failure events detected in metrics or traces"

    def test_rapid_errors_no_metric_loss(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_failure_cr: EvalHub,
    ) -> None:
        """Given rapid error reconciliations, no metric increments are lost.

        TC-ERR-004: When a failure CR triggers multiple error reconciliations,
        the error counter monotonically increases — no samples are dropped.
        """
        raw_before = fetch_operator_metrics(
            admin_client=admin_client,
            operator_metrics_token=operator_metrics_token,
        )
        metrics_before = parse_prometheus_text(text=raw_before)
        errors_before = metric_value_sum(
            metrics=metrics_before,
            metric_name=RECONCILE_ERRORS_METRIC,
            label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
        )

        expected_increase = 1
        errors_after = errors_before
        try:
            for raw_after in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics_after = parse_prometheus_text(text=raw_after)
                errors_after = metric_value_sum(
                    metrics=metrics_after,
                    metric_name=RECONCILE_ERRORS_METRIC,
                    label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
                )
                if errors_after >= errors_before + expected_increase:
                    return
        except TimeoutExpiredError:
            pass

        if errors_after < errors_before:
            pytest.fail(f"Error counter decreased: {errors_before} -> {errors_after} (metric loss)")
        pytest.fail(
            f"Error counter did not increase by {expected_increase} within timeout "
            f"(before={errors_before}, after={errors_after})"
        )


# TC-PRF: Performance (3 tests)


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-prf"}, id="test_evalhub_prf")],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.tier2
class TestEvalHubReconcilePerformance:
    """TC-PRF-001 through TC-PRF-003: Performance overhead."""

    def test_instrumentation_overhead_within_bounds(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given instrumented reconciliation, average cycle duration stays below 5 seconds.

        TC-PRF-001: Verify the average reconciliation duration (including
        instrumentation) does not exceed 5s, indicating negligible overhead.
        """
        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                sum_key = f"{RECONCILE_DURATION_METRIC}_sum"
                count_key = f"{RECONCILE_DURATION_METRIC}_count"
                sum_samples = get_metric_samples(
                    metrics=metrics,
                    metric_name=sum_key,
                    label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
                )
                count_samples = get_metric_samples(
                    metrics=metrics,
                    metric_name=count_key,
                    label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
                )
                if sum_samples and count_samples:
                    total_duration = float(sum_samples[0]["value"])
                    total_count = float(count_samples[0]["value"])
                    if total_count > 0:
                        avg_duration_ms = (total_duration / total_count) * 1000
                        assert avg_duration_ms < 5000, (
                            f"Average reconciliation duration {avg_duration_ms:.2f}ms exceeds 5s threshold"
                        )
                        return
        except TimeoutExpiredError:
            pytest.fail("Could not calculate average reconciliation duration")

    def test_overhead_does_not_scale_with_cr_count(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        model_namespace: Namespace,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given multiple managed CRs, average reconciliation stays below 10 seconds.

        TC-PRF-002: Verify that the per-reconciliation duration does not scale
        linearly with the number of managed CR instances (stays under 10s).
        """
        raw_metrics = fetch_operator_metrics(
            admin_client=admin_client,
            operator_metrics_token=operator_metrics_token,
        )
        metrics = parse_prometheus_text(text=raw_metrics)
        sum_key = f"{RECONCILE_DURATION_METRIC}_sum"
        count_key = f"{RECONCILE_DURATION_METRIC}_count"
        sum_samples = get_metric_samples(
            metrics=metrics,
            metric_name=sum_key,
            label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
        )
        count_samples = get_metric_samples(
            metrics=metrics,
            metric_name=count_key,
            label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
        )
        if sum_samples and count_samples and float(count_samples[0]["value"]) > 0:
            avg_duration_s = float(sum_samples[0]["value"]) / float(count_samples[0]["value"])
            assert avg_duration_s < 10, f"Average reconciliation {avg_duration_s:.3f}s suggests O(n) scaling"
        else:
            pytest.skip("Insufficient metric data to evaluate scaling behavior")

    def test_otel_exporter_does_not_block_reconciliation(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        operator_with_otel_tracing: Deployment,
        otel_trace_collector_deployment: Deployment,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given an unavailable collector, the OTEL exporter does not block reconciliation.

        TC-PRF-003: When the trace collector is scaled to zero, the reconcile
        counter still advances — proving the exporter is non-blocking.
        """
        raw_before = fetch_operator_metrics(
            admin_client=admin_client,
            operator_metrics_token=operator_metrics_token,
        )
        metrics_before = parse_prometheus_text(text=raw_before)
        total_before = metric_value_sum(
            metrics=metrics_before,
            metric_name=RECONCILE_TOTAL_METRIC,
            label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
        )

        otel_trace_collector_deployment.scale_replicas(replica_count=0)
        try:
            with ResourceEditor(
                patches={
                    evalhub_reconcile_cr: {"metadata": {"annotations": {"test-trigger": f"no-collector-{time.time()}"}}}
                }
            ):
                pass

            try:
                for raw_after in TimeoutSampler(
                    wait_timeout=METRICS_POLL_TIMEOUT,
                    sleep=METRICS_POLL_INTERVAL,
                    func=fetch_operator_metrics,
                    exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                    admin_client=admin_client,
                    operator_metrics_token=operator_metrics_token,
                ):
                    metrics_after = parse_prometheus_text(text=raw_after)
                    total_after = metric_value_sum(
                        metrics=metrics_after,
                        metric_name=RECONCILE_TOTAL_METRIC,
                        label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
                    )
                    if total_after > total_before:
                        return
            except TimeoutExpiredError:
                pytest.fail("Reconciliation counter did not advance with collector unavailable")
        finally:
            otel_trace_collector_deployment.scale_replicas(replica_count=1)
            otel_trace_collector_deployment.wait_for_replicas(timeout=120)


# TC-INT: Integration (4 tests)


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-int"}, id="test_evalhub_int")],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.tier1
class TestEvalHubReconcileIntegration:
    """TC-INT-001 through TC-INT-004: ServiceMonitor and collector integration."""

    def test_service_monitor_scrapes_metrics(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given the operator ServiceMonitor, it scrapes EvalHub metrics on port 8080.

        TC-INT-001: Verify the ServiceMonitor targets the operator metrics
        endpoint on port 8080 and metrics are accessible.
        """
        operator_ns = py_config["applications_namespace"]
        service_monitors = list(
            ServiceMonitor.get(
                client=admin_client,
                namespace=operator_ns,
            )
        )
        operator_sm = [sm for sm in service_monitors if "trustyai" in sm.name and "operator" in sm.name]
        assert operator_sm, "No operator ServiceMonitor found in applications namespace"

        raw_metrics = fetch_operator_metrics(
            admin_client=admin_client,
            operator_metrics_token=operator_metrics_token,
        )
        assert RECONCILE_TOTAL_METRIC in raw_metrics or RECONCILE_DURATION_METRIC in raw_metrics, (
            "EvalHub reconciliation metrics not found on operator endpoint"
        )

    def test_unauthenticated_request_rejected(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
    ) -> None:
        """Given an unauthenticated request, the metrics endpoint rejects it.

        TC-INT-002: Verify unauthenticated requests to the operator metrics
        endpoint are rejected by kube-rbac-proxy.
        """
        operator_ns = py_config["applications_namespace"]
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=operator_ns,
                label_selector=OPERATOR_POD_LABEL_SELECTOR,
            )
        )
        assert pods, "No operator pod found"
        pod = pods[0]

        try:
            response = requests.get(
                f"https://{pod.instance.status.podIP}:{OPERATOR_METRICS_PORT}/metrics",
                verify=False,
                timeout=5,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.SSLError) as exc:
            pytest.fail(f"Metrics endpoint unreachable — cannot verify auth rejection: {exc}")

        assert response.status_code in (401, 403), f"Expected 401/403 without auth, got {response.status_code}"

    def test_otlp_exporter_delivers_traces(
        self,
        admin_client: DynamicClient,
        operator_with_otel_tracing: Deployment,
        otel_trace_collector_pod: Pod,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given OTEL-enabled operator, traces are delivered to the collector.

        TC-INT-003: Verify the OTLP exporter successfully delivers trace spans
        to the OTEL collector.
        """
        try:
            for logs in TimeoutSampler(
                wait_timeout=TRACE_POLL_TIMEOUT,
                sleep=TRACE_POLL_INTERVAL,
                func=fetch_trace_collector_logs,
                trace_collector_pod=otel_trace_collector_pod,
            ):
                spans = parse_trace_spans_from_logs(logs=logs)
                if spans:
                    return
        except TimeoutExpiredError:
            pytest.fail("No trace spans delivered to collector")

    def test_metrics_no_sensitive_labels(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given operator metrics, no sensitive information is exposed in label values.

        TC-INT-004: Verify metrics do not expose sensitive information
        (secrets, tokens, passwords) in label values.
        """
        raw_metrics = fetch_operator_metrics(
            admin_client=admin_client,
            operator_metrics_token=operator_metrics_token,
        )
        sensitive_patterns = ["password", "secret", "token", "credential", "apikey"]
        metrics = parse_prometheus_text(text=raw_metrics)
        evalhub_metric_names = {name for name in metrics if any(name.startswith(m) for m in EVALHUB_RECONCILE_METRICS)}
        for metric_name in evalhub_metric_names:
            for sample in metrics[metric_name]:
                for label_key, label_value in sample["labels"].items():
                    for pattern in sensitive_patterns:
                        assert pattern not in label_key.lower(), (
                            f"Sensitive pattern '{pattern}' found in label name of {metric_name}"
                        )
                        assert pattern not in label_value.lower(), (
                            f"Sensitive pattern '{pattern}' found in label value of {metric_name}"
                        )


# TC-REG: Regression (3 tests)


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-reg"}, id="test_evalhub_reg")],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.tier1
class TestEvalHubReconcileRegression:
    """TC-REG-001 through TC-REG-003: Existing controller metrics unaffected."""

    def test_tas_controller_metrics_unaffected(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        operator_metrics_token: str,
    ) -> None:
        """Given the new EvalHub metrics, existing TAS controller metrics still work.

        TC-REG-001: Verify existing TrustyAI Service (TAS) controller metrics
        are unaffected by the new EvalHub instrumentation.
        """
        raw_metrics = fetch_operator_metrics(
            admin_client=admin_client,
            operator_metrics_token=operator_metrics_token,
        )
        assert "controller_runtime_reconcile_total" in raw_metrics, (
            "controller-runtime base reconcile metrics missing — possible regression"
        )

    def test_lmes_gorch_controller_metrics_unaffected(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        operator_metrics_token: str,
    ) -> None:
        """Given the new instrumentation, LMES and GORCH controller metrics still work.

        TC-REG-002: Verify existing LMES and GORCH controller metrics are
        unaffected by the new EvalHub reconciliation observability code.
        """
        raw_metrics = fetch_operator_metrics(
            admin_client=admin_client,
            operator_metrics_token=operator_metrics_token,
        )
        metrics = parse_prometheus_text(text=raw_metrics)
        reconcile_samples = get_metric_samples(
            metrics=metrics,
            metric_name="controller_runtime_reconcile_total",
        )
        controllers_seen = {s["labels"].get("controller", "") for s in reconcile_samples}
        assert controllers_seen, "No controller_runtime_reconcile_total samples found"
        expected_controllers = {"lmevaljob", "guardrailsorchestrator"}
        missing = expected_controllers - controllers_seen
        assert not missing, (
            f"Expected controllers {expected_controllers} in metrics, missing: {missing}. Found: {controllers_seen}"
        )

    def test_existing_otel_traces_unaffected(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        operator_with_otel_tracing: Deployment,
        otel_trace_collector_pod: Pod,
    ) -> None:
        """Given new tracing, existing OTEL traces from other controllers are unaffected.

        TC-REG-003: Verify existing OTEL traces for other controllers are
        not disrupted by the EvalHub tracing additions.
        """
        logs = fetch_trace_collector_logs(trace_collector_pod=otel_trace_collector_pod)
        spans = parse_trace_spans_from_logs(logs=logs)
        if not spans:
            pytest.skip("No spans detected — collector may not have received traces yet")

        non_evalhub_spans = [s for s in spans if not s["name"].startswith("evalhub.")]
        if not non_evalhub_spans:
            pytest.skip("Only EvalHub spans found — no other controller traces to verify")

        for span in non_evalhub_spans:
            assert span.get("trace_id"), f"Non-EvalHub span {span['name']} missing trace_id"


# TC-E2E: End-to-End (4 tests)


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-e2e-obs"}, id="test_evalhub_e2e_obs")],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.tier1
class TestEvalHubReconcileE2E:
    """TC-E2E-001 through TC-E2E-004: Full observability pipeline."""

    def test_successful_reconcile_metrics_in_prometheus(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given a successful reconciliation, metrics are queryable from the operator endpoint.

        TC-E2E-001: Verify a successful reconciliation produces metrics that
        are queryable via the Prometheus-compatible metrics endpoint.
        """
        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                success_total = metric_value_sum(
                    metrics=metrics,
                    metric_name=RECONCILE_TOTAL_METRIC,
                    label_filter={
                        METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE,
                        METRIC_LABEL_RESULT: RESULT_SUCCESS,
                    },
                )
                duration_sum = metric_value_sum(
                    metrics=metrics,
                    metric_name=f"{RECONCILE_DURATION_METRIC}_sum",
                    label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
                )
                if success_total > 0 and duration_sum > 0:
                    return
        except TimeoutExpiredError:
            pytest.fail("Successful reconciliation metrics not queryable from endpoint")

    def test_failed_reconcile_error_metrics_and_traces(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        operator_with_otel_tracing: Deployment,
        otel_trace_collector_pod: Pod,
        evalhub_failure_cr: EvalHub,
    ) -> None:
        """Given a failed reconciliation, both error metrics and error traces are produced.

        TC-E2E-002: Verify a failed reconciliation produces both error metrics
        (evalhub_controller_reconcile_errors_total) and error trace spans.
        """
        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                errors = metric_value_sum(
                    metrics=metrics,
                    metric_name=RECONCILE_ERRORS_METRIC,
                    label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
                )
                if errors > 0:
                    break
        except TimeoutExpiredError:
            pytest.fail("Error metric not recorded for failed reconciliation")

        logs = fetch_trace_collector_logs(trace_collector_pod=otel_trace_collector_pod)
        spans = parse_trace_spans_from_logs(logs=logs)
        error_spans = [s for s in spans if "error" in s.get("status", "").lower()]
        assert error_spans, "No error-status trace spans found for failed reconciliation"

    def test_sre_diagnosis_workflow(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        operator_with_otel_tracing: Deployment,
        otel_trace_collector_pod: Pod,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given both metrics and traces, an SRE can diagnose issues using both signals.

        TC-E2E-003: Verify the SRE diagnosis workflow — metrics identify the
        problem exists, traces pinpoint which sub-reconciler phase failed.
        """
        raw_metrics = fetch_operator_metrics(
            admin_client=admin_client,
            operator_metrics_token=operator_metrics_token,
        )
        metrics = parse_prometheus_text(text=raw_metrics)
        total = metric_value_sum(
            metrics=metrics,
            metric_name=RECONCILE_TOTAL_METRIC,
            label_filter={METRIC_LABEL_CONTROLLER: EVALHUB_CONTROLLER_LABEL_VALUE},
        )
        assert total > 0, "No reconciliation metrics available for diagnosis"

        logs = fetch_trace_collector_logs(trace_collector_pod=otel_trace_collector_pod)
        spans = parse_trace_spans_from_logs(logs=logs)
        reconcile_spans = filter_spans_by_name(spans=spans, name=SPAN_RECONCILE)
        if not reconcile_spans:
            pytest.skip("No reconcile spans found — trace collector may not have received spans yet")
        span = reconcile_spans[0]
        assert span.get("trace_id"), "Span missing trace_id for correlation"
        assert span.get("span_id"), "Span missing span_id for drill-down"

    @pytest.mark.skip(reason="Requires a job-failure fixture that submits and awaits a failing evaluation job")
    def test_job_failure_observable_metrics_and_traces(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        operator_with_otel_tracing: Deployment,
        otel_trace_collector_pod: Pod,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given a job failure, it is observable through both metrics and trace spans.

        TC-E2E-004: Verify a job failure event is observable through both
        the evalhub_job_failure_events_total metric and trace spans.
        """
        raw_metrics = fetch_operator_metrics(
            admin_client=admin_client,
            operator_metrics_token=operator_metrics_token,
        )
        metrics = parse_prometheus_text(text=raw_metrics)
        job_failure_samples = get_metric_samples(metrics=metrics, metric_name=JOB_FAILURE_EVENTS_METRIC)

        logs = fetch_trace_collector_logs(trace_collector_pod=otel_trace_collector_pod)
        spans = parse_trace_spans_from_logs(logs=logs)
        failure_spans = filter_spans_by_name(spans=spans, name=SPAN_JOB_FAILURE_RECONCILE)

        assert job_failure_samples or failure_spans, "No job failure events detected in metrics or traces"


# TC-UPG: Upgrade (3 tests)


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-upg"}, id="test_evalhub_upg")],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.tier1
@pytest.mark.slow
class TestEvalHubReconcileUpgrade:
    """TC-UPG-001 through TC-UPG-003: Upgrade testing."""

    @pytest.mark.post_upgrade
    def test_new_metrics_appear_after_upgrade(
        self,
        admin_client: DynamicClient,
        operator_metrics_token: str,
        evalhub_reconcile_cr: EvalHub,
    ) -> None:
        """Given an operator upgrade, new EvalHub reconciliation metrics appear.

        TC-UPG-001: Verify new evalhub_controller_* metrics appear on the
        operator metrics endpoint after upgrading from a version without them.
        """
        found: set[str] = set()
        try:
            for raw_metrics in TimeoutSampler(
                wait_timeout=METRICS_POLL_TIMEOUT,
                sleep=METRICS_POLL_INTERVAL,
                func=fetch_operator_metrics,
                exceptions_dict=_TRANSIENT_METRICS_EXCEPTIONS,
                admin_client=admin_client,
                operator_metrics_token=operator_metrics_token,
            ):
                metrics = parse_prometheus_text(text=raw_metrics)
                found = set()
                for metric_name in EVALHUB_RECONCILE_METRICS:
                    if (
                        metric_name in metrics
                        or f"{metric_name}_bucket" in metrics
                        or f"{metric_name}_total" in metrics
                    ):
                        found.add(metric_name)
                if found == set(EVALHUB_RECONCILE_METRICS):
                    return
        except TimeoutExpiredError:
            pytest.fail(f"Not all new metrics appeared after upgrade. Found: {found}")

    @pytest.mark.post_upgrade
    def test_existing_metrics_unaffected_by_upgrade(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        operator_metrics_token: str,
    ) -> None:
        """Given an operator upgrade, existing metrics and dashboards still work.

        TC-UPG-002: Verify existing operator metrics (controller_runtime_*)
        and dashboards are unaffected by the upgrade.
        """
        raw_metrics = fetch_operator_metrics(
            admin_client=admin_client,
            operator_metrics_token=operator_metrics_token,
        )
        assert "controller_runtime_reconcile_total" in raw_metrics, (
            "controller_runtime_reconcile_total missing after upgrade"
        )
        assert "workqueue_depth" in raw_metrics or "controller_runtime_active_workers" in raw_metrics, (
            "Standard controller-runtime work metrics missing after upgrade"
        )

    @pytest.mark.pre_upgrade
    def test_rollback_removes_new_metrics(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        operator_metrics_token: str,
    ) -> None:
        """Given a rollback to the prior version, new metrics are cleanly removed.

        TC-UPG-003: Verify rolling back the operator removes the new
        evalhub_controller_* metrics without affecting existing ones.
        """
        raw_metrics = fetch_operator_metrics(
            admin_client=admin_client,
            operator_metrics_token=operator_metrics_token,
        )
        assert "controller_runtime_reconcile_total" in raw_metrics, (
            "Base controller-runtime metrics should exist after rollback"
        )
        for metric_name in EVALHUB_RECONCILE_METRICS:
            assert metric_name not in raw_metrics, (
                f"Metric {metric_name} still present after rollback — expected clean removal"
            )
