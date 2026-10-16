from dataclasses import replace
from datetime import UTC, datetime

import pytest

from tests.observability.fixtures import (
    build_marker_identity,
    source_metric_available,
    summarize_traffic,
    wait_for_source_metric,
)
from tests.observability.query import NormalizedSeries, RawQueryResult

pytestmark = pytest.mark.tier1


def test_summarize_traffic_keeps_success_and_rate_limited_counts_separate() -> None:
    """Given deterministic MaaS response statuses, preserve request and rate-limit totals independently."""
    summary = summarize_traffic(statuses=[200, 200, 429])

    assert summary.to_dict() == {
        "total_requests": 3,
        "successful_requests": 2,
        "rate_limited_requests": 1,
    }


def test_summarize_traffic_rejects_unclassified_responses() -> None:
    """Given an unexpected gateway response, fail instead of misclassifying usage evidence."""
    with pytest.raises(ValueError, match="unexpected"):
        summarize_traffic(statuses=[200, 500])


def test_source_metric_wait_requires_success_and_a_series() -> None:
    """Given an empty success followed by source telemetry, return only after a series is present."""
    empty = RawQueryResult(
        persona="admin",
        principal="admin",
        requested_namespace="ns-a",
        fixture_namespace="ns-a",
        datasource="thanos",
        url_path="/api/v1/query",
        http_method="GET",
        http_status=200,
        response_time_ms=1.0,
        query_params={},
        promql="up",
        time_range={},
        prometheus_status="success",
        error_type=None,
        error=None,
        warnings=(),
        result_type="vector",
        series=(),
        expected_disposition="pass",
    )
    source = replace(
        empty,
        series=(NormalizedSeries(labels=(("namespace", "ns-a"),), values=((1.0, "1"),)),),
    )
    results = iter((empty, source))

    assert source_metric_available(result=source)
    assert wait_for_source_metric(query=lambda: next(results), timeout=2, sleep=0) is source


def test_marker_identity_uses_live_resource_tuple_and_bounded_window() -> None:
    """Given a workload identity and emission timestamp, produce stable correlation metadata without instrumentation."""
    emitted_at = datetime(year=2026, month=1, day=1, hour=12, minute=0, tzinfo=UTC)

    marker = build_marker_identity(
        run_id="run-1",
        signal="metrics",
        namespace="ns-a",
        workload="model-a",
        pod="model-a-123",
        pod_uid="pod-uid",
        emitted_at=emitted_at,
        query_window_seconds=30,
    )

    assert marker.correlation_id == "rhoai-observability/run-1"
    assert (
        marker.marker_id
        == build_marker_identity(
            run_id="run-1",
            signal="metrics",
            namespace="ns-a",
            workload="model-a",
            pod="model-a-123",
            pod_uid="pod-uid",
            emitted_at=emitted_at,
            query_window_seconds=30,
        ).marker_id
    )
    assert marker.query_window_start.endswith("11:59:30+00:00")
    assert marker.query_window_end.endswith("12:00:30+00:00")


def test_marker_identity_changes_for_distinct_emissions_and_pods() -> None:
    """Given the same workload, distinguish emissions by their window and live pod identity."""
    emitted_at = datetime(year=2026, month=1, day=1, hour=12, minute=0, tzinfo=UTC)
    marker = build_marker_identity(
        run_id="run-1",
        signal="metrics",
        namespace="ns-a",
        workload="model-a",
        pod="model-a-123",
        pod_uid="pod-uid",
        emitted_at=emitted_at,
    )

    assert (
        marker.marker_id
        != build_marker_identity(
            run_id="run-1",
            signal="metrics",
            namespace="ns-a",
            workload="model-a",
            pod="model-a-123",
            pod_uid="replacement-pod-uid",
            emitted_at=emitted_at,
        ).marker_id
    )
    assert (
        marker.marker_id
        != build_marker_identity(
            run_id="run-1",
            signal="metrics",
            namespace="ns-a",
            workload="model-a",
            pod="model-a-123",
            pod_uid="pod-uid",
            emitted_at=emitted_at.replace(minute=1),
        ).marker_id
    )
