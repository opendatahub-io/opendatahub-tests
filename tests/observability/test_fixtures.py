from dataclasses import replace

import pytest

from tests.observability.fixtures import source_metric_available, summarize_traffic, wait_for_source_metric
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
