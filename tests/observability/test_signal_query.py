from unittest.mock import Mock

import pytest

from tests.observability.query import (
    Korrel8rQueryClient,
    LokiQueryClient,
    SignalQueryRequest,
    TempoQueryClient,
)

pytestmark = pytest.mark.tier1


def _request() -> SignalQueryRequest:
    return SignalQueryRequest(
        signal="traces",
        marker="marker-1",
        persona="cluster-admin",
        principal="admin",
        requested_namespace="ns-a",
        fixture_namespace="ns-a",
        route="/ignored-by-adapter",
        query_params={"namespace": "ns-a", "token": "secret-token"},
    )


def test_tempo_search_normalizes_results_and_never_records_bearer_token() -> None:
    """Given a Tempo search response, retain normalized attributes and the stable API path only."""
    response = Mock(status_code=200)
    response.json.return_value = {
        "traces": [{"traceID": "abc", "resource": {"service.name": "model"}, "spanSets": []}],
        "status": "success",
    }
    session = Mock()
    session.request.return_value = response
    client = TempoQueryClient(base_url="https://tempo.example", session=session)

    result = client.search(request=_request(), bearer_token="secret-token")

    assert result.url_path == "/api/search"
    assert result.result_count == 1
    assert result.attributes == (("service.name", "model"),)
    assert "secret-token" not in str(result.to_dict())
    assert session.request.call_args.kwargs["headers"]["Authorization"] == "Bearer secret-token"


def test_tempo_trace_quotes_trace_id_in_path() -> None:
    """Given a trace identifier, quote it before constructing the Tempo retrieval path."""
    response = Mock(status_code=200)
    response.json.return_value = {"trace": {"traceID": "abc/unsafe"}}
    session = Mock()
    session.request.return_value = response
    client = TempoQueryClient(base_url="https://tempo.example", session=session)

    result = client.trace(request=_request(), trace_id="abc/unsafe")

    assert result.url_path == "/api/traces/abc%2Funsafe"


def test_loki_query_range_sets_application_tenant_without_evidence_header() -> None:
    """Given a Loki range request, send the documented application tenant without exposing headers in evidence."""
    response = Mock(status_code=200)
    response.json.return_value = {"status": "success", "data": {"result": [{"stream": {"namespace": "ns-a"}}]}}
    session = Mock()
    session.request.return_value = response
    client = LokiQueryClient(base_url="https://loki.example", session=session)

    result = client.query_range(request=_request(), tenant="application")

    assert result.url_path == "/loki/api/v1/query_range"
    assert result.result_count == 1
    assert "X-Scope-OrgID" not in str(result.to_dict())
    assert session.request.call_args.kwargs["headers"]["X-Scope-OrgID"] == "application"


def test_korrel8r_graph_limits_are_bounded() -> None:
    """Given an excessive graph depth, reject an unbounded Korrel8r traversal before issuing a request."""
    client = Korrel8rQueryClient(base_url="https://korrel8r.example", session=Mock())

    with pytest.raises(ValueError, match="max_depth"):
        client.graphs_goals(request=_request(), max_depth=4)


def test_http_success_with_backend_error_is_retained_as_error_evidence() -> None:
    """Given HTTP 200 with a backend error payload, retain that error instead of treating it as success."""
    response = Mock(status_code=200)
    response.json.return_value = {"status": "error", "errorType": "bad_data", "error": "invalid query"}
    session = Mock()
    session.request.return_value = response
    client = TempoQueryClient(base_url="https://tempo.example", session=session)

    result = client.search(request=_request())

    assert result.http_status == 200
    assert result.backend_status == "error"
    assert result.backend_error_type == "bad_data"
    assert result.backend_error == "invalid query"
