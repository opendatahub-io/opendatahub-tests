"""Raw Prometheus and signal-specific telemetry query evidence clients."""

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlsplit

import requests

from tests.observability.contract import ContractRecord, render_promql

SENSITIVE_PARAMETER_NAMES = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "client_secret",
    "cookie",
    "password",
    "refresh_token",
    "secret",
    "token",
}
SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?i)\b(?:bearer|basic)\s+[^\s,;]+|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*"
)
SENSITIVE_KEY_VALUE_PATTERN = re.compile(
    r"(?i)\b((?:[a-z0-9]+[_-])*(?:api[_-]?key|authorization|bearer|cookie|password|secret|token))"
    r"\s*[=:]\s*[^\s,;&]+"
)


@dataclass(frozen=True)
class NormalizedSeries:
    """Stable representation of one Prometheus result series."""

    labels: tuple[tuple[str, str], ...]
    values: tuple[tuple[float, str], ...]


@dataclass(frozen=True)
class QueryRequest:
    """Inputs needed to issue one route-level Prometheus query."""

    persona: str
    principal: str
    requested_namespace: str | None
    fixture_namespace: str | None
    datasource: str
    route: str
    promql: str
    query_params: dict[str, str | int | float]
    expected_disposition: str
    method: str = "GET"
    time_range: dict[str, str | int | float] | None = None
    signal: str = "metrics"
    marker: str = ""
    rollout_state: str = "passed"
    failure_category: str | None = None


@dataclass(frozen=True)
class RawQueryResult:
    """Complete sanitized HTTP and Prometheus response contract."""

    persona: str
    principal: str
    requested_namespace: str | None
    fixture_namespace: str | None
    datasource: str
    url_path: str
    http_method: str
    http_status: int | None
    response_time_ms: float
    query_params: dict[str, str | int | float]
    promql: str
    time_range: dict[str, str | int | float]
    prometheus_status: str | None
    error_type: str | None
    error: str | None
    warnings: tuple[str, ...]
    result_type: str | None
    series: tuple[NormalizedSeries, ...]
    expected_disposition: str
    transport_error: str | None = None
    parse_error: str | None = None
    signal: str = "metrics"
    marker: str = ""
    rollout_state: str = "passed"
    failure_category: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return sanitized machine-readable evidence."""
        return {
            "persona": self.persona,
            "principal": self.principal,
            "requested_namespace": self.requested_namespace,
            "fixture_namespace": self.fixture_namespace,
            "datasource": self.datasource,
            "url_path": self.url_path,
            "http_method": self.http_method,
            "http_status": self.http_status,
            "response_time_ms": self.response_time_ms,
            "query_params": _redact_mapping(self.query_params),
            "promql": self.promql,
            "time_range": dict(self.time_range),
            "prometheus_status": self.prometheus_status,
            "error_type": self.error_type,
            "error": self.error,
            "warnings": list(self.warnings),
            "result_type": self.result_type,
            "series": [
                {"labels": dict(series.labels), "values": [list(value) for value in series.values]}
                for series in self.series
            ],
            "expected_disposition": self.expected_disposition,
            "transport_error": self.transport_error,
            "parse_error": self.parse_error,
            "signal": self.signal,
            "marker": self.marker,
            "rollout_state": self.rollout_state,
            "failure_category": self.failure_category,
        }


class RawQueryClient:
    """Issue raw queries through a configured dashboard datasource route."""

    def __init__(
        self,
        base_url: str,
        session: requests.Session | Any | None = None,
        timeout: float = 30.0,
        verify: bool | str = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.verify = verify

    def query(self, request: QueryRequest, bearer_token: str | None = None) -> RawQueryResult:
        """Execute a query while retaining transport and Prometheus response details."""
        route_parts = urlsplit(url=request.route)
        url_path = route_parts.path or "/"
        url = (
            f"{route_parts.scheme}://{route_parts.netloc}{url_path}"
            if route_parts.scheme
            else f"{self.base_url}{url_path}"
        )
        params: dict[str, str | int | float] = dict(request.query_params)
        params.setdefault("query", request.promql)
        if request.time_range:
            params.update(request.time_range)
        headers = {"Accept": "application/json"}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"

        started = time.perf_counter()
        try:
            response = self.session.request(
                method=request.method,
                url=url,
                params=params,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify,
            )
        except (OSError, requests.RequestException) as error:
            elapsed = _elapsed_ms(started=started)
            return RawQueryResult(
                persona=request.persona,
                principal=request.principal,
                requested_namespace=request.requested_namespace,
                fixture_namespace=request.fixture_namespace,
                datasource=request.datasource,
                url_path=url_path,
                http_method=request.method,
                http_status=None,
                response_time_ms=elapsed,
                query_params=_redact_mapping(params),
                promql=request.promql,
                time_range=dict(request.time_range or {}),
                prometheus_status=None,
                error_type=None,
                error=None,
                warnings=(),
                result_type=None,
                series=(),
                expected_disposition=request.expected_disposition,
                transport_error=type(error).__name__,
                signal=request.signal,
                marker=request.marker,
                rollout_state=request.rollout_state,
                failure_category=request.failure_category,
            )

        elapsed = _elapsed_ms(started=started)
        try:
            payload = response.json()
        except (TypeError, ValueError) as error:
            return RawQueryResult(
                persona=request.persona,
                principal=request.principal,
                requested_namespace=request.requested_namespace,
                fixture_namespace=request.fixture_namespace,
                datasource=request.datasource,
                url_path=url_path,
                http_method=request.method,
                http_status=response.status_code,
                response_time_ms=elapsed,
                query_params=_redact_mapping(params),
                promql=request.promql,
                time_range=dict(request.time_range or {}),
                prometheus_status=None,
                error_type=None,
                error=None,
                warnings=(),
                result_type=None,
                series=(),
                expected_disposition=request.expected_disposition,
                parse_error=type(error).__name__,
                signal=request.signal,
                marker=request.marker,
                rollout_state=request.rollout_state,
                failure_category=request.failure_category,
            )

        if not isinstance(payload, dict):
            payload = {}
        raw_data = payload.get("data")
        data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
        raw_result_value = data.get("result")
        raw_result = raw_result_value if isinstance(raw_result_value, list) else []
        raw_warnings = payload.get("warnings")
        warnings = raw_warnings if isinstance(raw_warnings, list) else []
        return RawQueryResult(
            persona=request.persona,
            principal=request.principal,
            requested_namespace=request.requested_namespace,
            fixture_namespace=request.fixture_namespace,
            datasource=request.datasource,
            url_path=url_path,
            http_method=request.method,
            http_status=response.status_code,
            response_time_ms=elapsed,
            query_params=_redact_mapping(params),
            promql=request.promql,
            time_range=dict(request.time_range or {}),
            prometheus_status=_optional_string(payload.get("status")),
            error_type=_optional_string(payload.get("errorType")),
            error=_optional_string(payload.get("error")),
            warnings=tuple(str(warning) for warning in warnings if warning is not None),
            result_type=_optional_string(data.get("resultType")),
            series=tuple(_normalize_series(item) for item in raw_result if isinstance(item, dict)),
            expected_disposition=request.expected_disposition,
            signal=request.signal,
            marker=request.marker,
            rollout_state=request.rollout_state,
            failure_category=request.failure_category,
        )


def build_contract_request(
    contract: ContractRecord,
    persona: str,
    principal: str,
    requested_namespace: str | None,
    fixture_namespace: str | None,
    variables: dict[str, str],
    query_params: dict[str, str | int | float] | None = None,
) -> QueryRequest:
    """Build a route query directly from one reviewed contract record."""
    params = dict(query_params or {})
    if requested_namespace is not None and contract.datasource in {"namespace-proxy", "tenancy"}:
        params.setdefault("namespace", requested_namespace)
    return QueryRequest(
        persona=persona,
        principal=principal,
        requested_namespace=requested_namespace,
        fixture_namespace=fixture_namespace,
        datasource=contract.datasource,
        route=contract.route,
        promql=render_promql(contract.promql, **variables),
        query_params=params,
        expected_disposition="pass" if contract.capability == "shipped" else contract.capability,
        time_range=contract.time_range,
    )


def _normalize_series(result: dict[str, Any]) -> NormalizedSeries:
    raw_labels = result.get("metric")
    labels = (
        tuple(sorted((str(key), str(value)) for key, value in raw_labels.items()))
        if isinstance(raw_labels, dict)
        else ()
    )
    raw_values = result.get("values")
    if raw_values is None:
        raw_values = [result.get("value")] if result.get("value") is not None else []
    values: list[tuple[float, str]] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
            continue
        try:
            timestamp = round(float(raw_value[0]), 3)
        except (TypeError, ValueError) as conversion_error:
            _ = conversion_error
            continue
        values.append((timestamp, str(raw_value[1])))
    return NormalizedSeries(labels=labels, values=tuple(values))


def _redact_mapping(values: dict[str, str | int | float]) -> dict[str, str | int | float]:
    return {key: "[REDACTED]" if key.lower() in SENSITIVE_PARAMETER_NAMES else value for key, value in values.items()}


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


@dataclass(frozen=True)
class SignalQueryRequest:
    """Inputs for a sanitized Tempo, Loki, or Korrel8r request."""

    signal: str
    marker: str
    persona: str
    principal: str
    requested_namespace: str | None
    fixture_namespace: str | None
    route: str
    query_params: dict[str, str | int | float] = field(default_factory=dict)
    method: str = "GET"
    body: Mapping[str, object] | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    expected_disposition: str = "pass"
    rollout_state: str = "passed"
    failure_category: str | None = None


@dataclass(frozen=True)
class SignalQueryResult:
    """Sanitized response summary shared by all non-PromQL signal adapters."""

    signal: str
    marker: str
    persona: str
    principal: str
    requested_namespace: str | None
    fixture_namespace: str | None
    url_path: str
    http_method: str
    http_status: int | None
    response_time_ms: float
    query_params: dict[str, str | int | float]
    request_summary: dict[str, object]
    backend_status: str | None
    backend_error_type: str | None
    backend_error: str | None
    result_count: int
    labels: tuple[tuple[str, str], ...]
    attributes: tuple[tuple[str, str], ...]
    edges: tuple[str, ...]
    expected_disposition: str
    rollout_state: str
    failure_category: str | None = None
    transport_error: str | None = None
    parse_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return only normalized response data suitable for release evidence."""
        return {
            "signal": self.signal,
            "marker": self.marker,
            "persona": self.persona,
            "principal": self.principal,
            "requested_namespace": self.requested_namespace,
            "fixture_namespace": self.fixture_namespace,
            "url_path": self.url_path,
            "http_method": self.http_method,
            "http_status": self.http_status,
            "response_time_ms": self.response_time_ms,
            "query_params": _redact_mapping(self.query_params),
            "request_summary": _redact_object(self.request_summary),
            "backend_status": self.backend_status,
            "backend_error_type": self.backend_error_type,
            "backend_error": _sanitize_signal_text(value=self.backend_error),
            "result_count": self.result_count,
            "labels": dict(self.labels),
            "attributes": dict(self.attributes),
            "edges": list(self.edges),
            "expected_disposition": self.expected_disposition,
            "rollout_state": self.rollout_state,
            "failure_category": self.failure_category,
            "transport_error": self.transport_error,
            "parse_error": self.parse_error,
        }


class SignalQueryClient:
    """Issue a signal query without retaining raw response bodies or credentials."""

    def __init__(
        self,
        base_url: str,
        session: requests.Session | Any | None = None,
        timeout: float = 30.0,
        verify: bool | str = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.verify = verify

    def query(self, request: SignalQueryRequest, bearer_token: str | None = None) -> SignalQueryResult:
        """Execute one signal request and normalize only safe response metadata."""
        route_parts = urlsplit(url=request.route)
        url_path = route_parts.path or "/"
        url = (
            f"{route_parts.scheme}://{route_parts.netloc}{url_path}"
            if route_parts.scheme
            else f"{self.base_url}{url_path}"
        )
        params = dict(request.query_params)
        headers = {str(key): str(value) for key, value in request.headers.items()}
        headers.setdefault("Accept", "application/json")
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        started = time.perf_counter()
        try:
            response = self.session.request(
                method=request.method,
                url=url,
                params=params,
                json=request.body if request.method.upper() != "GET" else None,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify,
            )
        except (OSError, requests.RequestException) as error:
            return _signal_result(
                request=request,
                url_path=url_path,
                elapsed_ms=_elapsed_ms(started=started),
                http_status=None,
                payload={},
                request_summary=_request_summary(request),
                transport_error=type(error).__name__,
            )

        elapsed = _elapsed_ms(started=started)
        try:
            payload: object = response.json()
        except (TypeError, ValueError) as error:
            return _signal_result(
                request=request,
                url_path=url_path,
                elapsed_ms=elapsed,
                http_status=response.status_code,
                payload={},
                request_summary=_request_summary(request),
                parse_error=type(error).__name__,
            )
        return _signal_result(
            request=request,
            url_path=url_path,
            elapsed_ms=elapsed,
            http_status=response.status_code,
            payload=payload,
            request_summary=_request_summary(request),
        )


class TempoQueryClient(SignalQueryClient):
    """Adapter for the stable Tempo search and trace retrieval API."""

    def search(self, request: SignalQueryRequest, bearer_token: str | None = None) -> SignalQueryResult:
        """Search Tempo through ``GET /api/search``."""
        _require_get(request=request)
        return self.query(request=_replace_route(request=request, route="/api/search"), bearer_token=bearer_token)

    def trace(self, request: SignalQueryRequest, trace_id: str, bearer_token: str | None = None) -> SignalQueryResult:
        """Retrieve one Tempo trace through ``GET /api/traces/{traceID}``."""
        _require_get(request=request)
        path = f"/api/traces/{quote(trace_id, safe='')}"
        return self.query(request=_replace_route(request=request, route=path), bearer_token=bearer_token)


class LokiQueryClient(SignalQueryClient):
    """Adapter for the application-tenant Loki range query endpoint."""

    def query_range(
        self, request: SignalQueryRequest, tenant: str = "application", bearer_token: str | None = None
    ) -> SignalQueryResult:
        """Query Loki through ``GET /loki/api/v1/query_range`` with a tenant header."""
        _require_get(request=request)
        if not tenant.strip():
            raise ValueError("Loki tenant must not be empty")
        routed = _replace_route(
            request=_replace_headers(request=request, headers={"X-Scope-OrgID": tenant}),
            route="/loki/api/v1/query_range",
        )
        return self.query(request=routed, bearer_token=bearer_token)


class Korrel8rQueryClient(SignalQueryClient):
    """Adapter for bounded in-cluster Korrel8r discovery and graph requests."""

    def config(self, request: SignalQueryRequest, bearer_token: str | None = None) -> SignalQueryResult:
        """Probe the unsupported configuration endpoint so callers can assert its 404 contract."""
        _require_get(request=request)
        return self.query(
            request=_replace_route(request=request, route="/api/v1alpha1/config"),
            bearer_token=bearer_token,
        )

    def domains(self, request: SignalQueryRequest, bearer_token: str | None = None) -> SignalQueryResult:
        """Discover configured Korrel8r domains."""
        _require_get(request=request)
        return self.query(
            request=_replace_route(request=request, route="/api/v1alpha1/domains"),
            bearer_token=bearer_token,
        )

    def objects(self, request: SignalQueryRequest, bearer_token: str | None = None) -> SignalQueryResult:
        """Resolve bounded object references."""
        _require_get(request=request)
        return self.query(
            request=_replace_route(request=request, route="/api/v1alpha1/objects"),
            bearer_token=bearer_token,
        )

    def graphs_goals(
        self,
        request: SignalQueryRequest,
        bearer_token: str | None = None,
        max_depth: int = 3,
        max_nodes: int = 100,
        max_edges: int = 200,
    ) -> SignalQueryResult:
        """Run a bounded goal traversal and reject unbounded caller limits."""
        _require_get(request=request)
        if not 1 <= max_depth <= 3:
            raise ValueError("Korrel8r max_depth must be between 1 and 3")
        if not 1 <= max_nodes <= 500 or not 1 <= max_edges <= 1000:
            raise ValueError("Korrel8r graph limits are outside the supported bounds")
        params = dict(request.query_params)
        params.update({"maxDepth": max_depth, "maxNodes": max_nodes, "maxEdges": max_edges})
        return self.query(
            request=_replace_route(
                request=_replace_params(request=request, query_params=params), route="/api/v1alpha1/graphs/goals"
            ),
            bearer_token=bearer_token,
        )


def _signal_result(
    *,
    request: SignalQueryRequest,
    url_path: str,
    elapsed_ms: float,
    http_status: int | None,
    payload: object,
    request_summary: dict[str, object],
    transport_error: str | None = None,
    parse_error: str | None = None,
) -> SignalQueryResult:
    payload_mapping = payload if isinstance(payload, dict) else {"result": payload}
    data = payload_mapping.get("data") if isinstance(payload_mapping.get("data"), dict) else payload_mapping
    backend_error = payload_mapping.get("error")
    if backend_error is None and isinstance(data, dict):
        backend_error = data.get("error")
    backend_error_type = payload_mapping.get("errorType")
    if backend_error_type is None and isinstance(data, dict):
        backend_error_type = data.get("errorType")
    return SignalQueryResult(
        signal=request.signal,
        marker=request.marker,
        persona=request.persona,
        principal=request.principal,
        requested_namespace=request.requested_namespace,
        fixture_namespace=request.fixture_namespace,
        url_path=url_path,
        http_method=request.method,
        http_status=http_status,
        response_time_ms=elapsed_ms,
        query_params=_redact_mapping(values=dict(request.query_params)),
        request_summary=request_summary,
        backend_status=_string_value(payload_mapping.get("status")),
        backend_error_type=_string_value(backend_error_type),
        backend_error=_sanitize_signal_text(value=_string_value(backend_error)),
        result_count=_result_count(data),
        labels=_normalized_metadata(data, keys=("labels", "stream", "metric")),
        attributes=_normalized_metadata(data, keys=("attributes", "resource", "resourceAttributes")),
        edges=_normalized_edges(data),
        expected_disposition=request.expected_disposition,
        rollout_state=request.rollout_state,
        failure_category=request.failure_category,
        transport_error=transport_error,
        parse_error=parse_error,
    )


def _result_count(payload: object) -> int:
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, dict):
        return 0
    for key in ("result", "traces", "streams", "objects", "domains", "edges", "goals"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _normalized_metadata(payload: object, keys: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    pairs: dict[str, str] = {}
    if isinstance(payload, dict):
        candidates: list[object] = [payload]
        for key in ("result", "traces", "streams", "objects"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates.extend(value[:100])
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            for key in keys:
                values = candidate.get(key)
                if isinstance(values, dict):
                    for name, value in values.items():
                        if isinstance(name, str) and not _is_sensitive_name(name):
                            pairs[name] = _safe_text(value=value)
    return tuple(sorted(pairs.items()))


def _normalized_edges(payload: object) -> tuple[str, ...]:
    edge_names: set[str] = set()
    if isinstance(payload, dict):
        candidates = [payload.get("edges"), payload.get("result"), payload.get("traces"), payload.get("goals")]
        for candidate in candidates:
            if not isinstance(candidate, list):
                continue
            for item in candidate[:200]:
                if isinstance(item, str):
                    edge_names.add(_safe_text(value=item))
                elif isinstance(item, dict):
                    for key in ("edge", "name", "type", "kind", "relation"):
                        value = item.get(key)
                        if isinstance(value, str) and value:
                            edge_names.add(_safe_text(value=value))
    return tuple(sorted(edge_names))


def _request_summary(request: SignalQueryRequest) -> dict[str, object]:
    return {
        "method": request.method,
        "signal": request.signal,
        "body_keys": sorted(str(key) for key in (request.body or {})),
    }


def _require_get(request: SignalQueryRequest) -> None:
    if request.method.upper() != "GET":
        raise ValueError("this signal adapter only supports GET")


def _replace_route(request: SignalQueryRequest, route: str) -> SignalQueryRequest:
    return SignalQueryRequest(
        signal=request.signal,
        marker=request.marker,
        persona=request.persona,
        principal=request.principal,
        requested_namespace=request.requested_namespace,
        fixture_namespace=request.fixture_namespace,
        route=route,
        query_params=dict(request.query_params),
        method=request.method,
        body=request.body,
        headers=request.headers,
        expected_disposition=request.expected_disposition,
        rollout_state=request.rollout_state,
        failure_category=request.failure_category,
    )


def _replace_headers(request: SignalQueryRequest, headers: Mapping[str, str]) -> SignalQueryRequest:
    merged_headers = dict(request.headers)
    merged_headers.update(headers)
    return (
        _replace_route(request=request, route=request.route)
        if not merged_headers
        else SignalQueryRequest(
            signal=request.signal,
            marker=request.marker,
            persona=request.persona,
            principal=request.principal,
            requested_namespace=request.requested_namespace,
            fixture_namespace=request.fixture_namespace,
            route=request.route,
            query_params=dict(request.query_params),
            method=request.method,
            body=request.body,
            headers=merged_headers,
            expected_disposition=request.expected_disposition,
            rollout_state=request.rollout_state,
            failure_category=request.failure_category,
        )
    )


def _replace_params(request: SignalQueryRequest, query_params: dict[str, str | int | float]) -> SignalQueryRequest:
    return SignalQueryRequest(
        signal=request.signal,
        marker=request.marker,
        persona=request.persona,
        principal=request.principal,
        requested_namespace=request.requested_namespace,
        fixture_namespace=request.fixture_namespace,
        route=request.route,
        query_params=query_params,
        method=request.method,
        body=request.body,
        headers=request.headers,
        expected_disposition=request.expected_disposition,
        rollout_state=request.rollout_state,
        failure_category=request.failure_category,
    )


def _redact_object(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _is_sensitive_name(str(key)) else _redact_object(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_object(item) for item in value[:20]]
    if isinstance(value, str):
        return _sanitize_signal_text(value=value)
    return value


def _sanitize_signal_text(value: str | None) -> str | None:
    if value is None:
        return None
    redacted = SENSITIVE_VALUE_PATTERN.sub(repl="[REDACTED]", string=value)
    redacted = SENSITIVE_KEY_VALUE_PATTERN.sub(repl=r"\1=[REDACTED]", string=redacted)
    return redacted.replace("\r", r"\r").replace("\n", r"\n")[:500]


def _safe_text(value: object) -> str:
    return _sanitize_signal_text(value=str(_redact_object(value))) or ""


def _string_value(value: object) -> str | None:
    return _safe_text(value=value) if value is not None else None


def _is_sensitive_name(name: str) -> bool:
    lowered = name.casefold()
    return lowered in SENSITIVE_PARAMETER_NAMES or any(
        marker in lowered for marker in ("api_key", "authorization", "bearer", "cookie", "password", "secret", "token")
    )
