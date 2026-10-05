"""Raw Prometheus and Thanos query evidence client."""

import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import requests

from tests.observability.contract import ContractRecord, render_promql

SENSITIVE_PARAMETER_NAMES = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "cookie",
    "password",
    "secret",
    "token",
}


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
