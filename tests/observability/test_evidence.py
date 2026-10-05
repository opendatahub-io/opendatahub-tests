import base64
import json
from dataclasses import replace

import pytest

from tests.observability.evidence import EvidenceRecord, _sanitize, write_evidence, write_failure_log
from tests.observability.query import QueryRequest, RawQueryClient

pytestmark = pytest.mark.tier1


def test_evidence_writer_redacts_secrets_and_writes_machine_readable_json(tmp_path) -> None:
    """Given query evidence containing secret-like values, write JSON without exposing those values."""
    response = type(
        "Response",
        (),
        {
            "status_code": 200,
            "json": lambda _self: {
                "status": "success",
                "data": {"resultType": "vector", "result": []},
            },
        },
    )()
    session = type("Session", (), {"request": lambda _self, **_kwargs: response})()
    result = RawQueryClient(base_url="https://metrics.example", session=session).query(
        request=QueryRequest(
            persona="cluster-admin",
            principal="admin",
            requested_namespace=None,
            fixture_namespace="ns-a",
            datasource="cluster-thanos",
            route="/api/v1/query?token=secret-token",
            promql="up",
            query_params={"namespace": "ns-a", "bearer_token": "secret-token"},
            expected_disposition="pass",
        ),
        bearer_token="secret-token",
    )
    destination = tmp_path / "evidence.json"

    write_evidence(
        destination=destination,
        records=[
            EvidenceRecord(
                tracking_id="RHOAIENG-96476",
                test_identifier="test_query",
                release_stage="GA",
                component_versions={"rhoai": "test"},
                cluster_run_id="run-1",
                persona={"name": "cluster-admin", "principal": "admin"},
                fixture_resources={"namespace": "ns-a"},
                query=result,
            )
        ],
        handoff={
            "dashboard_handoff": {
                "path": str(tmp_path / "observability-dashboard-contract.json"),
                "schema_version": "1.0.0",
                "record_count": 1,
            }
        },
    )

    written = destination.read_text()
    assert "secret-token" not in written
    payload = json.loads(written)
    assert payload["records"][0]["query"]["http_status"] == 200
    assert payload["records"][0]["query"]["query_params"]["bearer_token"] == "[REDACTED]"
    assert payload["handoff"]["dashboard_handoff"]["schema_version"] == "1.0.0"
    assert payload["handoff"]["dashboard_handoff"]["record_count"] == 1
    failure_record = EvidenceRecord(
        tracking_id="RHOAIENG-96476",
        test_identifier="test_query",
        release_stage="GA",
        component_versions={"rhoai": "test"},
        cluster_run_id="run-1",
        persona={"name": "cluster-admin", "principal": "admin"},
        fixture_resources={"namespace": "ns-a"},
        query=replace(result, error="Bearer secret-token"),
        failure_category="query-error",
    )
    failure_log = write_failure_log(destination=tmp_path / "failure.log", records=[failure_record])
    assert "secret-token" not in failure_log.read_text()
    assert "error=[REDACTED]" in failure_log.read_text()

    jwt_header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').decode().rstrip("=")
    jwt_payload = base64.urlsafe_b64encode(b'{"sub":"123"}').decode().rstrip("=")
    jwt_value = f"{jwt_header}.{jwt_payload}.signature"
    redacted = _sanitize(
        value=(
            f"Basic dXNlcjpwYXNz jwt={jwt_value} password=secret-value api_key:another-secret "
            "access_token=access-value refresh_token=refresh-value client_secret=client-value"
        )
    )
    assert "Basic [REDACTED]" in redacted
    assert "password=[REDACTED]" in redacted
    assert "api_key=[REDACTED]" in redacted
    assert "access_token=[REDACTED]" in redacted
    assert "refresh_token=[REDACTED]" in redacted
    assert "client_secret=[REDACTED]" in redacted
    assert jwt_value not in redacted

    controlled_record = replace(
        failure_record,
        test_identifier="test_query\nforged-line",
        query=replace(result, error_type="error\r\nforged-type"),
    )
    controlled_log = write_failure_log(destination=tmp_path / "controlled-failure.log", records=[controlled_record])
    controlled_text = controlled_log.read_text()
    assert controlled_text.count("\n") == 1
    assert "test_query\\nforged-line" in controlled_text
    assert "error_type=error\\r\\nforged-type" in controlled_text
