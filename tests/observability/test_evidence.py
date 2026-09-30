import json
from dataclasses import replace

import pytest

from tests.observability.evidence import EvidenceRecord, write_evidence, write_failure_log
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
    )

    written = destination.read_text()
    assert "secret-token" not in written
    payload = json.loads(written)
    assert payload["records"][0]["query"]["http_status"] == 200
    assert payload["records"][0]["query"]["query_params"]["bearer_token"] == "[REDACTED]"
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
