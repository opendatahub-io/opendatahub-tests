from typing import Any

import pytest

from tests.observability.contract import load_release_contract
from tests.observability.handoff import dashboard_handoff_preflight_check, load_dashboard_metadata
from tests.observability.preflight import (
    PreflightCheck,
    PreflightDisposition,
    authorization_preflight_checks,
    evaluate_preflight,
    parse_preflight_present,
)

pytestmark = pytest.mark.tier1


def test_preflight_is_ready_when_all_prerequisites_are_present() -> None:
    """Given all selected prerequisites are present, report a ready release gate."""
    report = evaluate_preflight(
        checks=[
            PreflightCheck(name="dashboard", present=True, category="product"),
            PreflightCheck(name="gpu", present=True, category="environment"),
        ]
    )

    assert report.disposition is PreflightDisposition.READY
    assert report.blocked == ()
    assert report.failed == ()


def test_preflight_distinguishes_environment_blocking_from_product_failure() -> None:
    """Given missing external capacity and a broken claimed feature, report both dispositions separately."""
    report = evaluate_preflight(
        checks=[
            PreflightCheck(name="gpu", present=False, category="environment", detail="no schedulable GPU"),
            PreflightCheck(name="route", present=False, category="product", detail="route was expected"),
        ]
    )

    assert report.disposition is PreflightDisposition.FAILED
    assert report.blocked == ("gpu",)
    assert report.failed == ("route",)
    assert "no schedulable GPU" in report.details["gpu"]


def test_preflight_rejects_unknown_check_categories() -> None:
    """Given an invalid prerequisite category, reject an ambiguous release outcome."""
    with pytest.raises(ValueError, match="category"):
        PreflightCheck(name="route", present=False, category="unknown")


def test_preflight_rejects_string_boolean_values() -> None:
    """Given a string that resembles a boolean, reject it instead of treating any non-empty value as true."""
    with pytest.raises(TypeError, match="JSON boolean"):
        parse_preflight_present(value="false")


def test_preflight_marks_unreviewed_authorization_as_product_failure() -> None:
    """Given a contract with unresolved authorization behavior, fail before mutating cluster fixtures."""
    checks = authorization_preflight_checks(
        records=[
            ("namespace-proxy", "review-required"),
            ("cluster-system-health", "not-applicable"),
        ]
    )

    report = evaluate_preflight(checks=checks)

    assert report.disposition is PreflightDisposition.FAILED
    assert report.failed == ("authorization-contract:namespace-proxy",)


def test_preflight_marks_conflicting_authorization_handoff_as_product_failure() -> None:
    """Given conflicting reviewed authorization responses, fail handoff preflight before resource creation."""
    contract = load_release_contract(source=_conflicting_contract())
    metadata = load_dashboard_metadata(source=_conflicting_metadata())

    handoff_check = dashboard_handoff_preflight_check(contract=contract, metadata=metadata)
    report = evaluate_preflight(checks=[handoff_check])

    assert handoff_check.present is False
    assert "disagree" in handoff_check.detail
    assert "dashboard-handoff-configuration" in report.failed


def _conflicting_contract() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "dashboard": "cluster",
        "datasource": "cluster-thanos",
        "route": "/api/v1/query",
        "promql": "up",
        "expected_http_status": [200],
        "expected_prometheus_status": "success",
        "expected_result_type": "vector",
        "minimum_series": 0,
        "required_labels": [],
        "empty_result_valid": True,
        "empty_ui_state": "Zero",
        "capability": "shipped",
    }
    records = [
        {**defaults, "id": "first", "panel": "first", "authorization_response": "403"},
        {**defaults, "id": "second", "panel": "second", "authorization_response": "404"},
    ]
    return {
        "contract_version": "1.0.0",
        "release_stage": "GA",
        "product_versions": {"dashboard": "2.0.0"},
        "records": records,
    }


def _conflicting_metadata() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "dashboards": [
            {
                "name": "cluster",
                "display_name": "Cluster",
                "panels": [
                    {"id": "first", "display_name": "First"},
                    {"id": "second", "display_name": "Second"},
                ],
            }
        ],
    }
