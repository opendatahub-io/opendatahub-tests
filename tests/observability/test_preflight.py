import pytest

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
