import pytest

from tests.observability.run_id import ResolvedRunID, normalize_run_id, resolve_run_id

pytestmark = pytest.mark.tier1


def test_run_id_prefers_explicit_value_and_normalizes_marker_text() -> None:
    """Given explicit and CI identifiers, choose the explicit value and bound its safe form."""
    resolved = resolve_run_id(
        environment={
            "RHOAI_OBSERVABILITY_RUN_ID": "Release_Run/ABC with spaces and " + "x" * 100,
            "CI_JOB_ID": "ci-value",
        }
    )

    assert resolved.source == "explicit"
    assert resolved.value.startswith("release-run-abc-with-spaces-and-")
    assert len(resolved.value) == 63
    assert resolved.value == resolved.value.casefold()


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        pytest.param({"CI_JOB_ID": "job-1", "BUILD_ID": "build-1"}, "job-1", id="test_ci_job_id_precedence"),
        pytest.param({"BUILD_ID": "build-1", "BUILD_NUMBER": "number-1"}, "build-1", id="test_build_id_precedence"),
        pytest.param(
            {"BUILD_NUMBER": "number-1", "GITHUB_RUN_ID": "run-1"}, "number-1", id="test_build_number_precedence"
        ),
        pytest.param({"GITHUB_RUN_ID": "run-1"}, "run-1", id="test_github_run_id_fallback"),
    ],
)
def test_run_id_uses_first_usable_ci_identifier(environment: dict[str, str], expected: str) -> None:
    """Given no explicit value, choose the first usable CI identifier in the reviewed order."""
    resolved = resolve_run_id(environment=environment)

    assert resolved.source == "ci"
    assert resolved.value == expected


def test_run_id_generates_positive_random_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given no explicit or CI values, render the generated positive integer as a safe marker."""
    monkeypatch.setattr("tests.observability.run_id.secrets.randbelow", lambda _limit: 41)

    resolved = resolve_run_id(environment={})

    assert resolved == type(resolved)(value="42", source="generated")


def test_normalize_run_id_replaces_unsafe_values_and_has_empty_fallback() -> None:
    """Given unsafe or empty input, produce a non-empty bounded marker-safe value."""
    assert normalize_run_id("  A/B??C  ") == "a-b-c"
    assert normalize_run_id("---") == "run"


def test_run_id_fixture_is_reused_within_one_pytest_session(
    observability_run_id: ResolvedRunID,
    request: pytest.FixtureRequest,
) -> None:
    """Given a session-scoped fixture, return the same resolved value for every consumer in the session."""
    assert request.getfixturevalue("observability_run_id") is observability_run_id
