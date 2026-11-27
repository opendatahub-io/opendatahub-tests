import pytest

from tests.observability.rollout import (
    PipelineStage,
    RolloutState,
    StageOutcome,
    StageResult,
    classify_first_failure,
)

pytestmark = pytest.mark.tier1


def test_classifier_stops_at_first_failed_stage() -> None:
    """Given emission and ingestion success followed by a query failure, classify the query stage."""
    classification = classify_first_failure(
        results=[
            StageResult(
                stage=PipelineStage.EMISSION,
                outcome=StageOutcome.PASS,
                rollout_state=RolloutState.PASSED,
            ),
            StageResult(
                stage=PipelineStage.INGESTION,
                outcome=StageOutcome.PASS,
                rollout_state=RolloutState.PASSED,
            ),
            StageResult(
                stage=PipelineStage.STORAGE,
                outcome=StageOutcome.PASS,
                rollout_state=RolloutState.PASSED,
            ),
            StageResult(
                stage=PipelineStage.QUERY,
                outcome=StageOutcome.FAIL,
                rollout_state=RolloutState.FAILED,
                details={"http_status": 500, "result_count": 0},
            ),
        ]
    )

    assert classification.first_failure_stage is PipelineStage.QUERY
    assert classification.failure_category == "query-contract"
    assert classification.rollout_state is RolloutState.FAILED


def test_classifier_keeps_unactivated_prerequisite_skipped() -> None:
    """Given an unactivated signal, report not-evaluated/skipped rather than an environment failure."""
    classification = classify_first_failure(
        results=[
            StageResult(
                stage=PipelineStage.EMISSION,
                outcome=StageOutcome.PASS,
                rollout_state=RolloutState.PASSED,
            ),
            StageResult(
                stage=PipelineStage.INGESTION,
                outcome=StageOutcome.NOT_EVALUATED,
                rollout_state=RolloutState.SKIPPED,
                details={"reason": "trace fixture is not established"},
            ),
        ]
    )

    assert classification.outcome is StageOutcome.NOT_EVALUATED
    assert classification.not_evaluated_stage is PipelineStage.INGESTION
    assert classification.first_failure_stage is None
    assert classification.failure_category is None
    assert classification.rollout_state is RolloutState.SKIPPED


def test_classifier_requires_independent_store_read_before_storage_failure() -> None:
    """Given no direct store read, reject a storage failure instead of misclassifying the first broken stage."""
    with pytest.raises(ValueError, match="independent store read"):
        StageResult(
            stage=PipelineStage.STORAGE,
            outcome=StageOutcome.FAIL,
            rollout_state=RolloutState.FAILED,
        )


def test_classifier_passes_when_all_evaluated_stages_pass() -> None:
    """Given successful results for every pipeline stage, report an activated pass."""
    classification = classify_first_failure(
        results=(
            StageResult(stage=stage, outcome=StageOutcome.PASS, rollout_state=RolloutState.PASSED)
            for stage in PipelineStage
        )
    )

    assert classification.outcome is StageOutcome.PASS
    assert classification.rollout_state is RolloutState.PASSED


def test_classifier_rejects_duplicate_stage_results() -> None:
    """Given two results for one stage, reject ambiguous first-failure evidence."""
    result = StageResult(
        stage=PipelineStage.EMISSION,
        outcome=StageOutcome.PASS,
        rollout_state=RolloutState.PASSED,
    )

    with pytest.raises(ValueError, match="duplicate pipeline stage"):
        classify_first_failure(results=(result, result))


def test_classifier_marks_missing_stage_as_not_evaluated() -> None:
    """Given an incomplete stage sequence, report the first missing stage instead of passing it."""
    classification = classify_first_failure(
        results=(
            StageResult(
                stage=PipelineStage.EMISSION,
                outcome=StageOutcome.PASS,
                rollout_state=RolloutState.PASSED,
            ),
        )
    )

    assert classification.outcome is StageOutcome.NOT_EVALUATED
    assert classification.rollout_state is RolloutState.BLOCKED
    assert classification.not_evaluated_stage is PipelineStage.INGESTION


def test_classifier_marks_empty_stage_sequence_as_not_evaluated() -> None:
    """Given no stage results, report emission as not evaluated instead of an activated pass."""
    classification = classify_first_failure(results=())

    assert classification.outcome is StageOutcome.NOT_EVALUATED
    assert classification.not_evaluated_stage is PipelineStage.EMISSION
