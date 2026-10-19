"""Rollout states and first-broken-stage classification for telemetry checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class RolloutState(StrEnum):
    """Shared outcome vocabulary for activated and unactivated checks."""

    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"
    PASSED = "passed"


class StageOutcome(StrEnum):
    """Outcome of evaluating one pipeline stage."""

    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUATED = "not-evaluated"


class PipelineStage(StrEnum):
    """Ordered pipeline stages used for first-failure classification."""

    EMISSION = "emission"
    INGESTION = "ingestion/scrape"
    STORAGE = "storage"
    QUERY = "query"
    AUTHORIZATION = "authorization"
    USER_RETRIEVAL = "user-retrieval"


FAILURE_CATEGORIES = {
    PipelineStage.EMISSION: "marker-emission",
    PipelineStage.INGESTION: "ingestion-scrape",
    PipelineStage.STORAGE: "storage",
    PipelineStage.QUERY: "query-contract",
    PipelineStage.AUTHORIZATION: "authorization-contract",
    PipelineStage.USER_RETRIEVAL: "user-retrieval",
}


@dataclass(frozen=True)
class StageResult:
    """Sanitized result for one pipeline stage."""

    stage: PipelineStage
    outcome: StageOutcome
    rollout_state: RolloutState
    details: Mapping[str, object] = field(default_factory=dict)
    independent_store_read: bool = False

    def __post_init__(self) -> None:
        if (
            self.stage is PipelineStage.STORAGE
            and self.outcome is StageOutcome.FAIL
            and not self.independent_store_read
        ):
            raise ValueError("storage failure requires an independent store read")

    def to_dict(self) -> dict[str, object]:
        """Return stage evidence without unbounded backend payloads."""
        return {
            "stage": self.stage.value,
            "outcome": self.outcome.value,
            "rollout_state": self.rollout_state.value,
            "details": dict(self.details),
            "independent_store_read": self.independent_store_read,
        }


@dataclass(frozen=True)
class FailureClassification:
    """The first failure, or the first stage that could not be evaluated."""

    outcome: StageOutcome
    rollout_state: RolloutState
    first_failure_stage: PipelineStage | None
    not_evaluated_stage: PipelineStage | None
    failure_category: str | None
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return sanitized machine-readable classification evidence."""
        return {
            "outcome": self.outcome.value,
            "rollout_state": self.rollout_state.value,
            "first_failure_stage": self.first_failure_stage.value if self.first_failure_stage else None,
            "not_evaluated_stage": self.not_evaluated_stage.value if self.not_evaluated_stage else None,
            "failure_category": self.failure_category,
            "details": dict(self.details),
        }


def classify_first_failure(results: Iterable[StageResult]) -> FailureClassification:
    """Classify stages in emission-to-retrieval order and stop at the first problem.

    A missing prerequisite is deliberately returned as ``not-evaluated``. Its rollout
    state determines whether the visible result is ``skipped`` or ``blocked``. Storage
    is never reported as a failure unless a separate store read was attempted.
    """
    by_stage: dict[PipelineStage, StageResult] = {}
    for result in results:
        if result.stage in by_stage:
            raise ValueError(f"duplicate pipeline stage result: {result.stage.value}")
        by_stage[result.stage] = result

    missing_rollout_state = (
        RolloutState.SKIPPED
        if any(result.rollout_state is RolloutState.SKIPPED for result in by_stage.values())
        else RolloutState.BLOCKED
    )
    for stage in PipelineStage:
        result = by_stage.get(stage)
        if result is None:
            return FailureClassification(
                outcome=StageOutcome.NOT_EVALUATED,
                rollout_state=missing_rollout_state,
                first_failure_stage=None,
                not_evaluated_stage=stage,
                failure_category=None,
                details={"reason": "stage result is missing"},
            )
        if result.outcome is StageOutcome.FAIL:
            return FailureClassification(
                outcome=StageOutcome.FAIL,
                rollout_state=result.rollout_state,
                first_failure_stage=stage,
                not_evaluated_stage=None,
                failure_category=FAILURE_CATEGORIES[stage],
                details=result.details,
            )
        if result.outcome is StageOutcome.NOT_EVALUATED:
            return FailureClassification(
                outcome=StageOutcome.NOT_EVALUATED,
                rollout_state=result.rollout_state,
                first_failure_stage=None,
                not_evaluated_stage=stage,
                failure_category=None,
                details=result.details,
            )

    return FailureClassification(
        outcome=StageOutcome.PASS,
        rollout_state=RolloutState.PASSED,
        first_failure_stage=None,
        not_evaluated_stage=None,
        failure_category=None,
    )
