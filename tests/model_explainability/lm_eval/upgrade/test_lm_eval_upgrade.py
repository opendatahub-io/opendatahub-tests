import pytest

from tests.model_explainability.lm_eval.utils import (
    get_lmeval_tasks,
    validate_lmeval_job_completed,
    validate_lmeval_job_started,
)

LMEVALJOB_COMPLETE_STATE: str = "Complete"

TIER1_LMEVAL_TASKS: list[str] = get_lmeval_tasks(min_downloads=10000)

TIER2_LMEVAL_TASKS: list[str] = list(
    set(get_lmeval_tasks(min_downloads=0.70, max_downloads=10000)) - set(TIER1_LMEVAL_TASKS)
)


@pytest.mark.parametrize(
    "model_namespace, lmevaljob_hf_upgrade",
    [
        pytest.param(
            {"name": "test-lmeval-lifecycle"},
            {"task_list": {"taskNames": ["arc_easy"]}},
        ),
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
class TestLMEvalJobLifecyclePreUpgrade:
    @pytest.mark.pre_upgrade
    def test_lmeval_job_pod_lifecycle(
        self,
        admin_client,
        model_namespace,
        lmevaljob_hf_pod_upgrade,
    ):
        """Verify LMEval job pod lifecycle before upgrade."""
        validate_lmeval_job_started(lmevaljob_pod=lmevaljob_hf_pod_upgrade)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-lmeval-lifecycle"},
        ),
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
class TestLMEvalJobLifecyclePostUpgrade:
    @pytest.mark.post_upgrade
    @pytest.mark.skip_on_disconnected
    def test_lmeval_job_pod_lifecycle(
        self,
        admin_client,
        model_namespace,
        lmevaljob_hf_pod_upgrade,
    ):
        validate_lmeval_job_completed(lmevaljob_pod=lmevaljob_hf_pod_upgrade)
