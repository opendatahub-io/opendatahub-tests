import pytest
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.namespace import Namespace


@pytest.fixture(scope="function")
def lm_eval_job_hf(model_namespace: Namespace):
    with LMEvalJob(
        name="test-job",
        namespace=model_namespace.name,
        model="hf",
        model_args=[{"name": "pretrained", "value": "google/flan-t5-base"}],
        task_list={
            "taskRecipes": [
                {"card": {"name": "cards.wnli"}, "template": "templates.classification.multi_class.relation.default"}
            ]
        },
        log_samples=True,
        allow_online=True,
        allow_code_execution=True,
    ) as job:
        yield job
