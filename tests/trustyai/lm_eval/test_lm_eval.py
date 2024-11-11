import pytest
from ocp_resources.pod import Pod


TIMEOUT_10MIN = 60 * 10


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "lm-eval-job-hf"},
        )
    ],
    indirect=True,
)
def test_lm_eval_huggingface_model(model_namespace, lm_eval_job_hf):
    """Basic test that verifies that lm-eval can run successfully pulling a model from HuggingFace."""
    lm_eval_job_pod = Pod(name=lm_eval_job_hf.name, namespace=lm_eval_job_hf.namespace, wait_for_resource=True)
    lm_eval_job_pod.wait_for_status(lm_eval_job_pod.Status.SUCCEEDED, timeout=TIMEOUT_10MIN)
