import pytest
from ocp_resources.pod import Pod

from tests.trustyai.constants import TIMEOUT_10MIN
from tests.trustyai.lm_eval.utils import verify_lmevaljob_running


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "lmevaljob-hf"},
        )
    ],
    indirect=True,
)
# TODO: replace with pytest-jira marker
@pytest.mark.skip(reason="Feature not implemented yet")
def test_lmeval_huggingface_model(admin_client, model_namespace, lm_eval_job_hf):
    """Basic test that verifies that LMEval can run successfully pulling a model from HuggingFace."""
    lmevaljob_pod = Pod(
        client=admin_client, name=lm_eval_job_hf.name, namespace=lm_eval_job_hf.namespace, wait_for_resource=True
    )
    lmevaljob_pod.wait_for_status(status=lmevaljob_pod.Status.SUCCEEDED, timeout=TIMEOUT_10MIN)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "lmevaljob-local-offline-builtin"},
        )
    ],
    indirect=True,
)
def test_lmeval_local_offline_builtin_tasks(
    admin_client,
    model_namespace,
    patched_trustyai_operator_configmap_allow_online,
    lmeval_data_pvc,
    lmeval_data_downloader_pod_flan_arceasy,
    lmevaljob_local_offline_builtin_tasks,
):
    """Test that verifies that LMEval can run successfully in local, offline mode using builtin tasks"""
    verify_lmevaljob_running(client=admin_client, lmevaljob=lmevaljob_local_offline_builtin_tasks)
