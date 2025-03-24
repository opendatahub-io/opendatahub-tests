import pytest

from tests.model_explainability.lm_eval.utils import verify_lmevaljob_running
from utilities.constants import Timeout


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-lmeval-huggingface"},
        )
    ],
    indirect=True,
)
@pytest.mark.smoke
def test_lmeval_huggingface_model(admin_client, model_namespace, lmevaljob_hf_pod):
    """Basic test that verifies that LMEval can run successfully pulling a model from HuggingFace."""
    lmevaljob_hf_pod.wait_for_status(status=lmevaljob_hf_pod.Status.SUCCEEDED, timeout=Timeout.TIMEOUT_10MIN)


@pytest.mark.parametrize(
    "model_namespace, lmeval_data_downloader_pod, lmevaljob_local_offline",
    [
        pytest.param(
            {"name": "test-lmeval-local-offline-builtin"},
            {
                "image": "quay.io/trustyai_testing/lmeval-assets-flan-arceasy"
                "@sha256:11cc9c2f38ac9cc26c4fab1a01a8c02db81c8f4801b5d2b2b90f90f91b97ac98"
            },
            {"task_list": {"taskNames": ["arc_easy"]}},
        )
    ],
    indirect=True,
)
def test_lmeval_local_offline_builtin_tasks_flan_arceasy(
    admin_client,
    model_namespace,
    lmeval_data_downloader_pod,
    lmevaljob_local_offline,
):
    """Test that verifies that LMEval can run successfully in local, offline mode using builtin tasks"""
    verify_lmevaljob_running(client=admin_client, lmevaljob=lmevaljob_local_offline)


@pytest.mark.parametrize(
    "model_namespace, lmeval_data_downloader_pod, lmevaljob_local_offline",
    [
        pytest.param(
            {"name": "test-lmeval-local-offline-unitxt"},
            {
                "image": "quay.io/trustyai_testing/lmeval-assets-flan-20newsgroups"
                "@sha256:3778c15079f11ef338a82ee35ae1aa43d6db52bac7bbfdeab343ccabe2608a0c"
            },
            {
                "task_list": {
                    "taskRecipes": [
                        {
                            "card": {"name": "cards.20_newsgroups_short"},
                            "template": {"name": "templates.classification.multi_class.title"},
                        }
                    ]
                }
            },
        )
    ],
    indirect=True,
)
def test_lmeval_local_offline_unitxt_tasks_flan_20newsgroups(
    admin_client,
    model_namespace,
    lmeval_data_downloader_pod,
    lmevaljob_local_offline,
):
    """Test that verifies that LMEval can run successfully in local, offline mode using unitxt"""
    verify_lmevaljob_running(client=admin_client, lmevaljob=lmevaljob_local_offline)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-lmeval-vllm"},
        )
    ],
    indirect=True,
)
def test_lmeval_vllm_emulator(admin_client, model_namespace, lmevaljob_vllm_emulator_pod):
    """Basic test that verifies LMEval works with vLLM using a vLLM emulator for more efficient evaluation"""
    lmevaljob_vllm_emulator_pod.wait_for_status(
        status=lmevaljob_vllm_emulator_pod.Status.SUCCEEDED, timeout=Timeout.TIMEOUT_10MIN
    )
