import pytest
from typing import List

from tests.model_explainability.lm_eval.constants import LLMAAJ_TASK_DATA, CUSTOM_UNITXT_TASK_DATA
from tests.model_explainability.utils import validate_tai_component_images

from tests.model_explainability.lm_eval.utils import get_lmeval_tasks, validate_lmeval_job_pod_and_logs

LMEVALJOB_COMPLETE_STATE: str = "Complete"

TIER1_LMEVAL_TASKS: List[str] = get_lmeval_tasks(min_downloads=10000)

TIER2_LMEVAL_TASKS: List[str] = list(
    set(get_lmeval_tasks(min_downloads=0.70, max_downloads=10000)) - set(TIER1_LMEVAL_TASKS)
)


@pytest.mark.skip_on_disconnected
@pytest.mark.parametrize(
    "model_namespace, lmevaljob_hf",
    [
        pytest.param(
            {"name": "test-lmeval-hf-tier1"},
            {"task_list": {"taskNames": TIER1_LMEVAL_TASKS}},
        ),
        pytest.param(
            {"name": "test-lmeval-hf-tier2"},
            {"task_list": {"taskNames": TIER2_LMEVAL_TASKS}},
        ),
        pytest.param(
            {"name": "test-lmeval-hf-custom-task"},
            CUSTOM_UNITXT_TASK_DATA,
            id="custom_task",
        ),
        pytest.param(
            {"name": "test-lmeval-hf-llmaaj"},
            LLMAAJ_TASK_DATA,
            id="llmaaj_task",
        ),
    ],
    indirect=True,
)
def test_lmeval_huggingface_model(admin_client, model_namespace, lmevaljob_hf_pod):
    """Tests that verify running common evaluations (and a custom one) on a model pulled directly from HuggingFace.
    On each test we run a different evaluation task, limiting it to 0.5% of the questions on each eval."""
    validate_lmeval_job_pod_and_logs(lmevaljob_pod=lmevaljob_hf_pod)


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
    lmevaljob_local_offline_pod,
):
    """Test that verifies that LMEval can run successfully in local, offline mode using builtin tasks"""
    validate_lmeval_job_pod_and_logs(lmevaljob_pod=lmevaljob_local_offline_pod)


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
    lmevaljob_local_offline_pod,
):
    """Test that verifies that LMEval can run successfully in local, offline mode using unitxt"""
    validate_lmeval_job_pod_and_logs(lmevaljob_pod=lmevaljob_local_offline_pod)


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
    validate_lmeval_job_pod_and_logs(lmevaljob_pod=lmevaljob_vllm_emulator_pod)


@pytest.mark.parametrize(
    "model_namespace, minio_data_connection",
    [
        pytest.param(
            {"name": "test-s3-lmeval"},
            {"bucket": "models"},
        )
    ],
    indirect=True,
)
def test_lmeval_s3_storage(
    admin_client,
    model_namespace,
    lmevaljob_s3_offline_pod,
):
    """Test to verify that LMEval works with a model stored in a S3 bucket"""
    validate_lmeval_job_pod_and_logs(lmevaljob_pod=lmevaljob_s3_offline_pod)


@pytest.mark.parametrize(
    "model_namespace, minio_data_connection",
    [
        pytest.param(
            {"name": "test-lmeval-images"},
            {"bucket": "models"},
        )
    ],
    indirect=True,
)
@pytest.mark.smoke
def test_verify_lmeval_pod_images(lmevaljob_s3_offline_pod, trustyai_operator_configmap) -> None:
    """Test to verify LMEval pod images.
    Checks if the image tag from the ConfigMap is used within the Pod and if it's pinned using a sha256 digest.

    Verifies:
        - lmeval driver image
        - lmeval job runner image
    """
    validate_tai_component_images(
        pod=lmevaljob_s3_offline_pod, tai_operator_configmap=trustyai_operator_configmap, include_init_containers=True
    )
