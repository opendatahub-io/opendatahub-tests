import pytest

from tests.model_explainability.utils import validate_tai_component_images
from utilities.constants import Timeout

from tests.model_explainability.lm_eval.utils import get_lmeval_tasks

LMEVALJOB_COMPLETE_STATE: str = "Complete"

LMEVAL_TASKS_HEAD5 = get_lmeval_tasks()[:5]

params = [
    pytest.param(
        {"name": f"test-lmeval-hf-{x}"},
        {"task_list": {"taskNames": [LMEVAL_TASKS_HEAD5[x]]}},
        id=LMEVAL_TASKS_HEAD5[x]
    )
    for x in range(len(LMEVAL_TASKS_HEAD5))
]

@pytest.mark.parametrize(
    "model_namespace, lmevaljob_hf", params, indirect=True,
)

def test_lmeval_huggingface_model(admin_client, model_namespace, lmevaljob_hf_pod):
    """Tests that verify running common evaluations (and a custom one) on a model pulled directly from HuggingFace.
    On each test we run a different evaluation task, limiting it to 1% of the questions on each eval."""
    lmevaljob_hf_pod.wait_for_status(status=lmevaljob_hf_pod.Status.SUCCEEDED, timeout=Timeout.TIMEOUT_20MIN)


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
@pytest.mark.smoke
def test_lmeval_local_offline_builtin_tasks_flan_arceasy(
    admin_client,
    model_namespace,
    lmeval_data_downloader_pod,
    lmevaljob_local_offline_pod,
):
    """Test that verifies that LMEval can run successfully in local, offline mode using builtin tasks"""
    lmevaljob_local_offline_pod.wait_for_status(
        status=lmevaljob_local_offline_pod.Status.SUCCEEDED, timeout=Timeout.TIMEOUT_20MIN
    )


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
    lmevaljob_local_offline_pod.wait_for_status(
        status=lmevaljob_local_offline_pod.Status.SUCCEEDED, timeout=Timeout.TIMEOUT_20MIN
    )


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
        status=lmevaljob_vllm_emulator_pod.Status.SUCCEEDED, timeout=Timeout.TIMEOUT_20MIN
    )


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
    lmevaljob_s3_offline_pod.wait_for_status(
        status=lmevaljob_s3_offline_pod.Status.SUCCEEDED, timeout=Timeout.TIMEOUT_20MIN
    )


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
