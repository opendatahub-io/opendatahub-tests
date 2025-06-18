import pytest

from tests.model_explainability.utils import validate_tai_component_images
from utilities.constants import Timeout

LMEVALJOB_COMPLETE_STATE: str = "Complete"


@pytest.mark.parametrize(
    "model_namespace, lmevaljob_hf",
    [
        pytest.param(
            {"name": "test-lmeval-hf-arc"}, {"task_list": {"taskNames": ["arc_challenge"]}}, id="arc_challenge"
        ),
        pytest.param(
            {"name": "test-lmeval-hf-mmlu"},
            {"task_list": {"taskNames": ["mmlu_astronomy_generative"]}},
            id="mmlu_astronomy_generative",
        ),
        pytest.param({"name": "test-lmeval-hf-hellaswag"}, {"task_list": {"taskNames": ["hellaswag"]}}, id="hellaswag"),
        pytest.param(
            {"name": "test-lmeval-hf-truthfulqa"}, {"task_list": {"taskNames": ["truthfulqa_gen"]}}, id="truthfulqa_gen"
        ),
        pytest.param(
            {"name": "test-lmeval-hf-winogrande"}, {"task_list": {"taskNames": ["winogrande"]}}, id="winogrande"
        ),
        pytest.param(
            {"name": "test-lmeval-hf-custom-task"},
            {
                "task_list": {
                    "custom": {
                        "systemPrompts": [
                            {"name": "sp_0", "value": "Be concise. At every point give the shortest acceptable answer."}
                        ],
                        "templates": [
                            {
                                "name": "tp_0",
                                "value": '{ "__type__": "input_output_template", '
                                '"input_format": "{text_a_type}: {text_a}\\n'
                                '{text_b_type}: {text_b}", '
                                '"output_format": "{label}", '
                                '"target_prefix": '
                                '"The {type_of_relation} class is ", '
                                '"instruction": "Given a {text_a_type} and {text_b_type} '
                                'classify the {type_of_relation} of the {text_b_type} to one of {classes}.",'
                                ' "postprocessors": [ "processors.take_first_non_empty_line",'
                                ' "processors.lower_case_till_punc" ] }',
                            }
                        ],
                    },
                    "taskRecipes": [
                        {"card": {"name": "cards.wnli"}, "systemPrompt": {"ref": "sp_0"}, "template": {"ref": "tp_0"}}
                    ],
                }
            },
            id="custom_task",
        ),
    ],
    indirect=True,
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
