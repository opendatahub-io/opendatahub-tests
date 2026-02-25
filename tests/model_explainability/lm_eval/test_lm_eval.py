import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

from tests.model_explainability.lm_eval.constants import (
    ARC_EASY_DATASET_IMAGE,
    CUSTOM_UNITXT_TASK_DATA,
    LLMAAJ_TASK_DATA,
    LMEVAL_OCI_REPO,
    LMEVAL_OCI_TAG,
)
from tests.model_explainability.lm_eval.utils import get_lmeval_tasks, validate_lmeval_job_pod_and_logs
from tests.model_explainability.utils import validate_tai_component_images
from utilities.constants import OCIRegistry
from utilities.registry_utils import pull_manifest_from_oci_registry

LMEVALJOB_COMPLETE_STATE: str = "Complete"

TIER1_LMEVAL_TASKS: list[str] = get_lmeval_tasks(min_downloads=10000)

TIER2_LMEVAL_TASKS: list[str] = list(
    set(get_lmeval_tasks(min_downloads=0.70, max_downloads=10000)) - set(TIER1_LMEVAL_TASKS)
)

LOGGER = get_logger(name=__name__)


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
            {"dataset_image": ARC_EASY_DATASET_IMAGE},
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
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
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


@pytest.mark.parametrize(
    "model_namespace, oci_registry_pod_with_minio, lmeval_data_downloader_pod, lmevaljob_local_offline_oci",
    [
        pytest.param(
            {"name": "test-lmeval-local-offline-unitxt"},
            OCIRegistry.PodConfig.REGISTRY_BASE_CONFIG,
            {
                "dataset_image": "quay.io/trustyai_testing/lmeval-assets-20newsgroups"
                "@sha256:106023a7ee0c93afad5d27ae50130809ccc232298b903c8b12ea452e9faafce2"
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
def test_lmeval_local_offline_unitxt_tasks_flan_20newsgroups_oci_artifacts(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lmeval_data_downloader_pod: Pod,
    lmevaljob_local_offline_pod_oci: Pod,
    oci_registry_host: str,
):
    """Test that verifies LMEval can run successfully in local, offline mode using unitxt tasks with OCI artifacts."""
    validate_lmeval_job_pod_and_logs(lmevaljob_pod=lmevaljob_local_offline_pod_oci)
    LOGGER.info("Verifying OCI registry upload")
    registry_url = f"http://{oci_registry_host}"
    LOGGER.info(f"Verifying artifact in OCI registry: {registry_url}/v2/{LMEVAL_OCI_REPO}/manifests/{LMEVAL_OCI_TAG}")
    pull_manifest_from_oci_registry(registry_url=registry_url, repo=LMEVAL_OCI_REPO, tag=LMEVAL_OCI_TAG)
    LOGGER.info("Manifest found in OCI registry")
