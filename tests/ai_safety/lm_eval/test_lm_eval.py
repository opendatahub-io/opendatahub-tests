import itertools

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.image_constants import AiSafetyImages
from tests.ai_safety.lm_eval.constants import (
    ARC_EASY_DATASET_IMAGE,
    CUSTOM_UNITXT_TASK_DATA,
    LAST_SCHEDULED_GENERATION_ANNOTATION,
    LLMAAJ_TASK_DATA,
    LMEVAL_OCI_REPO,
    LMEVAL_OCI_TAG,
    LMEVAL_SHORT_JOB_TIMEOUT,
    LMEVALJOB_COMPLETE_STATE,
    ODH_TRUSTED_CA_BUNDLE_CONFIGMAP,
)
from tests.ai_safety.lm_eval.utils import (
    get_lmeval_tasks,
    validate_ca_bundle_injected,
    validate_ca_bundle_not_injected,
    validate_lmeval_job_pod_and_logs,
    wait_for_lmevaljob_state,
    wait_for_vllm_model_ready,
)
from tests.ai_safety.utils import validate_tai_component_images
from utilities.constants import OCIRegistry, SeaweedFs
from utilities.registry_utils import pull_manifest_from_oci_registry

# Tasks that cannot run unattended in CI, mapped to why.
LMEVAL_TASKS_EXCLUDED_IN_CI: dict[str, str] = {
    "realtoxicityprompts": "metric raises KeyError without PERSPECTIVE_API_KEY",
    "storycloze_2018": "datasets.exceptions.ManualDownloadError: requires manually downloaded data",
}

TIER1_LMEVAL_TASKS: list[str] = sorted(set(get_lmeval_tasks(min_downloads=10000)) - set(LMEVAL_TASKS_EXCLUDED_IN_CI))

TIER2_LMEVAL_TASKS: list[str] = sorted(
    set(get_lmeval_tasks(min_downloads=0.70, max_downloads=10000))
    - set(TIER1_LMEVAL_TASKS)
    - set(LMEVAL_TASKS_EXCLUDED_IN_CI)
)

TIER2_CHUNK_SIZE: int = 10
TIER2_LMEVAL_TASK_CHUNKS: list[list[str]] = [
    list(batch) for batch in itertools.batched(TIER2_LMEVAL_TASKS, TIER2_CHUNK_SIZE)
]

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.smoke
@pytest.mark.ai_safety
def test_lmevaljob_crd_exists(
    admin_client: DynamicClient,
) -> None:
    """Verify LMEvalJob CRD exists on the cluster."""
    crd_name = "lmevaljobs.trustyai.opendatahub.io"

    crd_resource = CustomResourceDefinition(
        client=admin_client,
        name=crd_name,
        ensure_exists=True,
    )

    assert crd_resource.exists, f"CRD {crd_name} does not exist on the cluster"


@pytest.mark.xfail(
    reason="RHOAIENG-94190: LMEvalJob scheduling can race the operator's reconciler "
    "(handleNewCR is not idempotent against its own last-scheduled-generation annotation write), "
    "leaving status.state stuck at 'Scheduled' forever even though the pod succeeds. Intermittent, "
    "so a passing run does not mean the bug is fixed. Fix in review: trustyai-service-operator#1073.",
    strict=False,
)
@pytest.mark.skip_on_disconnected
@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace, lmevaljob_hf",
    [
        pytest.param(
            {"name": "test-lmeval-hf-tier1"},
            {"task_list": {"taskNames": TIER1_LMEVAL_TASKS}},
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


@pytest.mark.xfail(
    reason="RHOAIENG-94190: LMEvalJob scheduling can race the operator's reconciler "
    "(handleNewCR is not idempotent against its own last-scheduled-generation annotation write), "
    "leaving status.state stuck at 'Scheduled' forever even though the pod succeeds. Intermittent, "
    "so a passing run does not mean the bug is fixed. Fix in review: trustyai-service-operator#1073.",
    strict=False,
)
@pytest.mark.skip_on_disconnected
@pytest.mark.tier2
@pytest.mark.parametrize(
    "model_namespace, lmevaljob_hf",
    [
        pytest.param(
            {"name": f"test-lmeval-hf-tier2-{idx}"},
            {"task_list": {"taskNames": chunk}},
            id=f"tier2_chunk{idx}",
        )
        for idx, chunk in enumerate(TIER2_LMEVAL_TASK_CHUNKS)
    ],
    indirect=True,
)
def test_lmeval_huggingface_model_tier2(admin_client, model_namespace, lmevaljob_hf_pod):
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
@pytest.mark.tier1
def test_lmeval_local_offline_builtin_tasks_flan_arceasy(
    admin_client,
    model_namespace,
    lmeval_data_downloader_pod,
    lmevaljob_local_offline_pod,
):
    """Test that verifies that LMEval can run successfully in local, offline mode using builtin tasks"""
    validate_lmeval_job_pod_and_logs(lmevaljob_pod=lmevaljob_local_offline_pod)


@pytest.mark.tier1
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


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace, lmeval_seaweedfs_data_connection",
    [
        pytest.param(
            {"name": "test-s3-lmeval"},
            {"bucket": SeaweedFs.Buckets.MODELMESH_EXAMPLE_MODELS},
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
    "model_namespace, lmeval_seaweedfs_data_connection",
    [
        pytest.param(
            {"name": "test-lmeval-images"},
            {"bucket": SeaweedFs.Buckets.MODELMESH_EXAMPLE_MODELS},
        )
    ],
    indirect=True,
)
@pytest.mark.tier1
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


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace, oci_registry_pod_with_seaweedfs, lmeval_data_downloader_pod, lmevaljob_local_offline_oci",
    [
        pytest.param(
            {"name": "test-lmeval-local-offline-unitxt"},
            OCIRegistry.PodConfig.REGISTRY_BASE_CONFIG,
            {"dataset_image": AiSafetyImages.NEWSGROUPS_DATASET},
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


@pytest.mark.vllm_nvidia_single_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.tier2
@pytest.mark.skip_on_disconnected
@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-lmeval-gpu"},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed", "skip_if_no_supported_accelerator_type")
def test_lmeval_gpu(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    patched_dsc_lmeval_allow_all,
    lmeval_vllm_inference_service,
    lmevaljob_gpu_pod,
):
    """Test LMEval with GPU-backed model deployment via vLLM.

    Verifies that LMEval can successfully evaluate a model deployed on GPU using vLLM runtime.
    The model is downloaded directly from HuggingFace Hub and evaluated using the arc_easy task.
    """
    wait_for_vllm_model_ready(
        client=admin_client,
        namespace=model_namespace.name,
        inference_service_name=lmeval_vllm_inference_service.name,
    )

    validate_lmeval_job_pod_and_logs(lmevaljob_pod=lmevaljob_gpu_pod, timeout=LMEVAL_SHORT_JOB_TIMEOUT)


@pytest.mark.tier2
@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-lmeval-https"},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
def test_lmeval_vllm_emulator_https_ca_bundle(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lmevaljob_vllm_emulator_https: LMEvalJob,
    lmevaljob_vllm_emulator_https_pod: Pod,
):
    """Test CA bundle injection for HTTPS LMEvalJob.

    Given: A vLLM emulator exposed via TLS edge-terminated Route
    When: LMEvalJob is created with HTTPS base_url
    Then: Operator injects CA bundle (ConfigMap, volume, mount, REQUESTS_CA_BUNDLE env var)
          and evaluation completes without SSL errors

    Validates RHOAIENG-60487.
    """
    validate_ca_bundle_injected(pod=lmevaljob_vllm_emulator_https_pod, job_name=lmevaljob_vllm_emulator_https.name)
    validate_lmeval_job_pod_and_logs(lmevaljob_pod=lmevaljob_vllm_emulator_https_pod, timeout=LMEVAL_SHORT_JOB_TIMEOUT)


@pytest.mark.tier2
@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-lmeval-http-no-ca"},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
def test_lmeval_vllm_emulator_http_no_ca_bundle(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lmevaljob_vllm_emulator: LMEvalJob,
    lmevaljob_vllm_emulator_pod: Pod,
):
    """Test no CA bundle injection for HTTP LMEvalJob.

    Given: A vLLM emulator exposed via plain HTTP service
    When: LMEvalJob is created with HTTP base_url
    Then: Operator does not inject CA bundle and evaluation completes successfully
    """
    validate_ca_bundle_not_injected(pod=lmevaljob_vllm_emulator_pod, job_name=lmevaljob_vllm_emulator.name)
    validate_lmeval_job_pod_and_logs(lmevaljob_pod=lmevaljob_vllm_emulator_pod, timeout=LMEVAL_SHORT_JOB_TIMEOUT)


@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-lmeval-https-verify-cert"},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
def test_lmeval_https_verify_certificate_no_ca_bundle(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lmevaljob_vllm_emulator_https_verify_cert: LMEvalJob,
    lmevaljob_vllm_emulator_https_verify_cert_pod: Pod,
):
    """Test no CA bundle injection when verify_certificate is set.

    Given: A vLLM emulator exposed via HTTPS TLS-terminated Route
    When: LMEvalJob is created with HTTPS base_url and verify_certificate explicitly set
    Then: Operator does not inject CA bundle despite HTTPS scheme
    """
    validate_ca_bundle_not_injected(
        pod=lmevaljob_vllm_emulator_https_verify_cert_pod,
        job_name=lmevaljob_vllm_emulator_https_verify_cert.name,
    )


@pytest.mark.xfail(
    reason="RHOAIENG-94190: LMEvalJob scheduling can race the operator's reconciler "
    "(handleNewCR is not idempotent against its own last-scheduled-generation annotation write), "
    "leaving status.state stuck at 'Scheduled' forever even though the pod succeeds. Intermittent, "
    "so a passing run does not mean the bug is fixed. Fix in review: trustyai-service-operator#1073.",
    strict=False,
)
@pytest.mark.tier1
@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-lmeval-rerun"},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
def test_lmeval_rerun_after_spec_change(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lmevaljob_vllm_emulator: LMEvalJob,
    lmevaljob_vllm_emulator_pod: Pod,
):
    """Test LMEvalJob re-runs after spec change.

    Given: A completed LMEvalJob with last-scheduled-generation annotation set
    When: Job spec is edited (limit changed, bumping metadata.Generation)
    Then: Job re-runs for the new generation, the generation annotation updates, and the job completes successfully

    Validates trustyai-service-operator#729 re-run behavior.
    """
    validate_lmeval_job_pod_and_logs(
        lmevaljob_pod=lmevaljob_vllm_emulator_pod,
        timeout=LMEVAL_SHORT_JOB_TIMEOUT,
    )

    wait_for_lmevaljob_state(
        lmevaljob=lmevaljob_vllm_emulator,
        state=LMEVALJOB_COMPLETE_STATE,
    )

    annotations = dict(lmevaljob_vllm_emulator.instance.metadata.annotations or {})
    assert LAST_SCHEDULED_GENERATION_ANNOTATION in annotations, (
        f"Expected annotation '{LAST_SCHEDULED_GENERATION_ANNOTATION}' not found on completed job"
    )
    initial_generation = annotations[LAST_SCHEDULED_GENERATION_ANNOTATION]

    LOGGER.info("Job completed, editing spec to trigger re-run")
    ResourceEditor(
        patches={
            lmevaljob_vllm_emulator: {
                "spec": {
                    "limit": "0.02",
                }
            }
        }
    ).update()

    new_generation = str(lmevaljob_vllm_emulator.instance.metadata.generation)
    assert new_generation != initial_generation, f"Spec patch did not bump metadata.generation (still {new_generation})"

    LOGGER.info(f"Waiting for re-run of generation {new_generation} to complete")
    last_annotation: str | None = None
    last_state: str | None = None
    last_reason: str | None = None
    try:
        for job_instance in TimeoutSampler(
            wait_timeout=LMEVAL_SHORT_JOB_TIMEOUT,
            sleep=5,
            func=lambda: lmevaljob_vllm_emulator.instance,
        ):
            annotations = dict(job_instance.metadata.annotations or {})
            last_annotation = annotations.get(LAST_SCHEDULED_GENERATION_ANNOTATION)
            status = job_instance.status or {}
            last_state = status.get("state")
            last_reason = status.get("reason")
            if last_annotation == new_generation and last_state == LMEVALJOB_COMPLETE_STATE:
                break
    except TimeoutExpiredError:
        pytest.fail(
            f"Re-run did not complete for generation {new_generation}. "
            f"Last annotation: {last_annotation}, state: {last_state}, reason: {last_reason}"
        )

    assert last_reason == "Succeeded", f"Re-run completed with reason '{last_reason}', expected 'Succeeded'"


# ── Tests for generalized CA bundle injection (RHOAIENG-60453) ──
# These tests define the expected behavior of the unconditional CA mount approach.
# They xfail against PR #729's conditional injection and will pass once the
# general fix is implemented in the operator.


@pytest.mark.xfail(
    reason="Requires generalized CA bundle injection (RHOAIENG-60453): "
    "unconditional mount of odh-trusted-ca-bundle with SSL_CERT_FILE",
    strict=True,
)
@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-lmeval-https-ssl-cert-file"},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
def test_lmeval_https_sets_ssl_cert_file(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lmevaljob_vllm_emulator_https_pod: Pod,
):
    """Test SSL_CERT_FILE is set alongside REQUESTS_CA_BUNDLE for HTTPS jobs.

    Given: A vLLM emulator exposed via HTTPS TLS-terminated Route
    When: LMEvalJob is created with HTTPS base_url
    Then: Both SSL_CERT_FILE and REQUESTS_CA_BUNDLE are set and point to the same CA bundle

    Note: PR #729 only sets REQUESTS_CA_BUNDLE. General fix (RHOAIENG-60453) should set both.
    """
    main_container = lmevaljob_vllm_emulator_https_pod.instance.spec.containers[0]
    env_map = {env_var.name: env_var.value for env_var in (main_container.env or [])}
    assert "SSL_CERT_FILE" in env_map, "SSL_CERT_FILE env var not found on HTTPS job pod"
    assert "REQUESTS_CA_BUNDLE" in env_map, "REQUESTS_CA_BUNDLE env var not found on HTTPS job pod"
    assert env_map["SSL_CERT_FILE"] == env_map["REQUESTS_CA_BUNDLE"], (
        "SSL_CERT_FILE and REQUESTS_CA_BUNDLE should point to the same CA bundle path"
    )


@pytest.mark.xfail(
    reason="Requires generalized CA bundle injection (RHOAIENG-60453): "
    "unconditional mount of odh-trusted-ca-bundle with SSL_CERT_FILE",
    strict=True,
)
@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-lmeval-http-has-ca"},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
def test_lmeval_http_has_ca_bundle(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lmevaljob_vllm_emulator_pod: Pod,
):
    """Test CA trust store is mounted unconditionally for HTTP jobs.

    Given: A vLLM emulator exposed via plain HTTP service
    When: LMEvalJob is created with HTTP base_url
    Then: CA trust store (odh-trusted-ca-bundle) is mounted and env vars set for outbound HTTPS

    Note: General fix (RHOAIENG-60453) mounts CA unconditionally for all outbound HTTPS traffic.
    """
    main_container = lmevaljob_vllm_emulator_pod.instance.spec.containers[0]
    env_map = {env_var.name: env_var.value for env_var in (main_container.env or [])}
    assert "REQUESTS_CA_BUNDLE" in env_map, (
        "REQUESTS_CA_BUNDLE env var not found on HTTP job pod — CA should be injected unconditionally"
    )
    assert "SSL_CERT_FILE" in env_map, (
        "SSL_CERT_FILE env var not found on HTTP job pod — CA should be injected unconditionally"
    )

    has_ca_volume = any(
        volume.configMap and volume.configMap.name == ODH_TRUSTED_CA_BUNDLE_CONFIGMAP
        for volume in lmevaljob_vllm_emulator_pod.instance.spec.volumes
        if volume.configMap is not None
    )
    assert has_ca_volume, f"No volume referencing '{ODH_TRUSTED_CA_BUNDLE_CONFIGMAP}' ConfigMap found on HTTP job pod"


@pytest.mark.xfail(
    reason="Requires generalized CA bundle injection (RHOAIENG-60453): "
    "unconditional mount of odh-trusted-ca-bundle with SSL_CERT_FILE",
    strict=True,
)
@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-lmeval-verify-cert-has-ca"},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("patched_dsc_kserve_headed")
def test_lmeval_https_verify_certificate_has_ca_bundle(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lmevaljob_vllm_emulator_https_verify_cert_pod: Pod,
):
    """Test CA trust store is mounted regardless of verify_certificate setting.

    Given: A vLLM emulator exposed via HTTPS with verify_certificate explicitly set
    When: LMEvalJob is created with HTTPS base_url and verify_certificate=False
    Then: CA trust store is mounted and env vars set regardless of verify_certificate

    Note: verify_certificate is lm-eval level; trust store is pod level (RHOAIENG-60453).
    """
    main_container = lmevaljob_vllm_emulator_https_verify_cert_pod.instance.spec.containers[0]
    env_map = {env_var.name: env_var.value for env_var in (main_container.env or [])}
    assert "REQUESTS_CA_BUNDLE" in env_map, (
        "REQUESTS_CA_BUNDLE not found — CA should be injected regardless of verify_certificate"
    )
    assert "SSL_CERT_FILE" in env_map, (
        "SSL_CERT_FILE not found — CA should be injected regardless of verify_certificate"
    )

    has_ca_volume = any(
        volume.configMap and volume.configMap.name == ODH_TRUSTED_CA_BUNDLE_CONFIGMAP
        for volume in lmevaljob_vllm_emulator_https_verify_cert_pod.instance.spec.volumes
        if volume.configMap is not None
    )
    assert has_ca_volume, (
        f"No volume referencing '{ODH_TRUSTED_CA_BUNDLE_CONFIGMAP}' found — "
        "CA mount should not be gated on verify_certificate"
    )
