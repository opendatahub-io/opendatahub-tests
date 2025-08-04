from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.pod import Pod

from tests.model_registry.async_job.constants import (
    ASYNC_JOB_ANNOTATIONS,
    ASYNC_JOB_LABELS,
    MODEL_SYNC_CONFIG,
    VOLUME_MOUNTS,
)


def get_job_pod(admin_client: DynamicClient, job: Job) -> Pod:
    """Get the Pod created by a Job"""
    pods = list(
        Pod.get(
            dyn_client=admin_client,
            namespace=job.namespace,
            label_selector=f"job-name={job.name}",
        )
    )
    assert len(pods) == 1, f"Expected 1 pod for job {job.name}, found {len(pods)}"
    return pods[0]


def validate_job_labels_and_annotations(job: Job) -> None:
    """Validate Job has all required labels and annotations"""
    job_labels = job.instance.metadata.labels or {}
    job_annotations = job.instance.metadata.annotations or {}

    # Validate required labels
    for key, expected_value in ASYNC_JOB_LABELS.items():
        actual_value = job_labels.get(key)
        assert actual_value == expected_value, f"Label {key}: expected {expected_value}, got {actual_value}"

    # Validate required annotations
    for key, expected_value in ASYNC_JOB_ANNOTATIONS.items():
        actual_value = job_annotations.get(key)
        assert actual_value == expected_value, f"Annotation {key}: expected {expected_value}, got {actual_value}"


def validate_job_pod_template_labels(job: Job) -> None:
    """Validate Job pod template has required model sync labels"""
    pod_template_labels = job.instance.spec.template.metadata.labels or {}

    expected_model_labels = {
        "modelregistry.opendatahub.io/model-sync-model-id": MODEL_SYNC_CONFIG["MODEL_ID"],
        "modelregistry.opendatahub.io/model-sync-model-version-id": MODEL_SYNC_CONFIG["MODEL_VERSION_ID"],
        "modelregistry.opendatahub.io/model-sync-model-artifact-id": MODEL_SYNC_CONFIG["MODEL_ARTIFACT_ID"],
    }

    for key, expected_value in expected_model_labels.items():
        actual_value = pod_template_labels.get(key)
        assert actual_value == expected_value, (
            f"Pod template label {key}: expected {expected_value}, got {actual_value}"
        )


def validate_job_environment_variables(job: Job) -> None:
    """Validate Job container has all required environment variables"""
    container = job.instance.spec.template.spec.containers[0]
    env_dict = {env.name: env.value for env in container.env}

    required_env_vars = [
        "MODEL_SYNC_SOURCE_TYPE",
        "MODEL_SYNC_DESTINATION_TYPE",
        "MODEL_SYNC_MODEL_ID",
        "MODEL_SYNC_MODEL_VERSION_ID",
        "MODEL_SYNC_MODEL_ARTIFACT_ID",
    ]

    for var in required_env_vars:
        assert var in env_dict, f"Required environment variable {var} not found"
        assert env_dict[var], f"Environment variable {var} is empty"


def validate_job_volume_mounts(job: Job) -> None:
    """Validate Job container has correct volume mounts"""
    container = job.instance.spec.template.spec.containers[0]
    volume_mounts = {mount.name: mount.mountPath for mount in container.volumeMounts}

    expected_mounts = {
        "source-credentials": VOLUME_MOUNTS["SOURCE_CREDS_PATH"],
        "destination-credentials": VOLUME_MOUNTS["DEST_CREDS_PATH"],
    }

    for volume_name, expected_path in expected_mounts.items():
        actual_path = volume_mounts.get(volume_name)
        assert actual_path == expected_path, f"Volume {volume_name}: expected {expected_path}, got {actual_path}"
