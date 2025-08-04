from typing import Self

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job

from tests.model_registry.async_job.constants import (
    ASYNC_UPLOAD_IMAGE,
    ASYNC_UPLOAD_JOB_NAME,
    MODEL_SYNC_CONFIG,
    PLACEHOLDER_OCI_SECRET_NAME,
    PLACEHOLDER_S3_SECRET_NAME,
)
from tests.model_registry.async_job.utils import (
    get_job_pod,
    validate_job_environment_variables,
    validate_job_labels_and_annotations,
    validate_job_pod_template_labels,
    validate_job_volume_mounts,
)


@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_class",
    "model_registry_mysql_metadata_db",
    "model_registry_instance_mysql",
)
@pytest.mark.custom_namespace
class TestModelRegistryAsyncJob:
    """Test async upload job deployment and configuration"""

    def test_job_creation_and_metadata(self: Self, model_sync_async_job: Job) -> None:
        """Verify Job is created with correct name, labels, and annotations"""
        assert model_sync_async_job.name == ASYNC_UPLOAD_JOB_NAME
        validate_job_labels_and_annotations(job=model_sync_async_job)

    def test_job_pod_template_configuration(self: Self, model_sync_async_job: Job) -> None:
        """Verify Job pod template has correct labels and specifications"""
        validate_job_pod_template_labels(job=model_sync_async_job)

        # Validate restart policy
        restart_policy = model_sync_async_job.instance.spec.template.spec.restartPolicy
        assert restart_policy == "Never"

    def test_job_container_configuration(self: Self, model_sync_async_job: Job) -> None:
        """Verify Job container is configured correctly"""
        container = model_sync_async_job.instance.spec.template.spec.containers[0]

        assert container.name == "async-upload"
        assert container.image == ASYNC_UPLOAD_IMAGE

    def test_job_environment_variables(self: Self, model_sync_async_job: Job) -> None:
        """Verify all required environment variables are configured"""
        validate_job_environment_variables(job=model_sync_async_job)

    def test_job_volume_configuration(self: Self, model_sync_async_job: Job) -> None:
        """Verify Job has correct volume and volume mount configuration"""
        volumes = {vol.name: vol for vol in model_sync_async_job.instance.spec.template.spec.volumes}

        # Validate source credentials volume
        assert "source-credentials" in volumes
        source_vol = volumes["source-credentials"]
        assert source_vol.secret.secretName == PLACEHOLDER_S3_SECRET_NAME

        # Validate destination credentials volume
        assert "destination-credentials" in volumes
        dest_vol = volumes["destination-credentials"]
        assert dest_vol.secret.secretName == PLACEHOLDER_OCI_SECRET_NAME

        # Validate volume mounts
        validate_job_volume_mounts(job=model_sync_async_job)

    @pytest.mark.skipif(
        ASYNC_UPLOAD_IMAGE.startswith("PLACEHOLDER"),
        reason="Downstream image not yet available - job will fail to start",
    )
    def test_job_pod_creation(self: Self, admin_client: DynamicClient, model_sync_async_job: Job) -> None:
        """Verify Job creates a pod (when image is available)"""
        job_pod = get_job_pod(admin_client=admin_client, job=model_sync_async_job)
        assert job_pod.name.startswith(ASYNC_UPLOAD_JOB_NAME)

    def test_job_model_sync_label_propagation(self: Self, model_sync_async_job: Job) -> None:
        """Verify model sync labels are properly set"""
        # Test that model sync labels are present on Job
        job_labels = model_sync_async_job.instance.metadata.labels
        assert job_labels.get("modelregistry.opendatahub.io/job-type") == "async-upload"

        # Test that model ID labels are present on pod template
        pod_labels = model_sync_async_job.instance.spec.template.metadata.labels
        assert pod_labels.get("modelregistry.opendatahub.io/model-sync-model-id") == MODEL_SYNC_CONFIG["MODEL_ID"]
