from typing import Self
import time
import json

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from model_registry.types import ArtifactState, RegisteredModelState
from tests.model_registry.model_registry.async_job.constants import (
    ASYNC_UPLOAD_JOB_NAME,
)
from tests.model_registry.model_registry.async_job.utils import (
    get_latest_job_pod,
)
from tests.model_registry.constants import MODEL_DICT
from utilities.constants import MinIo, OCIRegistry
from utilities.registry_utils import pull_manifest_from_oci_registry
from model_registry import ModelRegistry as ModelRegistryClient
from simple_logger.logger import get_logger
from tests.model_registry.model_registry.async_job.constants import MODEL_SYNC_CONFIG, REPO_NAME, TAG

LOGGER = get_logger(name=__name__)

MODEL_NAME = f"async-test-model-{int(time.time())}"
MODEL_DATA = {
    **MODEL_DICT,
    "model_name": MODEL_NAME,
    "model_storage_key": MODEL_SYNC_CONFIG["SOURCE_AWS_KEY"],
    "model_storage_path": "path/to/test/model",
}


@pytest.mark.parametrize(
    "minio_pod, oci_registry_pod_with_minio",
    [
        pytest.param(
            MinIo.PodConfig.MODEL_REGISTRY_MINIO_CONFIG,
            OCIRegistry.PodConfig.REGISTRY_BASE_CONFIG,
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_namespace",
    "model_registry_metadata_db_resources",
    "minio_pod",
    "create_test_data_in_minio_from_image",
    "s3_secret_for_async_job",
    "oci_secret_for_async_job",
    "oci_registry_pod_with_minio",
    "registered_model_from_image",
)
@pytest.mark.custom_namespace
@pytest.mark.parametrize(
    "registered_model_from_image",
    [
        pytest.param(MODEL_DATA, id="test_model_from_image"),
    ],
    indirect=True,
)
@pytest.mark.downstream_only
class TestAsyncUploadE2E:
    """RHOAIENG-32501: Test for async upload job with real MinIO, OCI registry, Connection Secrets and Model Registry"""

    @pytest.mark.dependency(name="job_creation_and_pod_spawning")
    def test_job_creation_and_pod_spawning(
        self: Self,
        admin_client: DynamicClient,
        model_sync_async_job: Job,
    ) -> None:
        """
        Verify job creation and pod spawning
        """
        LOGGER.info("Verifying job creation and pod spawning")

        # Wait for job to create a pod
        job_pod = get_latest_job_pod(admin_client=admin_client, job=model_sync_async_job)
        assert job_pod.name.startswith(ASYNC_UPLOAD_JOB_NAME)

    @pytest.mark.dependency(name="termination_message_verification", depends=["job_creation_and_pod_spawning"])
    def test_termination_message_contains_created_ids(
        self: Self,
        admin_client: DynamicClient,
        model_sync_async_job: Job,
        model_registry_client: list[ModelRegistryClient],
    ) -> None:
        """
        RHOAIENG-47260: Verify that created IDs are exposed to termination message

        The termination message should contain RegisteredModel, ModelVersion, and ModelArtifact IDs
        that match the actual resources created in the model registry.
        """
        LOGGER.info("Verifying termination message contains created IDs")

        # Get the job pod
        job_pod = get_latest_job_pod(admin_client=admin_client, job=model_sync_async_job)

        # Access container status and termination message
        container_statuses = job_pod.instance.status.containerStatuses

        # Get the main container's termination message
        container = container_statuses[0]
        assert container.state.terminated, "Container should be in terminated state"

        termination_message = container.state.terminated.message
        LOGGER.info(f"Termination message: {termination_message}")

        assert termination_message, "Termination message should not be empty"

        termination_data = json.loads(termination_message)
        assert "RegisteredModel" in termination_data
        assert "ModelVersion" in termination_data
        assert "ModelArtifact" in termination_data

        LOGGER.info(
            f"IDs from termination message - "
            f"RegisteredModel: {termination_data['RegisteredModel']}, "
            f"ModelVersion: {termination_data['ModelVersion']}, "
            f"ModelArtifact: {termination_data['ModelArtifact']}"
        )

        # Verify IDs match actual resources in model registry
        client = model_registry_client[0]

        model = client.get_registered_model(name=MODEL_NAME)
        assert model.id == termination_data["RegisteredModel"]["id"]

        model_version = client.get_model_version(name=MODEL_NAME, version=MODEL_DATA["model_version"])
        assert model_version.id == termination_data["ModelVersion"]["id"]

        model_artifact = client.get_model_artifact(name=MODEL_NAME, version=MODEL_DATA["model_version"])
        assert model_artifact.id == termination_data["ModelArtifact"]["id"]

        LOGGER.info("Successfully verified termination message IDs match actual resources")

    @pytest.mark.dependency(name="oci_registry_verification", depends=["termination_message_verification"])
    def test_oci_registry_verification(
        self: Self,
        oci_registry_host: str,
    ) -> None:
        """
        Verify OCI registry upload
        - Model manifest exists in OCI registry
        - Manifest has correct structure and layers
        """
        LOGGER.info("Verifying OCI registry upload")

        registry_url = f"http://{oci_registry_host}"

        LOGGER.info(f"Verifying artifact in OCI registry: {registry_url}/v2/{REPO_NAME}/manifests/{TAG}")

        # Check if the manifest exists in the OCI registry
        manifest = pull_manifest_from_oci_registry(registry_url=registry_url, repo=REPO_NAME, tag=TAG)

        LOGGER.info("Manifest found in OCI registry")
        LOGGER.info(f"Manifest schema version: {manifest.get('schemaVersion')}")
        LOGGER.info(f"Manifest media type: {manifest.get('mediaType')}")

        # Verify the manifest has the expected structure
        assert "manifests" in manifest, "Manifest should contain manifests section"
        assert len(manifest["manifests"]) > 0, "Manifest should have at least one manifest"
        LOGGER.info(f"Manifest contains {len(manifest['manifests'])} layer(s)")

    @pytest.mark.dependency(name="model_registry_verification", depends=["oci_registry_verification"])
    def test_model_registry_verification(
        self: Self,
        model_registry_client: list[ModelRegistryClient],
        oci_registry_host: str,
    ) -> None:
        """
        Verify model registration
        - Model is registered in model registry
        - Model artifact has correct attributes
        """
        LOGGER.info("Verifying model registry model and artifact")

        # Verify model registry metadata was updated
        client = model_registry_client[0]
        model = client.get_registered_model(name=MODEL_NAME)
        assert model.state == RegisteredModelState.LIVE

        model_artifact = client.get_model_artifact(name=MODEL_NAME, version=MODEL_DATA["model_version"])

        # Validate model artifact attributes
        assert model_artifact.name == MODEL_NAME
        assert model_artifact.state == ArtifactState.LIVE
        assert model_artifact.uri == f"oci://{oci_registry_host}/{REPO_NAME}"
        assert model_artifact.storage_key == MODEL_DATA["model_storage_key"]
        assert model_artifact.storage_path == MODEL_DATA["model_storage_path"]

        LOGGER.info("Async upload job test with KSERVE_MINIO_IMAGE: PASSED")
