import json
import time
from typing import Self

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import ArtifactState, RegisteredModelState
from ocp_resources.job import Job

from tests.ai_hub.constants import MODEL_DICT
from tests.ai_hub.model_registry.async_job.constants import (
    ASYNC_UPLOAD_JOB_NAME,
    CA_BUNDLE_CONFIG,
    MODEL_SYNC_CONFIG,
    REPO_NAME,
    TAG,
)
from tests.ai_hub.model_registry.async_job.utils import (
    get_latest_job_pod,
)
from utilities.constants import MinIo, OCIRegistry
from utilities.registry_utils import pull_manifest_from_oci_registry

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [pytest.mark.custom_namespace, pytest.mark.downstream_only]

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
@pytest.mark.parametrize(
    "registered_model_from_image",
    [
        pytest.param(MODEL_DATA, id="test_model_from_image"),
    ],
    indirect=True,
)
class TestAsyncUploadE2E:
    """
    Test for async upload job with real MinIO, OCI registry, Connection Secrets and Model Registry"""

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
        Verify that created IDs are exposed to termination message

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


@pytest.mark.tier2
@pytest.mark.custom_namespace
@pytest.mark.downstream_only
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
@pytest.mark.parametrize(
    "registered_model_from_image",
    [
        pytest.param(MODEL_DATA, id="test_model_from_image"),
    ],
    indirect=True,
)
class TestAsyncJobCABundle:
    """Test async upload job with CA bundle support"""

    def test_async_job_ca_bundle_volume_spec(
        self: Self,
        async_job_with_optional_ca_bundle: Job,
    ) -> None:
        """
        Verify that async job spec includes CA bundle ConfigMap volume with optional flag.

        Given the async job is created
        When the job spec is examined
        Then it should include CA bundle ConfigMap volume marked as optional
        """
        LOGGER.info("Verifying async job CA bundle volume spec")

        job_spec = async_job_with_optional_ca_bundle.instance.spec.template.spec
        volumes = job_spec.volumes or []

        ca_bundle_volumes = [
            volume
            for volume in volumes
            if volume.configMap and volume.configMap.name == CA_BUNDLE_CONFIG["CONFIG_MAP_NAME"]
        ]
        assert ca_bundle_volumes, "CA bundle ConfigMap volume not found in job spec"

        ca_bundle_volume = ca_bundle_volumes[0]
        assert ca_bundle_volume.name == "ca-bundle", f"Expected volume name 'ca-bundle', got {ca_bundle_volume.name}"
        assert ca_bundle_volume.configMap.optional is True, (
            "CA bundle volume should be optional for graceful degradation"
        )

        LOGGER.info("CA bundle volume spec is correct")

    def test_async_job_ca_bundle_volume_mount(
        self: Self,
        async_job_with_optional_ca_bundle: Job,
    ) -> None:
        """
        Verify that async job container includes CA bundle volume mount with correct path.

        Given the async job is created
        When the container spec is examined
        Then it should include CA bundle volume mount at the correct path
        """
        LOGGER.info("Verifying async job CA bundle volume mount")

        job_spec = async_job_with_optional_ca_bundle.instance.spec.template.spec
        container = job_spec.containers[0]
        volume_mounts = container.volumeMounts or []

        ca_bundle_mount = next((mount for mount in volume_mounts if mount.name == "ca-bundle"), None)
        assert ca_bundle_mount, "CA bundle volume mount not found in container spec"
        assert ca_bundle_mount.mountPath == CA_BUNDLE_CONFIG["MOUNT_PATH"], (
            f"Expected mount path {CA_BUNDLE_CONFIG['MOUNT_PATH']}, got {ca_bundle_mount.mountPath}"
        )
        assert ca_bundle_mount.readOnly is True, "CA bundle volume mount should be read-only"

        LOGGER.info("CA bundle volume mount is correct")

    def test_async_job_ca_bundle_env_var(
        self: Self,
        async_job_with_optional_ca_bundle: Job,
    ) -> None:
        """
        Verify that async job container includes CA bundle environment variable.

        Given the async job is created
        When the container environment is examined
        Then it should include MODEL_SYNC_REGISTRY_CUSTOM_CA pointing to the mounted file
        """
        LOGGER.info("Verifying async job CA bundle environment variable")

        job_spec = async_job_with_optional_ca_bundle.instance.spec.template.spec
        container = job_spec.containers[0]
        env_vars = container.env or []

        ca_env_var = next((env_var for env_var in env_vars if env_var.name == CA_BUNDLE_CONFIG["ENV_VAR"]), None)
        assert ca_env_var, f"Environment variable {CA_BUNDLE_CONFIG['ENV_VAR']} not found"

        expected_env_value = f"{CA_BUNDLE_CONFIG['MOUNT_PATH']}/{CA_BUNDLE_CONFIG['CA_FILE']}"
        assert ca_env_var.value == expected_env_value, (
            f"Expected env value {expected_env_value}, got {ca_env_var.value}"
        )

        LOGGER.info("CA bundle environment variable is correct")
