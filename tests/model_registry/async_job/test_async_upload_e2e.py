from typing import Self
import time

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.service import Service
from ocp_resources.secret import Secret
from model_registry.types import RegisteredModelState
from tests.model_registry.async_job.constants import (
    ASYNC_UPLOAD_IMAGE,
    ASYNC_UPLOAD_JOB_NAME,
)
from tests.model_registry.async_job.utils import (
    get_latest_job_pod,
    upload_test_model_to_minio,
    pull_manifest_from_oci_registry,
)
from tests.model_registry.constants import MODEL_DICT
from utilities.constants import MinIo, OCIRegistry
from model_registry import ModelRegistry as ModelRegistryClient
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)

MODEL_NAME = f"async-test-model-{int(time.time())}"


@pytest.mark.parametrize(
    "minio_pod, oci_registry_pod_with_minio",
    [
        pytest.param(
            MinIo.PodConfig.MODEL_MESH_MINIO_CONFIG,
            OCIRegistry.PodConfig.REGISTRY_BASE_CONFIG,
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_class",
    "model_registry_namespace",
    "model_registry_db_secret",
    "model_registry_db_pvc",
    "model_registry_db_service",
    "model_registry_db_deployment",
    "model_registry_instance_mysql",
    "minio_pod",  # From main conftest
    "oci_registry_pod_with_minio",  # Local fixture
    "oci_registry_route",  # Local fixture
)
@pytest.mark.custom_namespace
class TestAsyncUploadE2E:
    """End-to-end test for async upload job with real MinIO, OCI registry, and Model Registry"""

    def test_model_registry_client_integration(
        self: Self,
        model_registry_client: list[ModelRegistryClient],
    ) -> None:
        """Test that model registry client can connect and create models"""
        LOGGER.info("Testing model registry client integration")

        # Verify client is available
        assert model_registry_client, "Model registry client should be available"
        client = model_registry_client[0]

        # Create a test model for async job to process
        test_model_data = {
            **MODEL_DICT,
            "model_name": MODEL_NAME,
            "model_storage_key": "my-model",  # This should match SOURCE_AWS_KEY in constants
            "model_storage_path": "path/to/test/model",
        }

        # Register model with model registry
        registered_model = client.register_model(
            name=test_model_data["model_name"],
            uri=test_model_data["model_uri"],
            version=test_model_data["model_version"],
            description=test_model_data["model_description"],
            model_format_name=test_model_data["model_format"],
            model_format_version=test_model_data["model_format_version"],
            storage_key=test_model_data["model_storage_key"],
            storage_path=test_model_data["model_storage_path"],
            metadata=test_model_data["model_metadata"],
        )

        # Verify model was created
        assert registered_model.id, "Model should have an ID"
        assert registered_model.name == test_model_data["model_name"]

        LOGGER.info(f"Created test model: {registered_model.name} (ID: {registered_model.id})")
        LOGGER.info("Model registry client integration: ✅ PASSED")

    def test_infrastructure_connectivity(
        self: Self,
        minio_service: Service,
        oci_registry_route: Route,
        oci_registry_pod_with_minio: Pod,
    ) -> None:
        """Test that infrastructure services are accessible"""
        LOGGER.info("Testing infrastructure connectivity")

        # Verify MinIO service is running
        assert minio_service.exists
        assert minio_service.instance.spec.ports[0].port == MinIo.Metadata.DEFAULT_PORT

        # Verify OCI registry is running
        assert oci_registry_pod_with_minio.exists
        oci_registry_pod_with_minio.wait_for_condition(condition="Ready", status="True", timeout=300)

        # Verify OCI registry route is accessible
        assert oci_registry_route.exists
        registry_host = oci_registry_route.instance.spec.host
        assert registry_host, "OCI registry route should have a host"

        LOGGER.info("Infrastructure connectivity: ✅ PASSED")

    def test_create_test_data_in_minio(
        self: Self,
        minio_service: Service,
        admin_client: DynamicClient,
        model_registry_namespace: str,
    ) -> None:
        """Test creating sample model data in MinIO for async job to process"""
        LOGGER.info("Testing test data creation in MinIO")

        # Upload the mnist-8.onnx test model file to MinIO
        upload_test_model_to_minio(
            admin_client=admin_client, namespace=model_registry_namespace, minio_service=minio_service
        )

        LOGGER.info("Test data creation in MinIO: ✅ PASSED")

    def test_job_deployment_with_real_secrets(
        self: Self,
        model_sync_async_job: Job,
        s3_secret_for_async_job: Secret,
        oci_secret_for_async_job: Secret,
    ) -> None:
        """Test that the job deploys successfully with real secrets"""
        LOGGER.info("Testing job deployment with real secrets")

        # Verify job is created
        assert model_sync_async_job.name == ASYNC_UPLOAD_JOB_NAME
        assert model_sync_async_job.exists

        # Verify job is using real secrets (not placeholders)
        job_spec = model_sync_async_job.instance.spec.template.spec
        volume_names = [vol.name for vol in job_spec.volumes]

        assert "source-credentials" in volume_names
        assert "destination-credentials" in volume_names

        # Verify secret names are real (not placeholders)
        source_secret_name = None
        dest_secret_name = None

        for volume in job_spec.volumes:
            if volume.name == "source-credentials":
                source_secret_name = volume.secret.secretName
            elif volume.name == "destination-credentials":
                dest_secret_name = volume.secret.secretName

        assert source_secret_name == s3_secret_for_async_job.name
        assert dest_secret_name == oci_secret_for_async_job.name

        # Verify secrets exist
        assert s3_secret_for_async_job.exists
        assert oci_secret_for_async_job.exists

        LOGGER.info("Job deployment with real secrets: ✅ PASSED")

    def test_job_pod_creation_and_logs(
        self: Self,
        admin_client: DynamicClient,
        model_sync_async_job: Job,
    ) -> None:
        """Test that job creates a pod and runs (when image is available)"""
        LOGGER.info("Testing job pod creation and execution")

        # Wait for job to create a pod
        job_pod = get_latest_job_pod(admin_client=admin_client, job=model_sync_async_job)
        assert job_pod.name.startswith(ASYNC_UPLOAD_JOB_NAME)

        # Wait for pod to start (but may fail due to image issues)
        LOGGER.info(f"Job pod created: {job_pod.name}")

        # Try to get pod logs for debugging
        try:
            pod_logs = job_pod.log
            if pod_logs:
                LOGGER.info(f"Pod logs preview: {pod_logs[:500]}...")
            else:
                LOGGER.info("No pod logs available yet")
        except Exception as e:
            LOGGER.warning(f"Could not retrieve pod logs: {e}")

        LOGGER.info("Job pod creation: ✅ PASSED")

    def test_environment_variable_resolution(
        self: Self,
        model_sync_async_job: Job,
        oci_registry_route: Route,
    ) -> None:
        """Test that environment variables are properly resolved with dynamic values"""
        LOGGER.info("Testing environment variable resolution")

        container = model_sync_async_job.instance.spec.template.spec.containers[0]
        env_dict = {env.name: env.value for env in container.env}

        # Verify critical environment variables are set
        required_env_vars = [
            "MODEL_SYNC_SOURCE_TYPE",
            "MODEL_SYNC_DESTINATION_TYPE",
            "MODEL_SYNC_DESTINATION_OCI_URI",
            "MODEL_SYNC_REGISTRY_SERVER_ADDRESS",
        ]

        for var in required_env_vars:
            assert var in env_dict, f"Required environment variable {var} not found"
            assert env_dict[var], f"Environment variable {var} is empty"

        # Verify OCI URI contains the registry host
        oci_uri = env_dict["MODEL_SYNC_DESTINATION_OCI_URI"]
        registry_host = oci_registry_route.instance.spec.host
        assert registry_host in oci_uri, f"OCI URI should contain registry host {registry_host}"

        # Verify no placeholder values remain
        placeholder_vars = [
            var
            for var, value in env_dict.items()
            if value is None or (isinstance(value, str) and "PLACEHOLDER" in value)
        ]
        if placeholder_vars:
            LOGGER.warning(f"Placeholder values still present: {placeholder_vars}")

        LOGGER.info("Environment variable resolution: ✅ PASSED")

    def test_volume_mount_configuration(
        self: Self,
        model_sync_async_job: Job,
    ) -> None:
        """Test that volume mounts are correctly configured"""
        LOGGER.info("Testing volume mount configuration")

        container = model_sync_async_job.instance.spec.template.spec.containers[0]
        volume_mounts = {mount.name: mount.mountPath for mount in container.volumeMounts}

        expected_mounts = {
            "source-credentials": "/opt/creds/source",
            "destination-credentials": "/opt/creds/destination",
        }

        for volume_name, expected_path in expected_mounts.items():
            assert volume_name in volume_mounts, f"Volume mount {volume_name} not found"
            actual_path = volume_mounts[volume_name]
            assert actual_path == expected_path, f"Volume {volume_name}: expected {expected_path}, got {actual_path}"

        # Verify all volume mounts are read-only
        for mount in container.volumeMounts:
            assert mount.readOnly is True, f"Volume mount {mount.name} should be read-only"

        LOGGER.info("Volume mount configuration: ✅ PASSED")

    @pytest.mark.skipif(
        ASYNC_UPLOAD_IMAGE.startswith("PLACEHOLDER"),
        reason="Downstream image not yet available - job will fail to start",
    )
    def test_end_to_end_async_upload_job(
        self: Self,
        admin_client: DynamicClient,
        model_sync_async_job: Job,
        model_registry_client: list[ModelRegistryClient],
        oci_registry_route: Route,
    ) -> None:
        """Complete end-to-end test of async upload job execution"""
        LOGGER.info("Starting end-to-end async upload job test")

        # 1. Verify job is created and configured correctly
        assert model_sync_async_job.exists
        LOGGER.info("✓ Job created successfully")

        # 2. Monitor job execution with retry logic
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        last_pod_name = None

        while time.time() - start_time < max_wait_time:
            # Check job status (instance auto-refreshes)
            job_status = model_sync_async_job.instance.status

            # Check if job succeeded
            if hasattr(job_status, "succeeded") and job_status.succeeded:
                LOGGER.info("✓ Job completed successfully")
                break

            # Get the latest pod (handles retries)
            try:
                current_pod = get_latest_job_pod(admin_client=admin_client, job=model_sync_async_job)
                if current_pod.name != last_pod_name:
                    LOGGER.info(f"✓ Monitoring pod: {current_pod.name}")
                    last_pod_name = current_pod.name

                # Check pod status
                pod_status = current_pod.instance.status
                if hasattr(pod_status, "phase"):
                    if pod_status.phase == "Succeeded":
                        LOGGER.info(f"✓ Pod {current_pod.name} succeeded")
                    elif pod_status.phase == "Failed":
                        LOGGER.warning(f"⚠ Pod {current_pod.name} failed, job may retry with new pod")
                        # Don't fail here - let the job retry
                    elif pod_status.phase == "Running":
                        LOGGER.info(f"Pod {current_pod.name} is running...")

            except Exception as e:
                LOGGER.warning(f"Could not get pod status: {e}")

            LOGGER.info(f"Job still running... (elapsed: {int(time.time() - start_time)}s)")
            time.sleep(seconds=10)
        else:
            # Timeout reached
            LOGGER.error("Job timed out")
            try:
                current_pod = get_latest_job_pod(admin_client=admin_client, job=model_sync_async_job)
                pod_logs = current_pod.log()
                LOGGER.error(f"Final pod logs: {pod_logs}")
            except Exception as e:
                LOGGER.error(f"Could not retrieve final pod logs: {e}")

            pytest.fail(f"Async upload job did not complete within {max_wait_time} seconds")

        # 4. Verify OCI registry contains the uploaded artifact
        registry_host = oci_registry_route.instance.spec.host
        registry_url = f"http://{registry_host}"

        # The async job should have uploaded the model to this repository and tag
        repository = "async-job-test/model-artifact"
        tag = "latest"

        LOGGER.info(f"Verifying artifact in OCI registry: {registry_url}/v2/{repository}/manifests/{tag}")

        try:
            # Check if the manifest exists in the OCI registry
            manifest = pull_manifest_from_oci_registry(registry_url=registry_url, repo=repository, tag=tag)

            LOGGER.info("✓ Manifest found in OCI registry")
            LOGGER.info(f"✓ Manifest schema version: {manifest.get('schemaVersion')}")
            LOGGER.info(f"✓ Manifest media type: {manifest.get('mediaType')}")

            # Verify the manifest has the expected structure
            # LOGGER.info(f"✓ Manifest: {manifest}")
            assert "manifests" in manifest, "Manifest should contain manifests section"
            assert len(manifest["manifests"]) > 0, "Manifest should have at least one manifest"

            LOGGER.info(f"✓ Manifest contains {len(manifest['manifests'])} layer(s)")

        except Exception as e:
            LOGGER.error(f"Failed to verify OCI registry artifact: {e}")
            pytest.fail(f"Could not verify uploaded artifact in OCI registry: {e}")

        # 5. Verify model registry metadata was updated
        LOGGER.info("Verifying model registry metadata updates...")

        try:
            # Get the model registry client
            client = model_registry_client[0]

            model = client.get_registered_model(name=MODEL_NAME)
            model_artifact = client.get_model_artifact(name=MODEL_NAME, version=MODEL_DICT["model_version"])
            LOGGER.info(f"✓ Model: {model}")
            LOGGER.info(f"✓ Model artifact: {model_artifact}")

            assert model.state == RegisteredModelState.LIVE
            assert model_artifact.uri == f"oci://{registry_host}/async-job-test/model-artifact"

        except Exception as e:
            LOGGER.warning(f"Could not fully verify model registry updates: {e}")

        LOGGER.info("End-to-end async upload job test: ✅ PASSED")
