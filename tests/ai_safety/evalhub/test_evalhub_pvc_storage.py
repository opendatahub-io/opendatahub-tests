"""PVC as a storage source for evaluation provider test data."""

from collections.abc import Callable

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.route import Route

from tests.ai_safety.evalhub.utils import (
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
    wait_for_evalhub_runtime_job_count,
)

PVC_MODEL_NAMESPACE = pytest.param({"name": "test-evalhub-pvc-storage"})


@pytest.mark.parametrize("model_namespace", [PVC_MODEL_NAMESPACE], indirect=True)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubPVCStorage:
    """PVC-backed test data source for evaluation jobs."""

    def test_pvc_mount_job_completes(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_pvc_job: Callable[..., str],
        evalhub_test_data_populated: PersistentVolumeClaim,
    ) -> None:
        """Given a PVC with test data in the tenant namespace,
        when an evaluation job is submitted with test_data_ref.pvc,
        then the job completes successfully and results are persisted."""
        job_id = submit_pvc_job(
            claim_name=evalhub_test_data_populated.name,
            job_name="pvc-mount-test",
        )
        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        validate_evalhub_job_completed(job_data=job_data)

        batch_jobs = wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )
        batch_job = batch_jobs[0]
        spec = batch_job.instance.spec.template.spec

        pvc_volumes = [
            volume for volume in (spec.volumes or []) if getattr(volume, "persistentVolumeClaim", None) is not None
        ]
        assert len(pvc_volumes) >= 1, (
            f"Expected PVC volume in pod spec, got volumes: {[volume.name for volume in spec.volumes]}"
        )
        pvc_volume = pvc_volumes[0]
        assert pvc_volume.persistentVolumeClaim.claimName == evalhub_test_data_populated.name
        assert pvc_volume.persistentVolumeClaim.readOnly is True

        init_containers = spec.initContainers or []
        init_container_names = [container.name for container in init_containers if "init" in container.name.lower()]
        assert "eval-runtime-init" not in init_container_names, (
            "PVC jobs should not have an init container for data download"
        )

        adapter_container = next((container for container in spec.containers if container.name == "adapter"), None)
        assert adapter_container is not None, "Expected adapter container in pod spec"
        s3_env_names = {
            env_var.name
            for env_var in (adapter_container.env or [])
            if "AWS" in env_var.name or "S3" in env_var.name.upper()
        }
        assert not s3_env_names, f"PVC jobs should not have S3 credential env vars, found: {s3_env_names}"

    def test_pvc_sub_path_loading(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_pvc_job: Callable[..., str],
        evalhub_test_data_populated: PersistentVolumeClaim,
    ) -> None:
        """Given a PVC with data at a specific sub-path,
        when an evaluation job specifies test_data_ref.pvc with sub_path,
        then the job completes successfully using data from that sub-path."""
        job_id = submit_pvc_job(
            claim_name=evalhub_test_data_populated.name,
            job_name="pvc-sub-path-test",
            sub_path="provider_a",
        )
        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        validate_evalhub_job_completed(job_data=job_data)

    def test_missing_pvc_job_fails(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_pvc_job: Callable[..., str],
        evalhub_mt_deployment,
    ) -> None:
        """Given a job referencing a PVC that does not exist,
        when the job is submitted,
        then the pod spec references the missing PVC and the job fails."""
        job_id = submit_pvc_job(
            claim_name="nonexistent-pvc",
            job_name="pvc-missing-test",
        )
        batch_jobs = wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )
        batch_job = batch_jobs[0]
        spec = batch_job.instance.spec.template.spec

        pvc_volumes = [
            volume for volume in (spec.volumes or []) if getattr(volume, "persistentVolumeClaim", None) is not None
        ]
        assert len(pvc_volumes) >= 1, (
            f"Expected PVC volume referencing nonexistent-pvc in pod spec, "
            f"got volumes: {[v.name for v in spec.volumes]}"
        )
        assert pvc_volumes[0].persistentVolumeClaim.claimName == "nonexistent-pvc"

        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        assert job_data.get("status", {}).get("state") == "failed", "Job referencing a nonexistent PVC should fail"

    def test_pvc_read_only_mount(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_pvc_job: Callable[..., str],
        evalhub_test_data_populated: PersistentVolumeClaim,
    ) -> None:
        """Given a PVC-backed evaluation job,
        when the pod spec is inspected,
        then the PVC volume mount has readOnly: true."""
        job_id = submit_pvc_job(
            claim_name=evalhub_test_data_populated.name,
            job_name="pvc-readonly-test",
        )
        batch_jobs = wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )
        batch_job = batch_jobs[0]
        spec = batch_job.instance.spec.template.spec

        pvc_volumes = [
            volume for volume in (spec.volumes or []) if getattr(volume, "persistentVolumeClaim", None) is not None
        ]
        assert len(pvc_volumes) >= 1, "Expected PVC volume in pod spec"
        assert pvc_volumes[0].persistentVolumeClaim.readOnly is True, "PVC must be mounted read-only"

        adapter_container = next((container for container in spec.containers if container.name == "adapter"), None)
        assert adapter_container is not None
        pvc_mount = next(
            (mount for mount in (adapter_container.volumeMounts or []) if mount.name == pvc_volumes[0].name),
            None,
        )
        assert pvc_mount is not None, "Adapter container should have the PVC volume mount"
        assert pvc_mount.readOnly is True, "Adapter PVC volume mount must be read-only"

        job_data = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        validate_evalhub_job_completed(job_data=job_data)

    def test_multiple_providers_same_pvc(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        submit_pvc_job: Callable[..., str],
        evalhub_test_data_populated: PersistentVolumeClaim,
    ) -> None:
        """Given a PVC with multiple provider datasets at different sub-paths,
        when separate evaluation jobs reference different sub-paths sequentially,
        then both jobs complete independently.

        Jobs run sequentially because the PVC uses ReadWriteOnce access mode
        (EBS gp3-csi), which does not support multi-node attachment."""
        job_id_a = submit_pvc_job(
            claim_name=evalhub_test_data_populated.name,
            job_name="pvc-multi-provider-a",
            sub_path="provider_a",
        )
        job_data_a = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id_a,
        )
        validate_evalhub_job_completed(job_data=job_data_a)

        job_id_b = submit_pvc_job(
            claim_name=evalhub_test_data_populated.name,
            job_name="pvc-multi-provider-b",
            sub_path="provider_b",
        )
        job_data_b = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id_b,
        )
        validate_evalhub_job_completed(job_data=job_data_b)
