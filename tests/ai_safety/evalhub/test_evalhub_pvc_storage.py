"""PVC as a storage source for evaluation provider test data."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.conftest import build_pvc_job_payload
from tests.ai_safety.evalhub.utils import (
    delete_evalhub_job,
    submit_evalhub_job,
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
        evalhub_vllm_emulator_service: Service,
        evalhub_test_data_populated: PersistentVolumeClaim,
    ) -> None:
        """Given a PVC with test data in the tenant namespace,
        when an evaluation job is submitted with test_data_ref.pvc,
        then the job completes successfully and results are persisted."""
        payload = build_pvc_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="pvc-mount-test",
            claim_name=evalhub_test_data_populated.name,
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        try:
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
                volume
                for volume in (spec.volumes or [])
                if getattr(volume, "persistentVolumeClaim", None) is not None
            ]
            assert len(pvc_volumes) >= 1, (
                f"Expected PVC volume in pod spec, got volumes: {[volume.name for volume in spec.volumes]}"
            )
            pvc_volume = pvc_volumes[0]
            assert pvc_volume.persistentVolumeClaim.claimName == evalhub_test_data_populated.name
            assert pvc_volume.persistentVolumeClaim.readOnly is True

            init_containers = spec.initContainers or []
            init_container_names = [
                container.name for container in init_containers if "init" in container.name.lower()
            ]
            assert "eval-runtime-init" not in init_container_names, (
                "PVC jobs should not have an init container for data download"
            )

            adapter_container = next(
                (container for container in spec.containers if container.name == "adapter"), None
            )
            assert adapter_container is not None, "Expected adapter container in pod spec"
            s3_env_names = {
                env_var.name
                for env_var in (adapter_container.env or [])
                if "AWS" in env_var.name or "S3" in env_var.name.upper()
            }
            assert not s3_env_names, f"PVC jobs should not have S3 credential env vars, found: {s3_env_names}"
        finally:
            delete_evalhub_job(
                host=evalhub_mt_route.host,
                token=tenant_a_token,
                ca_bundle_file=evalhub_mt_ca_bundle_file,
                tenant=tenant_a_namespace.name,
                job_id=job_id,
                hard_delete=True,
            )

    def test_pvc_sub_path_loading(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        evalhub_test_data_populated: PersistentVolumeClaim,
    ) -> None:
        """Given a PVC with data at a specific sub-path,
        when an evaluation job specifies test_data_ref.pvc with sub_path,
        then the job completes successfully using data from that sub-path."""
        payload = build_pvc_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="pvc-sub-path-test",
            claim_name=evalhub_test_data_populated.name,
            sub_path="provider_a",
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

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
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        evalhub_mt_deployment,
    ) -> None:
        """Given a job referencing a PVC that does not exist,
        when the job is submitted,
        then the API accepts it but the job fails because K8s cannot mount the volume."""
        payload = build_pvc_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="pvc-missing-test",
            claim_name="nonexistent-pvc",
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        try:
            job_data = wait_for_evalhub_job(
                host=evalhub_mt_route.host,
                token=tenant_a_token,
                ca_bundle_file=evalhub_mt_ca_bundle_file,
                tenant=tenant_a_namespace.name,
                job_id=job_id,
            )
            assert job_data["status"]["state"] == "failed", (
                f"Expected job to fail with missing PVC, got state: {job_data['status']['state']}"
            )
        finally:
            delete_evalhub_job(
                host=evalhub_mt_route.host,
                token=tenant_a_token,
                ca_bundle_file=evalhub_mt_ca_bundle_file,
                tenant=tenant_a_namespace.name,
                job_id=job_id,
                hard_delete=True,
            )

    def test_pvc_read_only_mount(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        evalhub_test_data_populated: PersistentVolumeClaim,
    ) -> None:
        """Given a PVC-backed evaluation job,
        when the pod spec is inspected,
        then the PVC volume mount has readOnly: true."""
        payload = build_pvc_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="pvc-readonly-test",
            claim_name=evalhub_test_data_populated.name,
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

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

        adapter_container = next(
            (container for container in spec.containers if container.name == "adapter"), None
        )
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
        evalhub_vllm_emulator_service: Service,
        evalhub_test_data_populated: PersistentVolumeClaim,
    ) -> None:
        """Given a PVC with multiple provider datasets at different sub-paths,
        when separate evaluation jobs reference different sub-paths,
        then both jobs complete independently."""
        payload_a = build_pvc_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="pvc-multi-provider-a",
            claim_name=evalhub_test_data_populated.name,
            sub_path="provider_a",
        )
        payload_b = build_pvc_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="pvc-multi-provider-b",
            claim_name=evalhub_test_data_populated.name,
            sub_path="provider_b",
        )

        data_a = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload_a,
        )
        data_b = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload_b,
        )
        job_id_a = data_a["resource"]["id"]
        job_id_b = data_b["resource"]["id"]

        job_data_a = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id_a,
        )
        job_data_b = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id_b,
        )
        validate_evalhub_job_completed(job_data=job_data_a)
        validate_evalhub_job_completed(job_data=job_data_b)
