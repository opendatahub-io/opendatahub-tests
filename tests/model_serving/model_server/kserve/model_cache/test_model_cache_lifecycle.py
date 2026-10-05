"""Tier 1 tests for KServe local model namespace cache."""

from typing import Any

import pytest
import shortuuid
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume import PersistentVolume
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_server.kserve.model_cache.utils import (
    LOCAL_MODEL_NODE_GROUP_NAME,
    MINT_ONNX_STORAGE_PATH,
    MODEL_CACHE_HOST_PATH,
    MODEL_CACHE_NODE_PVC_NAME,
    MODEL_CACHE_SIZE,
    MODEL_CACHE_STORAGE_CLASS,
    LocalModelNamespaceCache,
    assert_predictor_uses_cached_pvc,
    cache_status_dict,
    wait_for_local_model_cache_nodes_downloaded,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    Protocols,
    RunTimeConfigs,
)
from utilities.inference_utils import Inference, create_isvc
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.tier1,
    pytest.mark.rawdeployment,
    pytest.mark.usefixtures("valid_aws_config", "skip_if_disconnected"),
]


class TestModelCacheDeletion:
    """Tier 1: deleting a LocalModelNamespaceCache removes the CR and its PVCs."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "unprivileged_model_namespace, ovms_kserve_serving_runtime",
        [
            pytest.param(
                {"name": "kserve-cache-deletion"},
                RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
                id="test_cache_deletion_cleanup",
            )
        ],
        indirect=True,
    )
    def test_cache_deletion_cleans_up_resources(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        ovms_kserve_serving_runtime: ServingRuntime,
        model_cache_infra_ready: DataScienceCluster,
        model_cache_download_s3_secret: Secret,
        ci_s3_bucket_name: str,
    ) -> None:
        """Given a NodeDownloaded cache with no bound ISVCs,
        when the cache CR is deleted,
        then the CR is removed and PVCs in both the CR and operator namespaces are cleaned up.
        """
        ns_name = unprivileged_model_namespace.name
        apps_ns: str = py_config["applications_namespace"]
        cache_name = f"del-test-{shortuuid.uuid()[:10].lower()}"
        source_uri = f"s3://{ci_s3_bucket_name}/{MINT_ONNX_STORAGE_PATH}/"

        pvcs_before_cr_ns = {pvc.name for pvc in PersistentVolumeClaim.get(dyn_client=admin_client, namespace=ns_name)}
        pvcs_before_apps_ns = {
            pvc.name for pvc in PersistentVolumeClaim.get(dyn_client=admin_client, namespace=apps_ns)
        }

        cache = LocalModelNamespaceCache(
            client=admin_client,
            name=cache_name,
            namespace=ns_name,
            source_model_uri=source_uri,
            model_size="100Mi",
            node_groups=[LOCAL_MODEL_NODE_GROUP_NAME],
            storage={"key": model_cache_download_s3_secret.name},
            teardown=False,
        )
        try:
            cache.deploy()
            wait_for_local_model_cache_nodes_downloaded(cache=cache, timeout=600)

            pvcs_after_cr_ns = {
                pvc.name for pvc in PersistentVolumeClaim.get(dyn_client=admin_client, namespace=ns_name)
            }
            pvcs_after_apps_ns = {
                pvc.name for pvc in PersistentVolumeClaim.get(dyn_client=admin_client, namespace=apps_ns)
            }
            new_pvcs_cr = pvcs_after_cr_ns - pvcs_before_cr_ns
            new_pvcs_apps = pvcs_after_apps_ns - pvcs_before_apps_ns

            cache.clean_up()

            try:
                for sample in TimeoutSampler(
                    wait_timeout=120,
                    sleep=10,
                    func=lambda: not cache.exists,
                ):
                    if sample:
                        break
            except TimeoutExpiredError:
                pytest.fail(f"LocalModelNamespaceCache '{cache_name}' still exists {120}s after deletion")

            all_new_pvcs = [(name, ns_name) for name in new_pvcs_cr] + [(name, apps_ns) for name in new_pvcs_apps]
            for pvc_name, pvc_ns in all_new_pvcs:
                pvc_ref = PersistentVolumeClaim(client=admin_client, name=pvc_name, namespace=pvc_ns)
                try:
                    for sample in TimeoutSampler(
                        wait_timeout=120,
                        sleep=10,
                        func=lambda p=pvc_ref: not p.exists,
                    ):
                        if sample:
                            break
                except TimeoutExpiredError:
                    pytest.fail(f"PVC '{pvc_name}' in '{pvc_ns}' still exists after cache deletion")

        finally:
            if cache.exists:
                cache.clean_up()


class TestModelCacheReuse:
    """Tier 1: a second ISVC reuses an existing cache; namespace isolation holds."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "unprivileged_model_namespace, ovms_kserve_serving_runtime",
        [
            pytest.param(
                {"name": "kserve-cache-reuse"},
                RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
                id="test_second_isvc_reuses_cache",
            )
        ],
        indirect=True,
    )
    def test_second_isvc_reuses_existing_cache(
        self,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        ovms_kserve_serving_runtime: ServingRuntime,
        mnist_local_model_cache: LocalModelNamespaceCache,
        mnist_onnx_local_model_cache_inference_service: InferenceService,
        ci_s3_bucket_name: str,
    ) -> None:
        """Given a cached model with one bound ISVC,
        when a second ISVC with the same storageUri is deployed,
        then both ISVCs use PVC-backed storage and cache status lists both.
        """
        first_isvc = mnist_onnx_local_model_cache_inference_service
        model_format_name: str = ovms_kserve_serving_runtime.instance.spec.supportedModelFormats[0].name

        with create_isvc(
            client=unprivileged_client,
            name=f"{Protocols.HTTP}-{ModelFormat.ONNX}-lmcache-2",
            namespace=unprivileged_model_namespace.name,
            runtime=ovms_kserve_serving_runtime.name,
            storage_uri=f"s3://{ci_s3_bucket_name}/{MINT_ONNX_STORAGE_PATH}/",
            model_format=model_format_name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            external_route=True,
            timeout=900,
        ) as second_isvc:
            assert_predictor_uses_cached_pvc(
                client=unprivileged_client,
                isvc=second_isvc,
                runtime_name=ovms_kserve_serving_runtime.name,
            )

            verify_inference_response(
                inference_service=second_isvc,
                inference_config=ONNX_INFERENCE_CONFIG,
                inference_type=Inference.INFER,
                protocol=Protocols.HTTPS,
                use_default_query=True,
            )

            mnist_local_model_cache.get()
            status = cache_status_dict(cache=mnist_local_model_cache)
            bound_names = {
                ref["name"]
                for ref in (status.get("inferenceServices") or [])
                if ref.get("namespace") == unprivileged_model_namespace.name
            }
            assert first_isvc.name in bound_names, (
                f"First ISVC '{first_isvc.name}' not listed in cache inferenceServices; got {bound_names!r}"
            )
            assert second_isvc.name in bound_names, (
                f"Second ISVC '{second_isvc.name}' not listed in cache inferenceServices; got {bound_names!r}"
            )

    @pytest.mark.parametrize(
        "unprivileged_model_namespace, ovms_kserve_serving_runtime",
        [
            pytest.param(
                {"name": "kserve-cache-reuse"},
                RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
                id="test_cache_namespace_isolation",
            )
        ],
        indirect=True,
    )
    def test_cache_namespace_isolation(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        ovms_kserve_serving_runtime: Any,
        mnist_local_model_cache: LocalModelNamespaceCache,
    ) -> None:
        """Given a cache in the test namespace,
        when querying for it in the 'default' namespace,
        then the same-named cache does not exist there.
        """
        cache_ns = mnist_local_model_cache.namespace
        assert cache_ns == unprivileged_model_namespace.name

        cross_ns_cache = LocalModelNamespaceCache(
            client=admin_client,
            name=mnist_local_model_cache.name,
            namespace="default",
        )
        assert not cross_ns_cache.exists, (
            f"Cache '{mnist_local_model_cache.name}' should be isolated to namespace "
            f"'{cache_ns}' but was found in 'default'"
        )


class TestModelCacheStorageClass:
    """Tier 1: model cache PVCs use the expected StorageClass and capacity."""

    @pytest.mark.parametrize(
        "unprivileged_model_namespace, ovms_kserve_serving_runtime",
        [
            pytest.param(
                {"name": "kserve-cache-reuse"},
                RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
                id="test_node_pvc_storage_class",
            )
        ],
        indirect=True,
    )
    def test_node_pvc_uses_local_storage(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        ovms_kserve_serving_runtime: Any,
        model_cache_infra_ready: DataScienceCluster,
    ) -> None:
        """Given model cache infrastructure is enabled,
        when the node-level PVC is inspected,
        then it uses the local-storage StorageClass with the configured cacheSize.
        """
        apps_ns: str = py_config["applications_namespace"]
        node_pvc = PersistentVolumeClaim(
            client=admin_client,
            name=MODEL_CACHE_NODE_PVC_NAME,
            namespace=apps_ns,
        )
        assert node_pvc.exists, f"Node PVC '{MODEL_CACHE_NODE_PVC_NAME}' not found in '{apps_ns}'"
        node_pvc.get()

        sc_name = node_pvc.instance.spec.storageClassName
        assert sc_name == MODEL_CACHE_STORAGE_CLASS, (
            f"Node PVC StorageClass is '{sc_name}', expected '{MODEL_CACHE_STORAGE_CLASS}'"
        )

        requested_storage = node_pvc.instance.spec.resources.requests.get("storage")
        assert requested_storage == MODEL_CACHE_SIZE, (
            f"Node PVC capacity is '{requested_storage}', expected '{MODEL_CACHE_SIZE}'"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "unprivileged_model_namespace, ovms_kserve_serving_runtime",
        [
            pytest.param(
                {"name": "kserve-cache-reuse"},
                RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
                id="test_download_pv_storage_class",
            )
        ],
        indirect=True,
    )
    def test_download_pv_uses_local_storage_and_host_path(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        ovms_kserve_serving_runtime: Any,
        mnist_local_model_cache: LocalModelNamespaceCache,
    ) -> None:
        """Given a cache that has reached NodeDownloaded,
        when the download PV is inspected,
        then it uses the local-storage StorageClass and the correct hostPath.
        """
        cache_name = mnist_local_model_cache.name
        ns_name = unprivileged_model_namespace.name
        download_pv_name = f"{cache_name}-{LOCAL_MODEL_NODE_GROUP_NAME}-{ns_name}-download"

        pv = PersistentVolume(client=admin_client, name=download_pv_name)
        assert pv.exists, f"Download PV '{download_pv_name}' not found"
        pv.get()

        sc_name = pv.instance.spec.storageClassName
        assert sc_name == MODEL_CACHE_STORAGE_CLASS, (
            f"Download PV StorageClass is '{sc_name}', expected '{MODEL_CACHE_STORAGE_CLASS}'"
        )

        host_path = getattr(pv.instance.spec, "hostPath", None) or {}
        pv_path = host_path.get("path") if isinstance(host_path, dict) else getattr(host_path, "path", None)
        assert pv_path == MODEL_CACHE_HOST_PATH, (
            f"Download PV hostPath is '{pv_path}', expected '{MODEL_CACHE_HOST_PATH}'"
        )

        capacity = pv.instance.spec.capacity.get("storage")
        assert capacity == MODEL_CACHE_SIZE, f"Download PV capacity is '{capacity}', expected '{MODEL_CACHE_SIZE}'"


class TestModelCacheInvalidCredentials:
    """Tier 1: cache with invalid S3 credentials never reaches NodeDownloaded."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "unprivileged_model_namespace, ovms_kserve_serving_runtime",
        [
            pytest.param(
                {"name": "kserve-cache-invalid-creds"},
                RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
                id="test_invalid_credentials_cache",
            )
        ],
        indirect=True,
    )
    def test_invalid_credentials_cache_fails_download(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        ovms_kserve_serving_runtime: ServingRuntime,
        model_cache_infra_ready: DataScienceCluster,
        invalid_s3_download_secret: Secret,
        ci_s3_bucket_name: str,
    ) -> None:
        """Given a cache created with invalid S3 credentials,
        when the download is attempted,
        then no node reaches NodeDownloaded status.
        """
        cache_name = f"bad-creds-{shortuuid.uuid()[:10].lower()}"
        unique_path = f"nonexistent-model-{shortuuid.uuid()[:10].lower()}"
        source_uri = f"s3://{ci_s3_bucket_name}/{unique_path}/"

        with LocalModelNamespaceCache(
            client=admin_client,
            name=cache_name,
            namespace=unprivileged_model_namespace.name,
            source_model_uri=source_uri,
            model_size="100Mi",
            node_groups=[LOCAL_MODEL_NODE_GROUP_NAME],
            storage={"key": invalid_s3_download_secret.name},
        ) as cache:
            terminal_states = {"NodeDownloaded", "NodeDownloadError"}
            try:
                for status in TimeoutSampler(
                    wait_timeout=120,
                    sleep=10,
                    func=lambda: cache_status_dict(cache=cache),
                ):
                    node_status = status.get("nodeStatus") or {}
                    if node_status and all(s in terminal_states for s in node_status.values()):
                        break
            except TimeoutExpiredError:
                pass

            final_status = cache_status_dict(cache=cache)
            node_status = final_status.get("nodeStatus") or {}
            for node_name, state in node_status.items():
                assert state != "NodeDownloaded", (
                    f"Node '{node_name}' is NodeDownloaded with invalid credentials; expected failure state"
                )
