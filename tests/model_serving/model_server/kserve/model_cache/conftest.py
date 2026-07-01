"""Pytest fixtures for KServe ``LocalModelNamespaceCache`` tests."""

from collections.abc import Generator
from typing import Any

import pytest
import shortuuid
from kubernetes.dynamic import DynamicClient
from ocp_resources.daemonset import DaemonSet
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.node import Node
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_server.kserve.model_cache.utils import (
    LOCAL_MODEL_NODE_GROUP_NAME,
    MINT_ONNX_STORAGE_PATH,
    MODEL_CACHE_AGENT_DAEMONSET,
    MODEL_CACHE_NODE_COUNT,
    MODEL_CACHE_SIZE,
    LocalModelNamespaceCache,
    LocalModelNodeGroup,
    wait_for_local_model_cache_nodes_downloaded,
)
from utilities.constants import KServeDeploymentType, ModelFormat, Protocols, Timeout
from utilities.inference_utils import create_isvc
from utilities.infra import get_data_science_cluster, s3_endpoint_secret, wait_for_dsc_status_ready


@pytest.fixture(scope="session")
def model_cache_infra_ready(
    admin_client: DynamicClient,
) -> Generator[DataScienceCluster, Any, Any]:
    """Enable ``kserve.modelCache`` in the DSC and wait for the agent DaemonSet.

    Patches the DSC to set ``modelCache.managementState: Managed`` with a
    ``cacheSize`` and two worker ``nodeNames``.  On teardown the
    ``ResourceEditor`` restores the original DSC spec automatically.
    """
    dsc = get_data_science_cluster(client=admin_client)
    applications_namespace: str = py_config["applications_namespace"]

    already_labeled = sorted(
        [node.name for node in Node.get(client=admin_client, label_selector="kserve/localmodel=worker")],
    )
    all_workers = sorted(
        [node.name for node in Node.get(client=admin_client, label_selector="node-role.kubernetes.io/worker")],
    )

    if len(already_labeled) >= MODEL_CACHE_NODE_COUNT:
        selected_nodes = already_labeled[:MODEL_CACHE_NODE_COUNT]
    elif len(all_workers) >= MODEL_CACHE_NODE_COUNT:
        selected_nodes = all_workers[:MODEL_CACHE_NODE_COUNT]
    else:
        pytest.fail(f"Need at least {MODEL_CACHE_NODE_COUNT} worker nodes for model cache; found {len(all_workers)}")

    with ResourceEditor(
        patches={
            dsc: {
                "spec": {
                    "components": {
                        "kserve": {
                            "modelCache": {
                                "managementState": "Managed",
                                "cacheSize": MODEL_CACHE_SIZE,
                                "nodeNames": selected_nodes,
                            }
                        }
                    }
                }
            }
        }
    ):
        wait_for_dsc_status_ready(dsc_resource=dsc)

        try:
            for sample in TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_5MIN,
                sleep=10,
                func=lambda: LocalModelNodeGroup(client=admin_client, name=LOCAL_MODEL_NODE_GROUP_NAME).exists,
            ):
                if sample:
                    break
        except TimeoutExpiredError:
            pytest.fail(
                f"LocalModelNodeGroup '{LOCAL_MODEL_NODE_GROUP_NAME}' did not appear "
                f"within {Timeout.TIMEOUT_5MIN}s after enabling modelCache in DSC"
            )

        agent = DaemonSet(
            client=admin_client,
            name=MODEL_CACHE_AGENT_DAEMONSET,
            namespace=applications_namespace,
        )
        try:
            for sample in TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_5MIN,
                sleep=10,
                func=lambda: agent.exists,
            ):
                if sample:
                    break
        except TimeoutExpiredError:
            pytest.fail(
                f"DaemonSet '{MODEL_CACHE_AGENT_DAEMONSET}' did not appear in "
                f"'{applications_namespace}' within {Timeout.TIMEOUT_5MIN}s"
            )

        yield dsc


@pytest.fixture(scope="class")
def model_cache_download_s3_secret(
    admin_client: DynamicClient,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    """S3 credential secret in the job namespace for ``LocalModelNamespaceCache`` download Jobs.

    The download Job runs in the operator's job namespace (``redhat-ods-applications``),
    which is separate from the ISVC namespace.  The ``LocalModelNamespaceCache`` spec references
    this secret via ``spec.storage.key``.
    """
    applications_namespace: str = py_config["applications_namespace"]
    with s3_endpoint_secret(
        client=admin_client,
        name="model-cache-download-secret",
        namespace=applications_namespace,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def invalid_s3_download_secret(
    admin_client: DynamicClient,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    """S3 secret with invalid credentials for negative download testing."""
    applications_namespace: str = py_config["applications_namespace"]
    with s3_endpoint_secret(
        client=admin_client,
        name="model-cache-invalid-secret",
        namespace=applications_namespace,
        aws_access_key="INVALIDACCESSKEY12345",
        aws_secret_access_key="INVALIDSECRETACCESSKEY6789",  # pragma: allowlist secret
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def mnist_local_model_cache(
    admin_client: DynamicClient,
    model_cache_infra_ready: DataScienceCluster,
    model_cache_download_s3_secret: Secret,
    unprivileged_model_namespace: Namespace,
    ci_s3_bucket_name: str,
) -> Generator[LocalModelNamespaceCache, Any, Any]:
    """Create a ``LocalModelNamespaceCache`` for the MNIST ONNX model and wait for ``NodeDownloaded``."""
    cache_name = f"mnist-onnx-{shortuuid.uuid()[:10].lower()}"
    source_uri = f"s3://{ci_s3_bucket_name}/{MINT_ONNX_STORAGE_PATH}/"
    with LocalModelNamespaceCache(
        client=admin_client,
        name=cache_name,
        namespace=unprivileged_model_namespace.name,
        source_model_uri=source_uri,
        model_size="100Mi",
        node_groups=[LOCAL_MODEL_NODE_GROUP_NAME],
        storage={"key": model_cache_download_s3_secret.name},
    ) as cache:
        wait_for_local_model_cache_nodes_downloaded(cache=cache, timeout=Timeout.TIMEOUT_10MIN)
        yield cache


@pytest.fixture(scope="class")
def mnist_onnx_local_model_cache_inference_service(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    mnist_local_model_cache: LocalModelNamespaceCache,
) -> Generator[InferenceService, Any, Any]:
    """Deploy a raw ``InferenceService`` whose storageUri matches the cached model.

    The KServe defaulting webhook automatically detects a matching
    ``LocalModelNamespaceCache.spec.sourceModelUri`` and rewrites the ISVC to use
    PVC-backed storage — no manual ``localmodel`` label is needed.
    """
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}-lmcache",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime.name,
        storage_uri=f"s3://{ci_s3_bucket_name}/{MINT_ONNX_STORAGE_PATH}/",
        model_format=ovms_kserve_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=True,
        timeout=Timeout.TIMEOUT_15MIN,
    ) as isvc:
        yield isvc
