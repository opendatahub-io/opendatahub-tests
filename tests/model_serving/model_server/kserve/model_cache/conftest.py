"""Pytest fixtures for KServe `LocalModelCache` smoke tests."""

from collections.abc import Generator
from typing import Any

import pytest
import shortuuid
from kubernetes.dynamic import DynamicClient
from ocp_resources.daemon_set import DaemonSet
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.kserve.model_cache.utils import (
    KSERVE_LOCALMODEL_LABEL,
    LOCAL_MODEL_NODE_GROUP_NAME,
    MNIST_ONNX_S3_PATH,
    MODEL_CACHE_AGENT_DAEMONSET,
    MODEL_CACHE_DOWNLOAD_SA,
    MODEL_CACHE_JOBS_NAMESPACE,
    LocalModelCache,
    LocalModelNodeGroup,
    wait_for_local_model_cache_nodes_downloaded,
)
from utilities.constants import KServeDeploymentType, ModelFormat, Protocols, Timeout
from utilities.inference_utils import create_isvc


@pytest.fixture(scope="session")
def model_cache_infra_ready(admin_client: DynamicClient) -> None:
    """Skip when DSC model-cache reconciliation has not created required cluster objects."""
    node_group = LocalModelNodeGroup(client=admin_client, name=LOCAL_MODEL_NODE_GROUP_NAME)
    if not node_group.exists:
        pytest.skip(
            "LocalModelNodeGroup 'workers' not found; enable DSC "
            "components.kserve.modelCache with managementState Managed."
        )

    applications_namespace: str = py_config["applications_namespace"]
    agent = DaemonSet(
        client=admin_client,
        name=MODEL_CACHE_AGENT_DAEMONSET,
        namespace=applications_namespace,
    )
    if not agent.exists:
        pytest.skip(
            f"DaemonSet {MODEL_CACHE_AGENT_DAEMONSET} not found in {applications_namespace}; "
            "model cache agent not deployed."
        )

    download_sa = ServiceAccount(
        client=admin_client,
        name=MODEL_CACHE_DOWNLOAD_SA,
        namespace=MODEL_CACHE_JOBS_NAMESPACE,
    )
    if not download_sa.exists:
        pytest.skip(
            f"ServiceAccount {MODEL_CACHE_DOWNLOAD_SA} not found in {MODEL_CACHE_JOBS_NAMESPACE}; "
            "cannot run model download Jobs for LocalModelCache."
        )


@pytest.fixture(scope="class")
def mnist_local_model_cache(
    admin_client: DynamicClient,
    model_cache_infra_ready: None,
    ci_s3_bucket_name: str,
) -> Generator[LocalModelCache, Any, Any]:
    """Create a `LocalModelCache` for the CI MNIST ONNX path and remove it after tests."""
    cache_name = f"mnist-onnx-{shortuuid.uuid()[:10].lower()}"
    source_uri = f"s3://{ci_s3_bucket_name}/{MNIST_ONNX_S3_PATH}/"
    with LocalModelCache(
        client=admin_client,
        name=cache_name,
        source_model_uri=source_uri,
        model_size="100Mi",
        node_groups=[LOCAL_MODEL_NODE_GROUP_NAME],
        service_account_name=MODEL_CACHE_DOWNLOAD_SA,
    ) as cache:
        wait_for_local_model_cache_nodes_downloaded(cache=cache, timeout=Timeout.TIMEOUT_10MIN)
        yield cache


@pytest.fixture(scope="class")
def mnist_onnx_local_model_cache_inference_service(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
    mnist_local_model_cache: LocalModelCache,
) -> Generator[InferenceService, Any, Any]:
    """Deploy a raw `InferenceService` that references the MNIST cache via the localmodel label."""
    labels = {KSERVE_LOCALMODEL_LABEL: mnist_local_model_cache.name}
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}-lmcache",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime.name,
        storage_key=ci_endpoint_s3_secret.name,
        storage_path=MNIST_ONNX_S3_PATH,
        model_format=ovms_kserve_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=True,
        labels=labels,
        timeout=Timeout.TIMEOUT_15MIN,
    ) as isvc:
        yield isvc
