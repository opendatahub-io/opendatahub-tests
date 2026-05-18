"""Pytest fixtures for KServe `LocalModelCache` smoke tests."""

from collections.abc import Generator
from typing import Any

import pytest
import shortuuid
from kubernetes.dynamic import DynamicClient
from ocp_resources.daemonset import DaemonSet
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.kserve.model_cache.utils import (
    KSERVE_LOCALMODEL_LABEL,
    LOCAL_MODEL_NODE_GROUP_NAME,
    MINT_ONNX_STORAGE_PATH,
    MODEL_CACHE_AGENT_DAEMONSET,
    LocalModelCache,
    LocalModelNodeGroup,
    wait_for_local_model_cache_nodes_downloaded,
)
from utilities.constants import KServeDeploymentType, ModelFormat, Protocols, Timeout
from utilities.inference_utils import create_isvc
from utilities.infra import s3_endpoint_secret


@pytest.fixture(scope="session")
def model_cache_infra_ready(admin_client: DynamicClient) -> None:
    """Skip the session when the operator has not provisioned model-cache infra from the DSC."""
    node_group = LocalModelNodeGroup(client=admin_client, name=LOCAL_MODEL_NODE_GROUP_NAME)
    if not node_group.exists:
        pytest.skip(
            f"LocalModelNodeGroup '{LOCAL_MODEL_NODE_GROUP_NAME}' not found; "
            "set kserve.modelCache.managementState=Managed in the DSC."
        )

    applications_namespace: str = py_config["applications_namespace"]
    agent = DaemonSet(
        client=admin_client,
        name=MODEL_CACHE_AGENT_DAEMONSET,
        namespace=applications_namespace,
    )
    if not agent.exists:
        pytest.skip(
            f"DaemonSet '{MODEL_CACHE_AGENT_DAEMONSET}' not found in '{applications_namespace}'; "
            "model cache agent not deployed."
        )


@pytest.fixture(scope="class")
def model_cache_download_s3_secret(
    admin_client: DynamicClient,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    """S3 credential secret in the job namespace for `LocalModelCache` download Jobs.

    The download Job runs in the operator's job namespace (``redhat-ods-applications``),
    which is separate from the ISVC namespace.  The ``LocalModelCache`` spec references
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
def mnist_local_model_cache(
    admin_client: DynamicClient,
    model_cache_infra_ready: None,
    model_cache_download_s3_secret: Secret,
    ci_s3_bucket_name: str,
) -> Generator[LocalModelCache, Any, Any]:
    """Create a `LocalModelCache` for the MNIST ONNX model and wait for `NodeDownloaded`."""
    cache_name = f"mnist-onnx-{shortuuid.uuid()[:10].lower()}"
    source_uri = f"s3://{ci_s3_bucket_name}/{MINT_ONNX_STORAGE_PATH}/"
    with LocalModelCache(
        client=admin_client,
        name=cache_name,
        source_model_uri=source_uri,
        model_size="100Mi",
        node_groups=[LOCAL_MODEL_NODE_GROUP_NAME],
        storage_key=model_cache_download_s3_secret.name,
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
    """Deploy a raw `InferenceService` that uses the MNIST cache via the localmodel label."""
    labels = {KSERVE_LOCALMODEL_LABEL: mnist_local_model_cache.name}
    with create_isvc(
        client=unprivileged_client,
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}-lmcache",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime.name,
        storage_key=ci_endpoint_s3_secret.name,
        storage_path=MINT_ONNX_STORAGE_PATH,
        model_format=ovms_kserve_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=True,
        labels=labels,
        timeout=Timeout.TIMEOUT_15MIN,
    ) as isvc:
        yield isvc
