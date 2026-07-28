"""Pytest fixtures for KServe ``LocalModelNamespaceCache`` tests.

``model_cache_infra_ready`` (the DSC ``modelCache`` toggle shared with the
``LLMInferenceService`` local model cache tests in ``llmd/``) lives in the
parent ``tests/model_serving/model_server/conftest.py`` so it is visible to
both sibling test packages.
"""

from collections.abc import Generator
from typing import Any

import pytest
import shortuuid
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.kserve.model_cache.utils import (
    LOCAL_MODEL_NODE_GROUP_NAME,
    MINT_ONNX_STORAGE_PATH,
    LocalModelNamespaceCache,
    wait_for_local_model_cache_nodes_downloaded,
)
from utilities.constants import KServeDeploymentType, ModelFormat, Protocols
from utilities.inference_utils import create_isvc
from utilities.infra import s3_endpoint_secret


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
        name=f"model-cache-download-secret-{shortuuid.uuid()[:10].lower()}",
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
        name=f"model-cache-invalid-secret-{shortuuid.uuid()[:10].lower()}",
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
        wait_for_local_model_cache_nodes_downloaded(cache=cache, timeout=600)
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
        name=f"{Protocols.HTTP}-{ModelFormat.ONNX}-lmcache-{shortuuid.uuid()[:8].lower()}",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime.name,
        storage_uri=f"s3://{ci_s3_bucket_name}/{MINT_ONNX_STORAGE_PATH}/",
        model_format=ovms_kserve_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=True,
        timeout=900,
    ) as isvc:
        yield isvc
