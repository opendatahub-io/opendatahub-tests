from collections.abc import Generator
from typing import Any
from urllib.parse import urlparse

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelName,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.infra import s3_endpoint_secret
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def dummy_s3_secret(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    with s3_endpoint_secret(
        client=unprivileged_client,
        name="dummy-ci-bucket-secret",
        namespace=unprivileged_model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def dummy_ovms_serving_runtime(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=unprivileged_client,
        name=f"dummy-{ModelName.MNIST}-runtime",
        namespace=unprivileged_model_namespace.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as runtime:
        yield runtime


@pytest.fixture(scope="class")
def dummy_ovms_raw_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    dummy_ovms_serving_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    dummy_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    storage_uri = f"s3://{ci_s3_bucket_name}/{request.param['model-dir']}/"
    with create_isvc(
        client=unprivileged_client,
        name=f"dummy-{Protocols.HTTP}-{ModelFormat.ONNX}",
        namespace=unprivileged_model_namespace.name,
        runtime=dummy_ovms_serving_runtime.name,
        storage_key=dummy_s3_secret.name,
        storage_path=urlparse(storage_uri).path,
        model_format=dummy_ovms_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        external_route=True,
    ) as isvc:
        yield isvc
