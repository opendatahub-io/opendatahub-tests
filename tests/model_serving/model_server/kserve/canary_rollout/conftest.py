"""Fixtures for KServe canary rollout (RawDeployment) tests."""

from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.kserve.canary_rollout.constants import (
    CANARY_FEATURE_NAME,
    CANARY_MODEL_DIR,
    CANARY_MODEL_FORMAT,
    CANARY_NAMESPACE_PREFIX,
    DEFAULT_CANARY_TRAFFIC_PERCENT,
    DEFAULT_DEPLOYMENT_MODE,
    STABLE_MODEL_DIR,
    STABLE_MODEL_FORMAT,
)
from tests.model_serving.model_server.kserve.canary_rollout.utils import create_canary_inference_service
from utilities.constants import RuntimeTemplates
from utilities.infra import create_ns, s3_endpoint_secret
from utilities.serving_runtime import ServingRuntimeFromTemplate

pytestmark = [pytest.mark.rawdeployment, pytest.mark.tier1, pytest.mark.usefixtures("valid_aws_config")]


@pytest.fixture(scope="package")
def canary_rollout_namespace(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Shared namespace for canary rollout tests."""
    with create_ns(
        admin_client=admin_client,
        unprivileged_client=unprivileged_client,
        name=f"{CANARY_NAMESPACE_PREFIX}-ns",
    ) as namespace:
        yield namespace


@pytest.fixture(scope="package")
def canary_rollout_s3_secret(
    unprivileged_client: DynamicClient,
    canary_rollout_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    """S3 credentials secret for canary rollout model storage."""
    with s3_endpoint_secret(
        client=unprivileged_client,
        name="canary-models-bucket-secret",
        namespace=canary_rollout_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="package")
def canary_rollout_service_account(
    admin_client: DynamicClient,
    canary_rollout_namespace: Namespace,
    canary_rollout_s3_secret: Secret,
) -> Generator[ServiceAccount, Any, Any]:
    """Service account linked to the canary rollout S3 secret."""
    with ServiceAccount(
        client=admin_client,
        namespace=canary_rollout_namespace.name,
        name="canary-models-bucket-sa",
        secrets=[{"name": canary_rollout_s3_secret.name}],
    ) as service_account:
        yield service_account


@pytest.fixture(scope="package")
def canary_mlserver_runtime(
    admin_client: DynamicClient,
    canary_rollout_namespace: Namespace,
    mlserver_runtime_image: str,
) -> Generator[ServingRuntime, Any, Any]:
    """MLServer ServingRuntime for RawDeployment canary tests."""
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=f"{CANARY_FEATURE_NAME}-runtime",
        namespace=canary_rollout_namespace.name,
        template_name=RuntimeTemplates.MLSERVER,
        deployment_type=DEFAULT_DEPLOYMENT_MODE,
        runtime_image=mlserver_runtime_image,
    ) as model_runtime:
        yield model_runtime


def _model_storage_uri(bucket_name: str, model_dir: str) -> str:
    return f"s3://{bucket_name}/{model_dir.rstrip('/')}/"


@pytest.fixture(scope="package")
def canary_sklearn_inference_service(
    admin_client: DynamicClient,
    canary_rollout_namespace: Namespace,
    canary_mlserver_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    canary_rollout_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    """InferenceService with a canary array entry at 10% traffic."""
    with create_canary_inference_service(
        client=admin_client,
        name=f"{CANARY_NAMESPACE_PREFIX}-10",
        namespace=canary_rollout_namespace.name,
        runtime=canary_mlserver_runtime.name,
        stable_model_format=STABLE_MODEL_FORMAT,
        stable_storage_uri=_model_storage_uri(ci_s3_bucket_name, STABLE_MODEL_DIR),
        canary_model_format=CANARY_MODEL_FORMAT,
        canary_storage_uri=_model_storage_uri(ci_s3_bucket_name, CANARY_MODEL_DIR),
        canary_traffic_percent=DEFAULT_CANARY_TRAFFIC_PERCENT,
        deployment_mode=DEFAULT_DEPLOYMENT_MODE,
        external_route=True,
        model_service_account=canary_rollout_service_account.name,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="package")
def canary_ctrl_inference_service(
    admin_client: DynamicClient,
    canary_rollout_namespace: Namespace,
    canary_mlserver_runtime: ServingRuntime,
    ci_s3_bucket_name: str,
    canary_rollout_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    """InferenceService with 20% canary traffic for controller behavior tests."""
    with create_canary_inference_service(
        client=admin_client,
        name=f"{CANARY_NAMESPACE_PREFIX}-ctrl",
        namespace=canary_rollout_namespace.name,
        runtime=canary_mlserver_runtime.name,
        stable_model_format=STABLE_MODEL_FORMAT,
        stable_storage_uri=_model_storage_uri(ci_s3_bucket_name, STABLE_MODEL_DIR),
        canary_model_format=CANARY_MODEL_FORMAT,
        canary_storage_uri=_model_storage_uri(ci_s3_bucket_name, CANARY_MODEL_DIR),
        canary_traffic_percent=20,
        deployment_mode=DEFAULT_DEPLOYMENT_MODE,
        external_route=True,
        model_service_account=canary_rollout_service_account.name,
    ) as isvc:
        yield isvc
