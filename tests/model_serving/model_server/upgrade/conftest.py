import os
from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from simple_logger.logger import get_logger

from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelFormat,
    ModelStoragePath,
    ModelVersion,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns, s3_endpoint_secret
from utilities.serving_runtime import ServingRuntimeFromTemplate


LOGGER = get_logger(name=__name__)
UPGRADE_NAMESPACE: str = "upgrade-model-server"

UPGRADE_RESOURCES: str = (
    f"{{Namespace: {{{UPGRADE_NAMESPACE:}}},"
    f"ServingRuntime: {{onnx-serverless: {UPGRADE_NAMESPACE},"
    f"caikit-raw: {UPGRADE_NAMESPACE},ovms-model-mesh: {UPGRADE_NAMESPACE}}},"
    f"InferenceService: {{onnx-serverless: {UPGRADE_NAMESPACE},"
    f"caikit-raw: {UPGRADE_NAMESPACE}, ovms-model-mesh: {UPGRADE_NAMESPACE}}},"
    f"Secret: {{ci-bucket-secret: {UPGRADE_NAMESPACE}, models-bucket-secret: {UPGRADE_NAMESPACE}}},"
    f"ServiceAccount: {{models-bucket-sa: {UPGRADE_NAMESPACE}}}}}"
)


@pytest.fixture(scope="session")
def skipped_teardown_resources(pytestconfig: pytest.Config) -> None:
    if not pytestconfig.option.delete_pre_upgrade_resources:
        LOGGER.info(f"Setting `SKIP_RESOURCE_TEARDOWN` environment variable to {UPGRADE_RESOURCES}")
        os.environ["SKIP_RESOURCE_TEARDOWN"] = UPGRADE_RESOURCES


@pytest.fixture(scope="session")
def reused_resources() -> None:
    LOGGER.info(f"Setting `REUSE_IF_RESOURCE_EXISTS` environment variable to {UPGRADE_RESOURCES}")
    os.environ["REUSE_IF_RESOURCE_EXISTS"] = UPGRADE_RESOURCES


@pytest.fixture(scope="session")
def model_namespace_scope_session(
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    with create_ns(
        admin_client=admin_client,
        name=UPGRADE_NAMESPACE,
        labels={"modelmesh-enabled": "true"},
    ) as ns:
        yield ns


@pytest.fixture(scope="session")
def models_endpoint_s3_secret_scope_session(
    admin_client: DynamicClient,
    model_namespace_scope_session: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    with s3_endpoint_secret(
        admin_client=admin_client,
        name="models-bucket-secret",
        namespace=model_namespace_scope_session.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_bucket=models_s3_bucket_name,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="session")
def ci_endpoint_s3_secret_scope_session(
    admin_client: DynamicClient,
    model_namespace_scope_session: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Generator[Secret, Any, Any]:
    with s3_endpoint_secret(
        admin_client=admin_client,
        name="ci-bucket-secret",
        namespace=model_namespace_scope_session.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="session")
def model_mesh_model_service_account_scope_session(
    admin_client: DynamicClient, ci_endpoint_s3_secret_scope_session: Secret
) -> Generator[ServiceAccount, Any, Any]:
    with ServiceAccount(
        client=admin_client,
        namespace=ci_endpoint_s3_secret_scope_session.namespace,
        name="models-bucket-sa",
        secrets=[{"name": ci_endpoint_s3_secret_scope_session.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="session")
def openvino_serverless_serving_runtime_scope_session(
    admin_client: DynamicClient,
    model_namespace_scope_session: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="onnx-serverless",
        namespace=model_namespace_scope_session.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
        multi_model=False,
        resources={
            ModelFormat.OVMS: {
                "requests": {"cpu": "1", "memory": "4Gi"},
                "limits": {"cpu": "2", "memory": "8Gi"},
            }
        },
        model_format_name={ModelFormat.ONNX: ModelVersion.OPSET13},
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="session")
def ovms_serverless_inference_service_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    openvino_serverless_serving_runtime_scope_session: ServingRuntime,
    ci_endpoint_s3_secret_scope_session: Secret,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": openvino_serverless_serving_runtime_scope_session.name,
        "namespace": openvino_serverless_serving_runtime_scope_session.namespace,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc
        isvc.clean_up()

    else:
        with create_isvc(
            runtime=openvino_serverless_serving_runtime_scope_session.name,
            storage_path="test-dir",
            storage_key=ci_endpoint_s3_secret_scope_session.name,
            model_format=ModelAndFormat.OPENVINO_IR,
            deployment_mode=KServeDeploymentType.SERVERLESS,
            model_version=ModelVersion.OPSET13,
            **isvc_kwargs,
        ) as isvc:
            yield isvc


@pytest.fixture(scope="session")
def caikit_raw_serving_runtime_scope_session(
    admin_client: DynamicClient,
    model_namespace_scope_session: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="caikit-raw",
        namespace=model_namespace_scope_session.name,
        template_name=RuntimeTemplates.CAIKIT_STANDALONE_SERVING,
        multi_model=False,
        enable_http=True,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="session")
def caikit_raw_inference_service_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    caikit_raw_serving_runtime_scope_session: ServingRuntime,
    models_endpoint_s3_secret_scope_session: Secret,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": caikit_raw_serving_runtime_scope_session.name,
        "namespace": caikit_raw_serving_runtime_scope_session.namespace,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc

        isvc.clean_up()

    else:
        with create_isvc(
            runtime=caikit_raw_serving_runtime_scope_session.name,
            model_format=caikit_raw_serving_runtime_scope_session.instance.spec.supportedModelFormats[0].name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            storage_key=models_endpoint_s3_secret_scope_session.name,
            storage_path=ModelStoragePath.EMBEDDING_MODEL,
            external_route=True,
            **isvc_kwargs,
        ) as isvc:
            yield isvc


@pytest.fixture(scope="session")
def s3_ovms_model_mesh_serving_runtime_scope_session(
    admin_client: DynamicClient,
    model_namespace_scope_session: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="ovms-model-mesh",
        namespace=model_namespace_scope_session.name,
        template_name=RuntimeTemplates.OVMS_MODEL_MESH,
        multi_model=True,
        protocol=Protocols.REST.upper(),
        resources={
            ModelFormat.OVMS: {
                "requests": {"cpu": "1", "memory": "4Gi"},
                "limits": {"cpu": "2", "memory": "8Gi"},
            }
        },
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="session")
def openvino_model_mesh_inference_service_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    s3_ovms_model_mesh_serving_runtime_scope_session: ServingRuntime,
    ci_endpoint_s3_secret_scope_session: Secret,
    model_mesh_model_service_account_scope_session: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": s3_ovms_model_mesh_serving_runtime_scope_session.name,
        "namespace": s3_ovms_model_mesh_serving_runtime_scope_session.namespace,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc
        isvc.clean_up()

    else:
        with create_isvc(
            runtime=s3_ovms_model_mesh_serving_runtime_scope_session.name,
            model_service_account=model_mesh_model_service_account_scope_session.name,
            storage_key=ci_endpoint_s3_secret_scope_session.name,
            storage_path=ModelStoragePath.OPENVINO_EXAMPLE_MODEL,
            model_format=ModelAndFormat.OPENVINO_IR,
            deployment_mode=KServeDeploymentType.MODEL_MESH,
            model_version=ModelVersion.OPSET1,
            **isvc_kwargs,
        ) as isvc:
            yield isvc
