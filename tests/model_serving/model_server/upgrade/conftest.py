import os
from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.utils import create_isvc
from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelFormat,
    ModelStoragePath,
    ModelVersion,
    RuntimeTemplates,
)
from utilities.infra import create_ns
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="session")
def skipped_teardown_resources() -> None:
    os.environ["SKIP_RESOURCE_TEARDOWN"] = (
        "{Namespace: {upgrade-model-server}, ServingRuntime: {}, InferenceService: {},Secret: {},ServiceAccount:  {}}"
    )


@pytest.fixture(scope="session")
def model_namespace_scope_session(
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    with create_ns(
        admin_client=admin_client,
        name="upgrade-model-server",
        labels={"modelmesh-enabled": "true"},
    ) as ns:
        yield ns


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
            "ovms": {
                "requests": {"cpu": "1", "memory": "4Gi"},
                "limits": {"cpu": "2", "memory": "8Gi"},
            }
        },
        model_format_name={ModelFormat.ONNX: ModelVersion.OPSET13},
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="session")
def ovms_serverless_inference_service_scope_session(
    admin_client: DynamicClient,
    openvino_serverless_serving_runtime_scope_session: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="serverless-ovms",
        namespace=openvino_serverless_serving_runtime_scope_session.namespace,
        runtime=openvino_serverless_serving_runtime_scope_session.name,
        storage_path="test-dir",
        storage_key=ci_endpoint_s3_secret.name,
        model_format=ModelAndFormat.OPENVINO_IR,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        model_version=ModelVersion.OPSET13,
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
    admin_client: DynamicClient,
    caikit_raw_serving_runtime_scope_session: ServingRuntime,
    models_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="caikit-raw",
        namespace=caikit_raw_serving_runtime_scope_session.namespace,
        model_format=caikit_raw_serving_runtime_scope_session.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.EMBEDDING_MODEL,
        external_route=True,
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
        protocol="REST",
        resources={
            "ovms": {
                "requests": {"cpu": "1", "memory": "4Gi"},
                "limits": {"cpu": "2", "memory": "8Gi"},
            }
        },
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="session")
def openvino_model_mesh_inference_service_scope_session(
    admin_client: DynamicClient,
    s3_ovms_model_mesh_serving_runtime_scope_session: ServingRuntime,
    ci_model_mesh_endpoint_s3_secret: Secret,
    model_mesh_model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="openvino-model-mesh",
        namespace=s3_ovms_model_mesh_serving_runtime_scope_session.namespace,
        runtime=s3_ovms_model_mesh_serving_runtime_scope_session.name,
        model_service_account=model_mesh_model_service_account.name,
        storage_key=ci_model_mesh_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.OPENVINO_EXAMPLE_MODEL,
        model_format=ModelAndFormat.OPENVINO_IR,
        deployment_mode=KServeDeploymentType.MODEL_MESH,
        model_version=ModelVersion.OPSET1,
    ) as isvc:
        yield isvc
