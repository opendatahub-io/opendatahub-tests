from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from simple_logger.logger import get_logger
from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelFormat,
    ModelStoragePath,
    ModelVersion,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns, s3_endpoint_secret
from utilities.serving_runtime import ServingRuntimeFromTemplate


LOGGER = get_logger(name=__name__)

UPGRADE_NAMESPACE = "upgrade-model-server"
S3_CONNECTION = "upgrade-connection"


@pytest.fixture(scope="session")
def namespace_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    ns = Namespace(client=admin_client, name=UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        yield ns
        ns.clean_up()

    else:
        with create_ns(
            admin_client=admin_client,
            name=UPGRADE_NAMESPACE,
            model_mesh_enabled=False,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def s3_connection_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    namespace_fixture: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    secret_kwargs = {
        "client": admin_client,
        "name": S3_CONNECTION,
        "namespace": namespace_fixture.name,
    }

    secret = Secret(**secret_kwargs)

    if pytestconfig.option.post_upgrade:
        yield secret
        secret.clean_up()

    else:
        with s3_endpoint_secret(
            **secret_kwargs,
            aws_access_key=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_region=models_s3_bucket_region,
            aws_s3_bucket=models_s3_bucket_name,
            aws_s3_endpoint=models_s3_bucket_endpoint,
            teardown=teardown_resources,
        ) as secret:
            yield secret


@pytest.fixture(scope="session")
def serving_runtime_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    runtime_kwargs = {
        "client": admin_client,
        "name": "upgrade-runtime",
        "namespace": namespace_fixture.name,
    }

    model_runtime = ServingRuntime(**runtime_kwargs)

    if pytestconfig.option.post_upgrade:
        yield model_runtime
        model_runtime.clean_up()

    else:
        with ServingRuntimeFromTemplate(
            **runtime_kwargs,
            template_name=RuntimeTemplates.OVMS_KSERVE,
            multi_model=False,
            enable_http=True,
            teardown=teardown_resources,
            resources={
                ModelFormat.OVMS: {
                    "requests": {"cpu": "1", "memory": "4Gi"},
                    "limits": {"cpu": "2", "memory": "8Gi"},
                }
            },
        ) as model_runtime:
            yield model_runtime


@pytest.fixture(scope="session")
def inference_service_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    serving_runtime_fixture: ServingRuntime,
    s3_connection_fixture: Secret,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": "upgrade-isvc",
        "namespace": serving_runtime_fixture.namespace,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc

        isvc.clean_up()

    else:
        with create_isvc(
            runtime=serving_runtime_fixture.name,
            model_format=ModelAndFormat.OPENVINO_IR,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            storage_key=s3_connection_fixture.name,
            storage_path=ModelStoragePath.OPENVINO_EXAMPLE_MODEL,
            model_version=ModelVersion.OPSET13,
            external_route=False,
            teardown=teardown_resources,
            **isvc_kwargs,
        ) as isvc:
            yield isvc
