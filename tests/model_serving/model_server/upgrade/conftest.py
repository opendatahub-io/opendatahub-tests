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
    ModelStoragePath,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns, s3_endpoint_secret
from utilities.serving_runtime import ServingRuntimeFromTemplate


LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session")
def model_namespace_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    name = "upgrade-model-server"
    ns = Namespace(client=admin_client, name=name)

    if pytestconfig.option.post_upgrade:
        yield ns
        ns.clean_up()

    else:
        with create_ns(
            admin_client=admin_client,
            name=name,
            model_mesh_enabled=False,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def models_endpoint_s3_secret_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace_scope_session: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    secret_kwargs = {
        "client": admin_client,
        "name": "models-bucket-secret",
        "namespace": model_namespace_scope_session.name,
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
def caikit_raw_serving_runtime_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace_scope_session: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    runtime_kwargs = {
        "client": admin_client,
        "name": "caikit-raw",
        "namespace": model_namespace_scope_session.name,
    }

    model_runtime = ServingRuntime(**runtime_kwargs)

    if pytestconfig.option.post_upgrade:
        yield model_runtime
        model_runtime.clean_up()

    else:
        with ServingRuntimeFromTemplate(
            **runtime_kwargs,
            template_name=RuntimeTemplates.CAIKIT_STANDALONE_SERVING,
            multi_model=False,
            enable_http=True,
            teardown=teardown_resources,
        ) as model_runtime:
            yield model_runtime


@pytest.fixture(scope="session")
def caikit_raw_inference_service_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    caikit_raw_serving_runtime_scope_session: ServingRuntime,
    models_endpoint_s3_secret_scope_session: Secret,
    teardown_resources: bool,
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
            external_route=False,
            teardown=teardown_resources,
            **isvc_kwargs,
        ) as isvc:
            yield isvc
