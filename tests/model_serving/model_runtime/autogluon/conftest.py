"""
Pytest fixtures for AutoGluon KServe model serving runtime tests.

Provides ServingRuntime lifecycle, InferenceService creation, runtime image resolution,
and resource helpers for S3-backed AutoGluon models.
"""

import copy
from collections.abc import Generator

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_serving.model_runtime.autogluon.constant import (
    MODEL_CONFIGS,
    PREDICT_RESOURCES,
    build_serving_runtime_kwargs,
)
from tests.model_serving.model_runtime.autogluon.utils import (
    cleanup_autogluon_inference_service,
    get_runtime_image_override,
    resolve_autogluon_runtime_image,
)
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelInferenceRuntime,
    ModelVersion,
)
from utilities.inference_utils import create_isvc

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="session")
def autogluon_runtime_image(
    admin_client: DynamicClient,
    related_images_refs: set[str],
) -> str:
    """
    Resolve the AutoGluon server container image from the cluster or an override.

    Override via AUTOGLUON_RUNTIME_IMAGE environment variable when CSR/CSV lookup fails.
    """
    return resolve_autogluon_runtime_image(
        admin_client=admin_client,
        applications_namespace=py_config["applications_namespace"],
        related_images_refs=related_images_refs,
        override_image=get_runtime_image_override(),
    )


@pytest.fixture(scope="class")
def autogluon_serving_runtime(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    autogluon_runtime_image: str,
) -> Generator[ServingRuntime]:
    """
    Create a namespace-scoped AutoGluon ServingRuntime from in-repo spec.

    Args:
        request: Pytest fixture request with deployment_mode in param.
        admin_client: Kubernetes dynamic client.
        model_namespace: Target namespace.
        autogluon_runtime_image: Container image from cluster or override.

    Yields:
        ServingRuntime configured for AutoGluon models.
    """
    runtime_kwargs = build_serving_runtime_kwargs(
        namespace=model_namespace.name,
        image=autogluon_runtime_image,
        name=ModelInferenceRuntime.AUTOGLUON_RUNTIME,
    )
    with ServingRuntime(
        client=admin_client,
        teardown=True,
        **runtime_kwargs,
    ) as serving_runtime:
        yield serving_runtime


@pytest.fixture(scope="class")
def autogluon_model_service_account(
    admin_client: DynamicClient,
    kserve_s3_secret: Secret,
) -> Generator[ServiceAccount]:
    """
    ServiceAccount with access to the KServe S3 secret for AutoGluon models.

    Args:
        admin_client: Kubernetes dynamic client.
        kserve_s3_secret: S3 credentials secret.

    Yields:
        ServiceAccount linked to the S3 secret.
    """
    with ServiceAccount(
        client=admin_client,
        namespace=kserve_s3_secret.namespace,
        name="autogluon-models-bucket-sa",
        secrets=[{"name": kserve_s3_secret.name}],
    ) as service_account:
        yield service_account


@pytest.fixture(scope="class")
def autogluon_inference_service(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    autogluon_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    autogluon_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """
    InferenceService for an AutoGluon model loaded from S3.

    Args:
        request: Pytest param with name, deployment_mode, and predictor_type.
        admin_client: Kubernetes dynamic client.
        model_namespace: Deployment namespace.
        autogluon_serving_runtime: Namespace ServingRuntime.
        s3_models_storage_uri: S3 URI for model artifacts.
        autogluon_model_service_account: SA for storage credentials.

    Yields:
        Ready InferenceService instance.
    """
    params = request.param
    predictor_type = params["predictor_type"]
    if predictor_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown predictor type: {predictor_type}")

    model_config = MODEL_CONFIGS[predictor_type]
    resources = copy.deepcopy(PREDICT_RESOURCES["resources"])

    with create_isvc(
        client=admin_client,
        name=params["name"],
        namespace=model_namespace.name,
        runtime=autogluon_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=ModelFormat.AUTOGLUON,
        model_version=ModelVersion.AUTOGLUON_1,
        model_service_account=autogluon_model_service_account.name,
        deployment_mode=params.get("deployment_mode", KServeDeploymentType.STANDARD),
        external_route=params.get("enable_external_route", False),
        resources=resources,
        protocol_version=model_config["protocol_version"],
        min_replicas=params.get("min-replicas"),
        teardown=False,
    ) as isvc:
        try:
            yield isvc
        finally:
            cleanup_autogluon_inference_service(isvc=isvc)
