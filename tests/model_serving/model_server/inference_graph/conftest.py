from secrets import token_hex
from typing import Generator, Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_graph import InferenceGraph
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import ModelFormat, KServeDeploymentType, ModelStoragePath, Annotations, Labels
from utilities.inference_utils import create_isvc
from utilities.infra import create_inference_token, create_inference_graph_view_role


@pytest.fixture
def dog_breed_inference_graph(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    dog_cat_inference_service: InferenceService,
    dog_breed_inference_service: InferenceService,
) -> Generator[InferenceGraph, Any, Any]:
    nodes = {
        "root": {
            "routerType": "Sequence",
            "steps": [
                {"name": "dog-cat-classifier", "serviceName": dog_cat_inference_service.name},
                {
                    "name": "dog-breed-classifier",
                    "serviceName": dog_breed_inference_service.name,
                    "data": "$request",
                    "condition": "[@this].#(outputs.0.data.1>=0)",
                },
            ],
        }
    }

    annotations = {}
    labels = {}
    networking_label = Labels.Kserve.NETWORKING_KNATIVE_IO
    try:
        if request.param.get("deployment-mode"):
            annotations[Annotations.KserveIo.DEPLOYMENT_MODE] = request.param["deployment-mode"]
            if request.param["deployment-mode"] == KServeDeploymentType.RAW_DEPLOYMENT:
                networking_label = Labels.Kserve.NETWORKING_KSERVE_IO
    except AttributeError:
        pass

    try:
        if request.param.get("enable-auth"):
            annotations[Annotations.KserveAuth.SECURITY] = "true"
    except AttributeError:
        pass

    try:
        name = request.param["name"]
    except (AttributeError, KeyError):
        name = "dog-breed-pipeline"

    try:
        if not request.param["external-route"]:
            labels[networking_label] = "cluster-local"
    except (AttributeError, KeyError):
        pass

    with InferenceGraph(
        client=admin_client,
        name=name,
        namespace=unprivileged_model_namespace.name,
        nodes=nodes,
        annotations=annotations,
        label=labels,
    ) as inference_graph:
        inference_graph.wait_for_condition(condition=inference_graph.Condition.READY, status="True")
        yield inference_graph


@pytest.fixture(scope="class")
def dog_cat_inference_service(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
    models_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="dog-cat-classifier",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime.name,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.CAT_DOG_ONNX,
        model_format=ModelFormat.ONNX,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        protocol_version="v2",
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def dog_breed_inference_service(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
    models_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="dog-breed-classifier",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime.name,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.DOG_BREED_ONNX,
        model_format=ModelFormat.ONNX,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        protocol_version="v2",
    ) as isvc:
        yield isvc


@pytest.fixture
def inference_graph_unprivileged_sa_token(
    bare_service_account: ServiceAccount,
) -> str:
    return create_inference_token(model_service_account=bare_service_account)


@pytest.fixture
def inference_graph_sa_token_with_access(
    service_account_with_access: ServiceAccount,
) -> str:
    return create_inference_token(model_service_account=service_account_with_access)


@pytest.fixture
def service_account_with_access(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    dog_breed_inference_graph: InferenceGraph,
    bare_service_account: ServiceAccount,
) -> Generator[ServiceAccount, Any, Any]:
    with create_inference_graph_view_role(
        client=admin_client,
        name=f"{dog_breed_inference_graph.name}-view",
        namespace=unprivileged_model_namespace.name,
        resource_names=[dog_breed_inference_graph.name],
    ) as role:
        with RoleBinding(
            client=admin_client,
            namespace=unprivileged_model_namespace.name,
            name=f"{bare_service_account.name}-view",
            role_ref_name=role.name,
            role_ref_kind=role.kind,
            subjects_kind=bare_service_account.kind,
            subjects_name=bare_service_account.name,
        ):
            yield bare_service_account


@pytest.fixture
def bare_service_account(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[ServiceAccount, Any, Any]:
    try:
        if request.param["name"]:
            name = request.param["name"]
    except (AttributeError, KeyError):
        name = "sa-" + token_hex(4)

    with ServiceAccount(
        client=admin_client,
        namespace=unprivileged_model_namespace.name,
        name=name,
    ) as sa:
        yield sa
