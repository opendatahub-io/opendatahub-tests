import logging
import time
from collections.abc import Generator
from secrets import token_hex
from typing import Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.inference_graph import InferenceGraph
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from utilities.constants import Annotations, KServeDeploymentType, Labels, ModelFormat, ModelStoragePath
from utilities.inference_utils import create_isvc
from utilities.infra import (
    create_inference_graph_view_role,
    create_inference_token,
    get_services_by_isvc_label,
)


@pytest.fixture(scope="class")
def kserve_raw_headless_service_config(
    dsc_resource: DataScienceCluster,
    admin_client: DynamicClient,
) -> Generator[DataScienceCluster, Any, Any]:
    logger = logging.getLogger(__name__)

    current_config = None
    if hasattr(dsc_resource.instance.spec.components.kserve, "rawDeploymentServiceConfig"):
        current_config = dsc_resource.instance.spec.components.kserve.rawDeploymentServiceConfig

    logger.info(msg=f"Current rawDeploymentServiceConfig: {current_config}")

    # If already Headed, keep the existing mode. For tests we want Headed even if the
    # cluster is explicitly configured as Headless.
    if current_config and current_config.lower() == "headed":
        logger.info(
            msg=(f"rawDeploymentServiceConfig is already set to '{current_config}', reusing existing configuration")
        )
        yield dsc_resource
        return

    logger.info(msg=f"Patching rawDeploymentServiceConfig from '{current_config}' to 'Headed'")
    # Patch DSC to set rawDeploymentServiceConfig to Headed when not explicitly configured
    with ResourceEditor(
        patches={dsc_resource: {"spec": {"components": {"kserve": {"rawDeploymentServiceConfig": "Headed"}}}}}
    ):
        logger.info(msg="Waiting for DSC to become ready after patch...")
        dsc_resource.wait_for_condition(
            condition=dsc_resource.Condition.READY,
            status=dsc_resource.Condition.Status.TRUE,
            timeout=300,
        )
        # Verify the patch was applied
        new_config = dsc_resource.instance.spec.components.kserve.rawDeploymentServiceConfig
        logger.info(msg=f"After patch, rawDeploymentServiceConfig is: {new_config}")

        logger.info(msg="Waiting for KServe controller to be ready and configuration to propagate...")
        kserve_deployments = list(
            Deployment.get(
                client=admin_client,
                namespace=py_config.get("applications_namespace", "redhat-ods-applications"),
                label_selector="control-plane=kserve-controller-manager",
            )
        )

        if kserve_deployments:
            for deployment in kserve_deployments:
                deployment.wait_for_replicas(timeout=180)
        else:
            logger.warning(msg="No KServe controller deployment found")
        logger.info(msg="Waiting for KServe controller to process configuration change...")
        time.sleep(seconds=60)

        yield dsc_resource


@pytest.fixture
def dog_breed_inference_graph(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    dog_cat_inference_service: InferenceService,
    dog_breed_inference_service: InferenceService,
    kserve_raw_headless_service_config: DataScienceCluster,
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
    kserve_raw_headless_service_config: DataScienceCluster,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="dog-cat-classifier",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime.name,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.CAT_DOG_ONNX,
        model_format=ModelFormat.ONNX,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        protocol_version="v2",
    ) as isvc:
        # Canary check: verify RawDeployment is using a headed service (non-headless).
        services = get_services_by_isvc_label(
            client=admin_client,
            isvc=isvc,
            runtime_name=ovms_kserve_serving_runtime.name,
        )
        is_headed = any(
            getattr(svc.instance.spec, "clusterIP", None) and svc.instance.spec.clusterIP != "None" for svc in services
        )
        assert is_headed, (
            "Expected Headed RawDeployment for dog-cat-classifier, but predictor Service is headless. "
            "Check rawDeploymentServiceConfig in DataScienceCluster."
        )
        yield isvc


@pytest.fixture(scope="class")
def dog_breed_inference_service(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
    models_endpoint_s3_secret: Secret,
    kserve_raw_headless_service_config: DataScienceCluster,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="dog-breed-classifier",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime.name,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.DOG_BREED_ONNX,
        model_format=ModelFormat.ONNX,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
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
    with (
        create_inference_graph_view_role(
            client=admin_client,
            name=f"{dog_breed_inference_graph.name}-view",
            namespace=unprivileged_model_namespace.name,
            resource_names=[dog_breed_inference_graph.name],
        ) as role,
        RoleBinding(
            client=admin_client,
            namespace=unprivileged_model_namespace.name,
            name=f"{bare_service_account.name}-view",
            role_ref_name=role.name,
            role_ref_kind=role.kind,
            subjects_kind=bare_service_account.kind,
            subjects_name=bare_service_account.name,
        ),
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
