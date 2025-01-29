from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.authorino import Authorino
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.service_mesh_control_plane import ServiceMeshControlPlane
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.storage_class import StorageClass
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.utils import create_isvc
from utilities.constants import DscComponents, StorageClassName
from utilities.infra import s3_endpoint_secret
from utilities.data_science_cluster_utils import update_components_in_dsc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="package")
def skip_if_no_deployed_openshift_serverless(admin_client: DynamicClient):
    name = "openshift-serverless"
    csvs = list(
        ClusterServiceVersion.get(
            client=admin_client,
            namespace=name,
            label_selector=f"operators.coreos.com/serverless-operator.{name}",
        )
    )
    if not csvs:
        pytest.skip("OpenShift Serverless is not deployed")

    csv = csvs[0]

    if not (csv.exists and csv.status == csv.Status.SUCCEEDED):
        pytest.skip("OpenShift Serverless is not deployed")


@pytest.fixture(scope="class")
def models_endpoint_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Secret:
    with s3_endpoint_secret(
        admin_client=admin_client,
        name="models-bucket-secret",
        namespace=model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_bucket=models_s3_bucket_name,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


# HTTP model serving
@pytest.fixture(scope="class")
def model_service_account(admin_client: DynamicClient, models_endpoint_s3_secret: Secret) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=models_endpoint_s3_secret.namespace,
        name="models-bucket-sa",
        secrets=[{"name": models_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def serving_runtime_from_template(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    runtime_kwargs = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "template_name": request.param["template-name"],
        "multi_model": request.param["multi-model"],
    }

    if (enable_http := request.param.get("enable-http")) is not None:
        runtime_kwargs["enable_http"] = enable_http

    if (enable_grpc := request.param.get("enable-grpc")) is not None:
        runtime_kwargs["enable_grpc"] = enable_grpc

    with ServingRuntimeFromTemplate(**runtime_kwargs) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def s3_models_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    serving_runtime_from_template: ServingRuntime,
    models_endpoint_s3_secret: Secret,
) -> InferenceService:
    isvc_kwargs = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": serving_runtime_from_template.name,
        "model_format": serving_runtime_from_template.instance.spec.supportedModelFormats[0].name,
        "deployment_mode": request.param["deployment-mode"],
        "storage_key": models_endpoint_s3_secret.name,
        "storage_path": request.param["model-dir"],
    }

    if (external_route := request.param.get("external-route")) is not None:
        isvc_kwargs["external_route"] = external_route

    if (enable_auth := request.param.get("enable-auth")) is not None:
        isvc_kwargs["enable_auth"] = enable_auth

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def model_pvc(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    access_mode = "ReadWriteOnce"
    pvc_kwargs = {
        "name": "model-pvc",
        "namespace": model_namespace.name,
        "client": admin_client,
        "size": request.param["pvc-size"],
    }
    if hasattr(request, "param"):
        access_mode = request.param.get("access-modes")

        if storage_class_name := request.param.get("storage-class-name"):
            pvc_kwargs["storage_class"] = storage_class_name

    pvc_kwargs["accessmodes"] = access_mode

    with PersistentVolumeClaim(**pvc_kwargs) as pvc:
        pvc.wait_for_status(status=pvc.Status.BOUND, timeout=120)
        yield pvc


@pytest.fixture(scope="session")
def skip_if_no_nfs_storage_class(admin_client: DynamicClient) -> None:
    if not StorageClass(client=admin_client, name=StorageClassName.NFS).exists:
        pytest.skip(f"StorageClass {StorageClassName.NFS} is missing from the cluster")


@pytest.fixture(scope="class")
def modelmesh_management_state(dsc_resource: DataScienceCluster) -> str:
    return dsc_resource.instance.spec.components[DscComponents.MODELMESHSERVING].managementState


@pytest.fixture(scope="class")
def kserve_management_state(dsc_resource: DataScienceCluster) -> str:
    return dsc_resource.instance.spec.components[DscComponents.KSERVE].managementState


@pytest.fixture(scope="package")
def skip_if_no_redhat_authorino_operator(admin_client: DynamicClient):
    name = "authorino"
    if not Authorino(
        client=admin_client,
        name=name,
        namespace=f"{py_config['applications_namespace']}-auth-provider",
    ).exists:
        pytest.skip(f"{name} operator is missing from the cluster")


@pytest.fixture(scope="package")
def enabled_kserve_in_dsc(dsc_resource: DataScienceCluster) -> Generator[DataScienceCluster, Any, Any]:
    with update_components_in_dsc(
        dsc=dsc_resource,
        components={DscComponents.KSERVE: DscComponents.ManagementState.MANAGED},
    ) as dsc:
        yield dsc


@pytest.fixture(scope="package")
def enabled_modelmesh_in_dsc(dsc_resource: DataScienceCluster) -> Generator[DataScienceCluster, Any, Any]:
    with update_components_in_dsc(
        dsc=dsc_resource,
        components={DscComponents.MODELMESHSERVING: DscComponents.ManagementState.MANAGED},
    ) as dsc:
        yield dsc


@pytest.fixture(scope="package")
def skip_if_no_deployed_openshift_service_mesh(admin_client: DynamicClient):
    smcp = ServiceMeshControlPlane(client=admin_client, name="data-science-smcp", namespace="istio-system")
    if not smcp or not smcp.exists:
        pytest.skip("OpenShift service mesh operator is not deployed")

    smcp.wait_for_condition(condition=smcp.Condition.READY, status="True")
