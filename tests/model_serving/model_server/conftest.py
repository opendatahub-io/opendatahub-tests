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
from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelFormat,
    ModelInferenceRuntime,
    ModelVersion,
    Protocols,
    RuntimeTemplates,
)
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


@pytest.fixture(scope="package")
def skip_if_no_deployed_redhat_authorino_operator(admin_client: DynamicClient):
    name = "authorino"
    namespace = f"{py_config['applications_namespace']}-auth-provider"

    if not Authorino(
        client=admin_client,
        name=name,
        namespace=namespace,
    ).exists:
        pytest.skip(f"Authorino {name} CR is missing from {namespace} namespace")


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


@pytest.fixture(scope="class")
def http_s3_openvino_model_mesh_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    http_s3_ovms_model_mesh_serving_runtime: ServingRuntime,
    ci_model_mesh_endpoint_s3_secret: Secret,
    model_mesh_model_service_account: ServiceAccount,
) -> InferenceService:
    with create_isvc(
        client=admin_client,
        name=f"{Protocols.HTTP}-{ModelFormat.OPENVINO}",
        namespace=model_namespace.name,
        runtime=http_s3_ovms_model_mesh_serving_runtime.name,
        model_service_account=model_mesh_model_service_account.name,
        storage_key=ci_model_mesh_endpoint_s3_secret.name,
        storage_path=request.param["model-path"],
        model_format=ModelAndFormat.OPENVINO_IR,
        deployment_mode=KServeDeploymentType.MODEL_MESH,
        model_version=ModelVersion.OPSET1,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_s3_ovms_model_mesh_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> ServingRuntime:
    runtime_kwargs = {
        "client": admin_client,
        "namespace": model_namespace.name,
        "name": f"{Protocols.HTTP}-{ModelInferenceRuntime.OPENVINO_RUNTIME}",
        "template_name": RuntimeTemplates.OVMS_MODEL_MESH,
        "multi_model": True,
        "protocol": "REST",
        "resources": {
            "ovms": {
                "requests": {"cpu": "1", "memory": "4Gi"},
                "limits": {"cpu": "2", "memory": "8Gi"},
            }
        },
    }

    enable_external_route = False
    enable_auth = False

    if hasattr(request, "param"):
        enable_external_route = request.param.get("enable-external-route")
        enable_auth = request.param.get("enable-auth")

    runtime_kwargs["enable_external_route"] = enable_external_route
    runtime_kwargs["enable_auth"] = enable_auth

    with ServingRuntimeFromTemplate(**runtime_kwargs) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def ci_model_mesh_endpoint_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Secret:
    with s3_endpoint_secret(
        admin_client=admin_client,
        name="ci-bucket-secret",
        namespace=model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def model_mesh_model_service_account(
    admin_client: DynamicClient, ci_model_mesh_endpoint_s3_secret: Secret
) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=ci_model_mesh_endpoint_s3_secret.namespace,
        name=f"{Protocols.HTTP}-models-bucket-sa",
        secrets=[{"name": ci_model_mesh_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def openvino_kserve_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> ServingRuntime:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=request.param["runtime-name"],
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
        multi_model=False,
        resources={
            "ovms": {
                "requests": {"cpu": "1", "memory": "4Gi"},
                "limits": {"cpu": "2", "memory": "8Gi"},
            }
        },
        model_format_name=request.param["model-format"],
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def ci_endpoint_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
) -> Secret:
    with s3_endpoint_secret(
        admin_client=admin_client,
        name="ci-bucket-secret",
        namespace=model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=ci_s3_bucket_region,
        aws_s3_bucket=ci_s3_bucket_name,
        aws_s3_endpoint=ci_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def ovms_serverless_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    openvino_kserve_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> InferenceService:
    with create_isvc(
        client=admin_client,
        name=f"{request.param['name']}-serverless",
        namespace=model_namespace.name,
        runtime=openvino_kserve_serving_runtime.name,
        storage_path=request.param["model-dir"],
        storage_key=ci_endpoint_s3_secret.name,
        model_format=ModelAndFormat.OPENVINO_IR,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        model_version=request.param["model-version"],
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_s3_tensorflow_model_mesh_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    http_s3_ovms_model_mesh_serving_runtime: ServingRuntime,
    ci_model_mesh_endpoint_s3_secret: Secret,
    model_mesh_model_service_account: ServiceAccount,
) -> InferenceService:
    with create_isvc(
        client=admin_client,
        name=f"{Protocols.HTTP}-{ModelFormat.TENSORFLOW}",
        namespace=model_namespace.name,
        runtime=http_s3_ovms_model_mesh_serving_runtime.name,
        model_service_account=model_mesh_model_service_account.name,
        storage_key=ci_model_mesh_endpoint_s3_secret.name,
        storage_path=request.param["model-path"],
        model_format=ModelFormat.TENSORFLOW,
        deployment_mode=KServeDeploymentType.MODEL_MESH,
        model_version="2",
    ) as isvc:
        yield isvc
