from typing import Any, Generator

import pytest
import yaml
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.authorino import Authorino
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.service_mesh_control_plane import ServiceMeshControlPlane
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.storage_class import StorageClass
from ocp_utilities.monitoring import Prometheus
from pytest_testconfig import config as py_config

from utilities.constants import StorageClassName
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelInferenceRuntime,
    Protocols,
    RuntimeTemplates,
)
from utilities.constants import (
    ModelAndFormat,
    ModelVersion,
)
from utilities.inference_utils import create_isvc
from utilities.infra import (
    get_openshift_token,
    s3_endpoint_secret,
    update_configmap_data,
)
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="package")
def skip_if_no_deployed_openshift_serverless(admin_client: DynamicClient) -> None:
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
) -> Generator[Secret, Any, Any]:
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
def model_service_account(
    admin_client: DynamicClient, models_endpoint_s3_secret: Secret
) -> Generator[ServiceAccount, Any, Any]:
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
        "models_priorities": request.param.get("models-priorities"),
        "supported_model_formats": request.param.get("supported-model-formats"),
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
) -> Generator[InferenceService, Any, Any]:
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

    if (scale_metric := request.param.get("scale-metric")) is not None:
        isvc_kwargs["scale_metric"] = scale_metric

    if (scale_target := request.param.get("scale-target")) is not None:
        isvc_kwargs["scale_target"] = scale_target

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
def skip_if_no_deployed_redhat_authorino_operator(admin_client: DynamicClient) -> None:
    name = "authorino"
    namespace = f"{py_config['applications_namespace']}-auth-provider"

    if not Authorino(
        client=admin_client,
        name=name,
        namespace=namespace,
    ).exists:
        pytest.skip(f"Authorino {name} CR is missing from {namespace} namespace")


@pytest.fixture(scope="package")
def skip_if_no_deployed_openshift_service_mesh(admin_client: DynamicClient) -> None:
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
    ci_endpoint_s3_secret: Secret,
    ci_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=f"{Protocols.HTTP}-{ModelFormat.OPENVINO}",
        namespace=model_namespace.name,
        runtime=http_s3_ovms_model_mesh_serving_runtime.name,
        model_service_account=ci_service_account.name,
        storage_key=ci_endpoint_s3_secret.name,
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
) -> Generator[ServingRuntime, Any, Any]:
    runtime_kwargs = {
        "client": admin_client,
        "namespace": model_namespace.name,
        "name": f"{Protocols.HTTP}-{ModelInferenceRuntime.OPENVINO_RUNTIME}",
        "template_name": RuntimeTemplates.OVMS_MODEL_MESH,
        "multi_model": True,
        "protocol": Protocols.REST.upper(),
        "resources": {
            ModelFormat.OVMS: {
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
def openvino_kserve_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=request.param["runtime-name"],
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
        multi_model=False,
        resources={
            ModelFormat.OVMS: {
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
) -> Generator[Secret, Any, Any]:
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
def ci_service_account(
    admin_client: DynamicClient, ci_endpoint_s3_secret: Secret
) -> Generator[ServiceAccount, Any, Any]:
    with ServiceAccount(
        client=admin_client,
        namespace=ci_endpoint_s3_secret.namespace,
        name="ci-models-bucket-sa",
        secrets=[{"name": ci_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def ovms_kserve_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    openvino_kserve_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    deployment_mode = request.param["deployment-mode"]
    isvc_kwargs = {
        "client": admin_client,
        "name": f"{request.param['name']}-{deployment_mode.lower()}",
        "namespace": model_namespace.name,
        "runtime": openvino_kserve_serving_runtime.name,
        "storage_path": request.param["model-dir"],
        "storage_key": ci_endpoint_s3_secret.name,
        "model_format": ModelAndFormat.OPENVINO_IR,
        "deployment_mode": deployment_mode,
        "model_version": request.param["model-version"],
    }

    if env_vars := request.param.get("env-vars"):
        isvc_kwargs["model_env_variables"] = env_vars

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    if max_replicas := request.param.get("max-replicas"):
        isvc_kwargs["max_replicas"] = max_replicas

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def http_s3_tensorflow_model_mesh_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    http_s3_ovms_model_mesh_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
    ci_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=f"{Protocols.HTTP}-{ModelFormat.TENSORFLOW}",
        namespace=model_namespace.name,
        runtime=http_s3_ovms_model_mesh_serving_runtime.name,
        model_service_account=ci_service_account.name,
        storage_key=ci_endpoint_s3_secret.name,
        storage_path=request.param["model-path"],
        model_format=ModelFormat.TENSORFLOW,
        deployment_mode=KServeDeploymentType.MODEL_MESH,
        model_version="2",
    ) as isvc:
        yield isvc


@pytest.fixture(scope="session")
def prometheus(admin_client: DynamicClient) -> Prometheus:
    return Prometheus(
        client=admin_client,
        resource_name="thanos-querier",
        verify_ssl=False,
        bearer_token=get_openshift_token(),
    )


@pytest.fixture(scope="class")
def user_workload_monitoring_config_map(
    admin_client: DynamicClient, cluster_monitoring_config: ConfigMap
) -> Generator[ConfigMap, None, None]:
    data = {
        "config.yaml": yaml.dump({
            "prometheus": {
                "logLevel": "debug",
                "retention": "15d",
                "volumeClaimTemplate": {"spec": {"resources": {"requests": {"storage": "40Gi"}}}},
            }
        })
    }

    with update_configmap_data(
        client=admin_client,
        name="user-workload-monitoring-config",
        namespace="openshift-user-workload-monitoring",
        data=data,
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def http_s3_ovms_external_route_model_mesh_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    runtime_kwargs = {
        "client": admin_client,
        "namespace": model_namespace.name,
        "name": f"{Protocols.HTTP}-{ModelInferenceRuntime.OPENVINO_RUNTIME}-exposed",
        "template_name": RuntimeTemplates.OVMS_MODEL_MESH,
        "multi_model": True,
        "protocol": Protocols.REST.upper(),
        "resources": {
            ModelFormat.OVMS: {
                "requests": {"cpu": "1", "memory": "4Gi"},
                "limits": {"cpu": "2", "memory": "8Gi"},
            },
        },
        "enable_external_route": True,
    }

    if hasattr(request, "param"):
        runtime_kwargs["enable_auth"] = request.param.get("enable-auth")

    with ServingRuntimeFromTemplate(**runtime_kwargs) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def http_s3_openvino_second_model_mesh_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    ci_endpoint_s3_secret: Secret,
    ci_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    # Dynamically select the used ServingRuntime by passing "runtime-fixture-name" request.param
    runtime = request.getfixturevalue(argname=request.param["runtime-fixture-name"])
    with create_isvc(
        client=admin_client,
        name=f"{Protocols.HTTP}-{ModelFormat.OPENVINO}-2",
        namespace=model_namespace.name,
        runtime=runtime.name,
        model_service_account=ci_service_account.name,
        storage_key=ci_endpoint_s3_secret.name,
        storage_path=request.param["model-path"],
        model_format=request.param["model-format"],
        deployment_mode=KServeDeploymentType.MODEL_MESH,
        model_version=request.param["model-version"],
    ) as isvc:
        yield isvc
