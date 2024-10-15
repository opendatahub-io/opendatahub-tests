import os
import shlex
from typing import List, Optional, Tuple

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.service_mesh_member import ServiceMeshMember
from ocp_resources.serving_runtime import ServingRuntime
from ocp_utilities.infra import get_pods_by_name_prefix
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.storage.pvc.utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="session")
def aws_access_key() -> Optional[str]:
    access_key = py_config.get("aws_access_key", os.environ.get("AWS_ACCESS_KEY_ID"))
    if not access_key:
        raise ValueError("AWS access key is not set")

    return access_key


@pytest.fixture(scope="session")
def aws_secret_access_key() -> Optional[str]:
    secret_access_key = py_config.get("aws_secret_key", os.environ.get("AWS_SECRET_ACCESS_KEY"))
    if not secret_access_key:
        raise ValueError("AWS secret key is not set")

    return secret_access_key


@pytest.fixture(scope="session")
def valid_aws_config(aws_access_key: str, aws_secret_access_key: str) -> Tuple[str, str]:
    return aws_access_key, aws_secret_access_key


@pytest.fixture(scope="class")
def service_mesh_member(admin_client: DynamicClient, model_namespace: Namespace) -> ServiceMeshMember:
    with ServiceMeshMember(
        client=admin_client,
        name="default",
        namespace=model_namespace.name,
        control_plane_ref={"name": "data-science-smcp", "namespace": "istio-system"},
    ) as smm:
        yield smm


@pytest.fixture(scope="class")
def ci_s3_storage_uri(request) -> str:
    return f"s3://{py_config['ci_s3_bucket_name']}/{request.param['model-dir']}/"


@pytest.fixture(scope="class")
def model_pvc(
    request,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> PersistentVolumeClaim:
    access_mode = "ReadWriteOnce"
    pvc_kwargs = {
        "name": "model-pvc",
        "namespace": model_namespace.name,
        "client": admin_client,
        "size": "15Gi",
    }
    if hasattr(request, "param"):
        access_mode = request.param.get("access-modes")

        if storage_class_name := request.param.get("storage-class-name"):
            pvc_kwargs["storage_class"] = storage_class_name

    pvc_kwargs["accessmodes"] = access_mode

    with PersistentVolumeClaim(**pvc_kwargs) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def downloaded_model_data(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    ci_s3_storage_uri: str,
    model_pvc: PersistentVolumeClaim,
    aws_secret_access_key: str,
    aws_access_key: str,
) -> str:
    mount_path: str = "data"
    model_dir: str = "model-dir"
    containers = [
        {
            "name": "model-downloader",
            "image": "quay.io/redhat_msi/qe-tools-base-image",
            "args": [
                "sh",
                "-c",
                "sleep infinity",
            ],
            "env": [
                {"name": "AWS_ACCESS_KEY_ID", "value": aws_access_key},
                {"name": "AWS_SECRET_ACCESS_KEY", "value": aws_secret_access_key},
            ],
            "volumeMounts": [{"mountPath": mount_path, "name": model_pvc.name, "subPath": model_dir}],
        }
    ]
    volumes = [{"name": model_pvc.name, "persistentVolumeClaim": {"claimName": model_pvc.name}}]

    with Pod(
        client=admin_client,
        namespace=model_namespace.name,
        name="download-model-data",
        containers=containers,
        volumes=volumes,
    ) as pod:
        pod.wait_for_status(status=Pod.Status.RUNNING)
        pod.execute(command=shlex.split(f"aws s3 cp --recursive {ci_s3_storage_uri} /{mount_path}/{model_dir}"))

    return model_dir


@pytest.fixture(scope="class")
def serving_runtime(
    request,
    admin_client: DynamicClient,
    service_mesh_member: ServiceMeshMember,
    model_namespace: Namespace,
    downloaded_model_data: str,
) -> ServingRuntime:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        template_name=request.param["template-name"],
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def inference_service(
    request,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    serving_runtime: ServingRuntime,
    model_pvc: PersistentVolumeClaim,
    downloaded_model_data: str,
) -> InferenceService:
    isvc_kwargs = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": serving_runtime.name,
        "storage_uri": f"pvc://{model_pvc.name}/{downloaded_model_data}",
        "model_format": serving_runtime.instance.spec.supportedModelFormats[0].name,
        "deployment_mode": request.param.get("deployment-mode", "Serverless"),
    }

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def isvc_deployment_ready(admin_client: DynamicClient, inference_service: InferenceService) -> None:
    deployment_name_prefix = f"{inference_service.name}-predictor"
    deployment = list(
        Deployment.get(
            dyn_client=admin_client,
            namespace=inference_service.namespace,
        )
    )

    if deployment and deployment[0].name.startswith(deployment_name_prefix):
        deployment[0].wait_for_replicas()
        return

    raise ResourceNotFoundError(f"Deployment with prefix {deployment_name_prefix} not found")


@pytest.fixture()
def predictor_pods_scope_function(admin_client: DynamicClient, inference_service: InferenceService) -> List[Pod]:
    return get_pods_by_name_prefix(
        client=admin_client,
        pod_prefix=f"{inference_service.name}-predictor",
        namespace=inference_service.namespace,
    )


@pytest.fixture(scope="class")
def predictor_pods_scope_class(
    admin_client: DynamicClient, inference_service: InferenceService, isvc_deployment_ready: None
) -> List[Pod]:
    return get_pods_by_name_prefix(
        client=admin_client,
        pod_prefix=f"{inference_service.name}-predictor",
        namespace=inference_service.namespace,
    )


@pytest.fixture()
def patched_isvc(request, inference_service: InferenceService, first_predictor_pod: Pod) -> InferenceService:
    with ResourceEditor(
        patches={
            inference_service: {
                "metadata": {
                    "annotations": {"storage.kserve.io/readonly": request.param["readonly"]},
                }
            }
        }
    ):
        first_predictor_pod.wait_deleted()
        yield inference_service
