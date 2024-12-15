from typing import Any, Generator, List

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_server.utils import create_isvc, get_pods_by_isvc_label
from utilities.constants import KServeDeploymentType
from utilities.general import download_model_data
from utilities.infra import wait_for_kserve_predictor_deployment_replicas
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def ci_s3_storage_uri(request: FixtureRequest, ci_s3_bucket_name: str) -> str:
    return f"s3://{ci_s3_bucket_name}/{request.param['model-dir']}/"


@pytest.fixture(scope="class")
def downloaded_model_data(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    ci_s3_storage_uri: str,
    model_pvc: PersistentVolumeClaim,
    aws_secret_access_key: str,
    aws_access_key_id: str,
) -> str:
    return download_model_data(
        admin_client=admin_client,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        storage_uri=ci_s3_storage_uri,
        model_namespace=model_namespace.name,
        model_pvc_name=model_pvc.name,
    )


@pytest.fixture(scope="class")
def pvc_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    downloaded_model_data: str,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        template_name=request.param["template-name"],
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def pvc_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    pvc_serving_runtime: ServingRuntime,
    model_pvc: PersistentVolumeClaim,
    downloaded_model_data: str,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": pvc_serving_runtime.name,
        "storage_uri": f"pvc://{model_pvc.name}/{downloaded_model_data}",
        "model_format": pvc_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "deployment_mode": request.param.get("deployment-mode", KServeDeploymentType.SERVERLESS),
    }

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def isvc_deployment_ready(admin_client: DynamicClient, pvc_inference_service: InferenceService) -> None:
    wait_for_kserve_predictor_deployment_replicas(
        client=admin_client,
        isvc=pvc_inference_service,
    )


@pytest.fixture()
def first_predictor_pod(predictor_pods_scope_function: List[Pod]) -> Pod:
    return predictor_pods_scope_function[0]


@pytest.fixture()
def predictor_pods_scope_function(admin_client: DynamicClient, pvc_inference_service: InferenceService) -> List[Pod]:
    return get_pods_by_isvc_label(
        client=admin_client,
        isvc=pvc_inference_service,
    )


@pytest.fixture(scope="class")
def predictor_pods_scope_class(
    admin_client: DynamicClient,
    pvc_inference_service: InferenceService,
    isvc_deployment_ready: None,
) -> List[Pod]:
    return get_pods_by_isvc_label(
        client=admin_client,
        isvc=pvc_inference_service,
    )


@pytest.fixture()
def patched_read_only_isvc(
    request, pvc_inference_service: InferenceService, first_predictor_pod: Pod
) -> InferenceService:
    with ResourceEditor(
        patches={
            pvc_inference_service: {
                "metadata": {
                    "annotations": {"storage.kserve.io/readonly": request.param["readonly"]},
                }
            }
        }
    ):
        first_predictor_pod.wait_deleted()
        yield pvc_inference_service
