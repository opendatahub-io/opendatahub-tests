from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from utilities.constants import KServeDeploymentType
from utilities.general import download_model_data
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label


@pytest.fixture(scope="class")
def ci_bucket_downloaded_model_data(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    model_pvc: PersistentVolumeClaim,
    aws_secret_access_key: str,
    aws_access_key_id: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_endpoint: str,
    ci_s3_bucket_region: str,
) -> str:
    return download_model_data(
        client=admin_client,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        model_namespace=unprivileged_model_namespace.name,
        model_pvc_name=model_pvc.name,
        bucket_name=ci_s3_bucket_name,
        aws_endpoint_url=ci_s3_bucket_endpoint,
        aws_default_region=ci_s3_bucket_region,
        model_path=request.param["model-dir"],
        use_sub_path=True,
    )


@pytest.fixture()
def predictor_pods_scope_function(
    unprivileged_client: DynamicClient, pvc_inference_service: InferenceService
) -> list[Pod]:
    return get_pods_by_isvc_label(
        client=unprivileged_client,
        isvc=pvc_inference_service,
    )


@pytest.fixture(scope="class")
def predictor_pods_scope_class(
    unprivileged_client: DynamicClient,
    pvc_inference_service: InferenceService,
) -> list[Pod]:
    return get_pods_by_isvc_label(
        client=unprivileged_client,
        isvc=pvc_inference_service,
    )


@pytest.fixture()
def patched_read_only_isvc(
    request: FixtureRequest, pvc_inference_service: InferenceService, first_predictor_pod: Pod
) -> Generator[InferenceService, Any, Any]:
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


@pytest.fixture(scope="class")
def pvc_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    serving_runtime_from_template: ServingRuntime,
    model_pvc: PersistentVolumeClaim,
    ci_bucket_downloaded_model_data: str,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": unprivileged_client,
        "name": request.param["name"],
        "namespace": unprivileged_model_namespace.name,
        "runtime": serving_runtime_from_template.name,
        "storage_uri": f"pvc://{model_pvc.name}/{ci_bucket_downloaded_model_data}",
        "model_format": serving_runtime_from_template.instance.spec.supportedModelFormats[0].name,
        "deployment_mode": request.param.get("deployment-mode", KServeDeploymentType.SERVERLESS),
        "wait_for_predictor_pods": True,
    }

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture()
def first_predictor_pod(predictor_pods_scope_function: list[Pod]) -> Pod:
    return predictor_pods_scope_function[0]
