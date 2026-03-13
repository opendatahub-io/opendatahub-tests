from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_explainability.trustyai_service.trustyai_service_utils import (
    wait_for_isvc_deployment_registered_by_trustyai_service,
)
from utilities.constants import KServeDeploymentType, MinIo, ModelFormat, RuntimeTemplates
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def ovms_runtime(
    admin_client: DynamicClient, minio_data_connection: Secret, model_namespace: Namespace
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=f"{ModelFormat.OVMS}-1.x",
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
        multi_model=False,
        enable_http=False,
        enable_grpc=True,
        model_format_name={"name": ModelFormat.ONNX, "version": "1"},
        # TODO: Remove runtime_image once model works with latest ovms
        runtime_image=MinIo.RunTimeConfig.IMAGE,
    ) as sr:
        yield sr


@pytest.fixture(scope="class")
def onnx_loan_model(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_pod: Pod,
    minio_service: Service,
    minio_data_connection: Secret,
    ovms_runtime: ServingRuntime,
    kserve_raw_config: ConfigMap,
    kserve_logger_ca_bundle: ConfigMap,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="demo-loan-nn-onnx-alpha",
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format=ModelFormat.ONNX,
        runtime=ovms_runtime.name,
        storage_key=minio_data_connection.name,
        storage_path="ovms/loan_model_alpha",
        min_replicas=1,
        resources={"limits": {"cpu": "2", "memory": "8Gi"}, "requests": {"cpu": "1", "memory": "4Gi"}},
        enable_auth=True,
        external_route=True,
        model_version="1",
        wait=True,
        wait_for_predictor_pods=False,
    ) as isvc:
        wait_for_isvc_deployment_registered_by_trustyai_service(
            client=admin_client,
            isvc=isvc,
            runtime_name=ovms_runtime.name,
        )
        yield isvc
