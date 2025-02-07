from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.trustyai_service import TrustyAIService

from tests.model_explainability.constants import TIMEOUT_1MIN
from tests.model_explainability.trustyai_service.utils import wait_for_isvc_deployment_registered_by_trustyaiservice
from utilities.constants import KServeDeploymentType
from utilities.infra import create_isvc

MLSERVER: str = "mlserver"
MLSERVER_RUNTIME_NAME: str = f"{MLSERVER}-1.x"
MLSERVER_QUAY_IMAGE: str = "quay.io/rh-ee-mmisiura/mlserver:1.6.1"
XGBOOST = "xgboost"
SKLEARN = "sklearn"
TIMEOUT_20MIN = 20 * TIMEOUT_1MIN


@pytest.fixture(scope="class")
def mlserver_runtime(
    admin_client: DynamicClient, minio_data_connection: Secret, model_namespace: Namespace
) -> Generator[ServingRuntime, Any, Any]:
    supported_model_formats = [
        {"name": "sklearn", "version": "0", "autoSelect": True, "priority": 2},
        {"name": "sklearn", "version": "1", "autoSelect": True, "priority": 2},
        {"name": "xgboost", "version": "1", "autoSelect": True, "priority": 2},
        {"name": "xgboost", "version": "2", "autoSelect": True, "priority": 2},
        {"name": "lightgbm", "version": "3", "autoSelect": True, "priority": 2},
        {"name": "lightgbm", "version": "4", "autoSelect": True, "priority": 2},
        {"name": "mlflow", "version": "1", "autoSelect": True, "priority": 1},
        {"name": "mlflow", "version": "2", "autoSelect": True, "priority": 1},
    ]
    containers = [
        {
            "name": "kserve-container",
            "image": "quay.io/rh-ee-mmisiura/mlserver:1.6.1",
            "env": [
                {"name": "MLSERVER_MODEL_IMPLEMENTATION", "value": "{{.Labels.modelClass}}"},
                {"name": "MLSERVER_HTTP_PORT", "value": "8080"},
                {"name": "MLSERVER_GRPC_PORT", "value": "9000"},
                {"name": "MODELS_DIR", "value": "/mnt/models/"},
            ],
            "resources": {"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "1", "memory": "2Gi"}},
        }
    ]

    with ServingRuntime(
        client=admin_client,
        name="kserve-mlserver",
        namespace=model_namespace.name,
        containers=containers,
        supported_model_formats=supported_model_formats,
        protocol_versions=["v2"],
        annotations={
            "opendatahub.io/accelerator-name": "",
            "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
            "opendatahub.io/template-display-name": "KServe MLServer",
            "prometheus.kserve.io/path": "/metrics",
            "prometheus.io/port": "8080",
            "openshift.io/display-name": "mlserver-1.x",
        },
        label={"opendatahub.io/dashboard": "true"},
    ) as mlserver:
        yield mlserver


@pytest.fixture(scope="class")
def gaussian_credit_model(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_data_connection: Secret,
    mlserver_runtime: ServingRuntime,
    trustyai_service_with_pvc_storage: TrustyAIService,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="gaussian-credit-model",
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        model_format=XGBOOST,
        runtime="kserve-mlserver",
        storage_key=minio_data_connection.name,
        storage_path="sklearn/gaussian_credit_model/1",
        enable_auth=True,
        wait_for_predictor_pods=False,
        resources={"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "1", "memory": "2Gi"}},
    ) as isvc:
        wait_for_isvc_deployment_registered_by_trustyaiservice(
            client=admin_client,
            isvc=isvc,
            trustyai_service=trustyai_service_with_pvc_storage,
            runtime_name=mlserver_runtime,
        )
        yield isvc
