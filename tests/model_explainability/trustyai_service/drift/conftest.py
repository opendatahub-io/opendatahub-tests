from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.trustyai_service import TrustyAIService

from tests.model_explainability.trustyai_service.trustyai_service_utils import (
    wait_for_isvc_deployment_registered_by_trustyai_service,
)
from utilities.constants import KServeDeploymentType, Ports, Timeout, Labels
from utilities.inference_utils import create_isvc

MLSERVER: str = "mlserver"
MLSERVER_RUNTIME_NAME: str = f"{MLSERVER}-1.x"
XGBOOST: str = "xgboost"
SKLEARN: str = "sklearn"
LIGHTGBM: str = "lightgbm"
MLFLOW: str = "mlflow"
TIMEOUT_20MIN: int = 20 * Timeout.TIMEOUT_1MIN


@pytest.fixture(scope="class")
def mlserver_runtime(
    admin_client: DynamicClient, minio_data_connection: Secret, model_namespace: Namespace
) -> Generator[ServingRuntime, Any, Any]:
    supported_model_formats = [
        {"name": SKLEARN, "version": "0", "autoSelect": True, "priority": 2},
        {"name": SKLEARN, "version": "1", "autoSelect": True, "priority": 2},
        {"name": XGBOOST, "version": "1", "autoSelect": True, "priority": 2},
        {"name": XGBOOST, "version": "2", "autoSelect": True, "priority": 2},
        {"name": LIGHTGBM, "version": "3", "autoSelect": True, "priority": 2},
        {"name": LIGHTGBM, "version": "4", "autoSelect": True, "priority": 2},
        {"name": MLFLOW, "version": "1", "autoSelect": True, "priority": 1},
        {"name": MLFLOW, "version": "2", "autoSelect": True, "priority": 1},
    ]
    containers = [
        {
            "name": "kserve-container",
            "image": "quay.io/trustyai_testing/mlserver"
            "@sha256:68a4cd74fff40a3c4f29caddbdbdc9e54888aba54bf3c5f78c8ffd577c3a1c89",
            "env": [
                {"name": "MLSERVER_MODEL_IMPLEMENTATION", "value": "{{.Labels.modelClass}}"},
                {"name": "MLSERVER_HTTP_PORT", "value": str(Ports.REST_PORT)},
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
            "prometheus.io/port": str(Ports.REST_PORT),
            "openshift.io/display-name": "mlserver-1.x",
        },
        label={Labels.OpenDataHub.DASHBOARD: "true"},
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
        runtime=mlserver_runtime.name,
        storage_key=minio_data_connection.name,
        storage_path="sklearn/gaussian_credit_model/1",
        enable_auth=True,
        wait_for_predictor_pods=False,
        resources={"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "1", "memory": "2Gi"}},
    ) as isvc:
        wait_for_isvc_deployment_registered_by_trustyai_service(
            client=admin_client,
            isvc=isvc,
            runtime_name=mlserver_runtime.name,
        )
        yield isvc
