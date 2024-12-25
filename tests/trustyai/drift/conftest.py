import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.trustyai_service import TrustyAIService

from utilities.constants import MODELMESH_SERVING
from tests.trustyai.drift.utils import wait_for_modelmesh_pods_registered_by_trustyai

MLSERVER: str = "mlserver"
MLSERVER_RUNTIME_NAME: str = f"{MLSERVER}-1.x"
MLSERVER_QUAY_IMAGE: str = "quay.io/aaguirre/mlserver@sha256:8884d989b3063a47bf0e6c20c1c0ff253662121a977fe5b74b54e682839360d4"  # TODO: Move this image to a better place
XGBOOST = "xgboost"
SKLEARN = "sklearn"


@pytest.fixture(scope="class")
def mlserver_runtime(
    admin_client: DynamicClient, minio_data_connection: Secret, ns_with_modelmesh_enabled: Namespace
) -> ServingRuntime:
    supported_model_formats = [
        {"name": SKLEARN, "version": "0", "autoselect": "true"},
        {"name": XGBOOST, "version": "1", "autoselect": "true"},
        {"name": "lightgbm", "version": "3", "autoselect": "true"},
    ]
    containers = [
        {
            "name": MLSERVER,
            "image": MLSERVER_QUAY_IMAGE,
            "env": [
                {"name": "MLSERVER_MODELS_DIR", "value": "/models/_mlserver_models/"},
                {"name": "MLSERVER_GRPC_PORT", "value": "8001"},
                {"name": "MLSERVER_HTTP_PORT", "value": "8002"},
                {"name": "MLSERVER_LOAD_MODELS_AT_STARTUP", "value": "false"},
                {"name": "MLSERVER_MODEL_NAME", "value": "dummy-model-fixme"},
                {"name": "MLSERVER_HOST", "value": "127.0.0.1"},
                {"name": "MLSERVER_GRPC_MAX_MESSAGE_LENGTH", "value": "-1"},
            ],
            "resources": {"requests": {"cpu": "500m", "memory": "1Gi"}, "limits": {"cpu": "5", "memory": "1Gi"}},
        }
    ]

    with ServingRuntime(
        client=admin_client,
        name=MLSERVER_RUNTIME_NAME,
        namespace=ns_with_modelmesh_enabled.name,
        containers=containers,
        supported_model_formats=supported_model_formats,
        multi_model=True,
        protocol_versions=["grpc-v2"],
        grpc_endpoint="port:8085",
        grpc_data_endpoint="port:8001",
        built_in_adapter={
            "serverType": MLSERVER,
            "runtimeManagementPort": 8001,
            "memBufferBytes": 134217728,
            "modelLoadingTimeoutMillis": 90000,
        },
        annotations={"enable-route": "true"},
        label={"name": f"{MODELMESH_SERVING}-{MLSERVER_RUNTIME_NAME}-SR"},
    ) as mlserver:
        yield mlserver


@pytest.fixture(scope="class")
def gaussian_credit_model(
    admin_client: DynamicClient,
    ns_with_modelmesh_enabled: Namespace,
    minio_data_connection: Secret,
    mlserver_runtime: ServingRuntime,
    trustyai_service_with_pvc_storage: TrustyAIService,
) -> InferenceService:
    name = "gaussian-credit-model"
    with InferenceService(
        client=admin_client,
        name=name,
        namespace=ns_with_modelmesh_enabled.name,
        predictor={
            "model": {
                "modelFormat": {"name": XGBOOST},
                "runtime": mlserver_runtime.name,
                "storage": {"key": minio_data_connection.name, "path": f"{SKLEARN}/{name.replace('-', '_')}.json"},
            }
        },
        annotations={f"{InferenceService.ApiGroup.SERVING_KSERVE_IO}/deploymentMode": "ModelMesh"},
    ) as inference_service:
        deployment = Deployment(
            client=admin_client,
            namespace=ns_with_modelmesh_enabled.name,
            name=f"{MODELMESH_SERVING}-{mlserver_runtime.name}",
            wait_for_resource=True,
        )
        deployment.wait_for_replicas()
        wait_for_modelmesh_pods_registered_by_trustyai(client=admin_client, namespace=ns_with_modelmesh_enabled.name)
        yield inference_service
