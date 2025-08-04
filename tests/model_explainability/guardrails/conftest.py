from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import (
    KServeDeploymentType,
)
from utilities.inference_utils import create_isvc


GUARDRAILS_ORCHESTRATOR_NAME = "guardrails-orchestrator"


# ServingRuntimes, InferenceServices, and related resources
@pytest.fixture(scope="class")
def huggingface_sr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntime(
        client=admin_client,
        name="guardrails-detector-runtime-prompt-injection",
        namespace=model_namespace.name,
        containers=[
            {
                "name": "kserve-container",
                "image": "quay.io/trustyai/guardrails-detector-huggingface-runtime:v0.2.0",
                "command": ["uvicorn", "app:app"],
                "args": [
                    "--workers=4",
                    "--host=0.0.0.0",
                    "--port=8000",
                    "--log-config=/common/log_conf.yaml",
                ],
                "env": [
                    {"name": "MODEL_DIR", "value": "/mnt/models"},
                    {"name": "HF_HOME", "value": "/tmp/hf_home"},
                ],
                "ports": [{"containerPort": 8000, "protocol": "TCP"}],
            }
        ],
        supported_model_formats=[{"name": "guardrails-detector-huggingface", "autoSelect": True}],
        multi_model=False,
        annotations={
            "openshift.io/display-name": "Guardrails Detector ServingRuntime for KServe",
            "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
            "prometheus.io/port": "8080",
            "prometheus.io/path": "/metrics",
        },
        label={
            "opendatahub.io/dashboard": "true",
        },
    ) as serving_runtime:
        yield serving_runtime


@pytest.fixture(scope="class")
def prompt_injection_detector_isvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_data_connection: Secret,
    huggingface_sr: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="prompt-injection-detector",
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format="guardrails-detector-huggingface",
        runtime=huggingface_sr.name,
        storage_key=minio_data_connection.name,
        storage_path="deberta-v3-base-prompt-injection-v2",
        wait_for_predictor_pods=False,
        enable_auth=False,
        resources={
            "requests": {"cpu": "1", "memory": "2Gi", "nvidia.com/gpu": "0"},
            "limits": {"cpu": "1", "memory": "2Gi", "nvidia.com/gpu": "0"},
        },
        max_replicas=1,
        min_replicas=1,
        labels={
            "opendatahub.io/dashboard": "true",
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def prompt_injection_detector_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    prompt_injection_detector_isvc: InferenceService,
) -> Generator[Route, Any, Any]:
    yield Route(
        name="prompt-injection-detector-route",
        namespace=model_namespace.name,
        service=prompt_injection_detector_isvc.name,
        wait_for_resource=True,
    )


# Other "helper" fixtures
@pytest.fixture(scope="class")
def openshift_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    return create_ca_bundle_file(client=admin_client, ca_type="openshift")
