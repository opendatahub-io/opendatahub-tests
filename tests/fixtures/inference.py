from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import py_config
from timeout_sampler import retry

from tests.fixtures.image_constants import FixturesImages
from utilities.constants import (
    KServeDeploymentType,
    LLMdInferenceSimConfig,
)
from utilities.inference_utils import create_isvc
from utilities.infra import get_data_science_cluster, wait_for_dsc_status_ready

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def llm_d_inference_sim_serving_runtime(
    admin_client: DynamicClient, model_namespace: Namespace, teardown_resources: bool, pytestconfig: pytest.Config
) -> Generator[ServingRuntime, Any, Any]:
    """Serving runtime for LLM-d Inference Simulator.

    While llm-d-inference-sim supports any model name, the /tokenizers endpoint will only support two models
        - qwen2.5-0.5b-instruct
        - Qwen2.5-1.5B-Instruct

    For other models, ensure:
        - the correct write permissions on the Pod
        - the model name matches what is available on HuggingFace (e.g., Qwen/Qwen2.5-1.5B-Instruct)
        - you have set a writeable "--tokenizers-cache-dir"
        - the cluster can pull from HuggingFace

    """
    if pytestconfig.option.post_upgrade:
        serving_runtime = ServingRuntime(
            client=admin_client,
            name=LLMdInferenceSimConfig.serving_runtime_name,
            namespace=model_namespace.name,
        )
        if not serving_runtime.exists:
            raise ResourceNotFoundError(
                f"ServingRuntime {LLMdInferenceSimConfig.serving_runtime_name} "
                f"does not exist in namespace {model_namespace.name} after upgrade"
            )
        yield serving_runtime
        serving_runtime.clean_up()

    else:
        with ServingRuntime(
            client=admin_client,
            name=LLMdInferenceSimConfig.serving_runtime_name,
            namespace=model_namespace.name,
            annotations={
                "description": "LLM-d Simulator KServe",
                "opendatahub.io/template-display-name": "LLM-d Inference Simulator Runtime",
                "openshift.io/display-name": "LLM-d Inference Simulator Runtime",
                "serving.kserve.io/enable-agent": "false",
            },
            label={
                "app.kubernetes.io/component": LLMdInferenceSimConfig.name,
                "app.kubernetes.io/instance": "llm-d-inference-sim-kserve",
                "app.kubernetes.io/name": "llm-d-sim",
                "app.kubernetes.io/version": "1.0.0",
                "opendatahub.io/dashboard": "true",
            },
            spec_annotations={
                "prometheus.io/path": "/metrics",
                "prometheus.io/port": "8000",
            },
            spec_labels={
                "opendatahub.io/dashboard": "true",
            },
            containers=[
                {
                    "name": "kserve-container",
                    "image": FixturesImages.LLMD_INFERENCE_SIM,
                    "imagePullPolicy": "Always",
                    "args": [
                        "--model",
                        LLMdInferenceSimConfig.model_name,
                        "--port",
                        str(LLMdInferenceSimConfig.port),
                        "--max-model-len",
                        str(LLMdInferenceSimConfig.max_model_len),
                        "--tokenizers-cache-dir",
                        "/data/tokenizers_cache",
                    ],
                    "ports": [{"containerPort": LLMdInferenceSimConfig.port, "protocol": "TCP"}],
                    "volumeMounts": [
                        {
                            "name": "tokenizers-cache",
                            "mountPath": "/data/tokenizers_cache",
                        }
                    ],
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                    },
                    "livenessProbe": {
                        "failureThreshold": 3,
                        "httpGet": {"path": "/health", "port": LLMdInferenceSimConfig.port, "scheme": "HTTP"},
                        "initialDelaySeconds": 15,
                        "periodSeconds": 20,
                        "timeoutSeconds": 5,
                    },
                    "readinessProbe": {
                        "failureThreshold": 3,
                        "httpGet": {"path": "/health", "port": LLMdInferenceSimConfig.port, "scheme": "HTTP"},
                        "initialDelaySeconds": 5,
                        "periodSeconds": 10,
                        "timeoutSeconds": 5,
                    },
                }
            ],
            volumes=[
                {
                    "name": "tokenizers-cache",
                    "emptyDir": {},
                }
            ],
            multi_model=False,
            supported_model_formats=[{"autoSelect": True, "name": LLMdInferenceSimConfig.name}],
            teardown=teardown_resources,
        ) as serving_runtime:
            yield serving_runtime


@pytest.fixture(scope="class")
def llm_d_inference_sim_isvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    llm_d_inference_sim_serving_runtime: ServingRuntime,
    teardown_resources: bool,
    pytestconfig: pytest.Config,
) -> Generator[InferenceService, Any, Any]:
    """Fixture for LLMdInferenceSim InferenceService."""
    if pytestconfig.option.post_upgrade:
        isvc = InferenceService(
            client=admin_client, name=LLMdInferenceSimConfig.isvc_name, namespace=model_namespace.name
        )
        yield isvc
        isvc.clean_up()
    else:
        with create_isvc(
            client=admin_client,
            name=LLMdInferenceSimConfig.isvc_name,
            namespace=model_namespace.name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            model_format=LLMdInferenceSimConfig.name,
            runtime=llm_d_inference_sim_serving_runtime.name,
            wait_for_predictor_pods=False,
            min_replicas=1,
            max_replicas=1,
            resources={
                "requests": {"cpu": "1", "memory": "1Gi"},
                "limits": {"cpu": "1", "memory": "1Gi"},
            },
            teardown=teardown_resources,
        ) as isvc:
            deployment = Deployment(
                client=admin_client,
                name=f"{isvc.name}-predictor",
                namespace=model_namespace.name,
            )
            deployment.wait_for_replicas(timeout=120)
            yield isvc


@pytest.fixture(scope="class")
def kserve_controller_manager_deployment(admin_client: DynamicClient) -> Generator[Deployment, Any, Any]:
    yield Deployment(
        client=admin_client,
        name="kserve-controller-manager",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def patched_dsc_kserve_headed(
    admin_client, kserve_controller_manager_deployment: Deployment
) -> Generator[DataScienceCluster]:
    """Configure KServe Services to work in Headed mode i.e. using the Service port instead of the Pod port"""

    def _kserve_status(dsc_resource: DataScienceCluster) -> str:
        condition = next(
            filter(lambda condition: condition["type"] == "KserveReady", dsc_resource.instance.status["conditions"]),
            None,
        )
        if condition is None:
            raise ValueError("KserveReady condition not found in DSC status")
        return condition["status"]

    @retry(wait_timeout=30, sleep=1)
    def _wait_for_kserve_upgrade(dsc_resource: DataScienceCluster):
        return _kserve_status(dsc_resource) != "True"

    @retry(wait_timeout=60, sleep=5)
    def _wait_for_kserve_ready(dsc_resource: DataScienceCluster) -> bool:
        return _kserve_status(dsc_resource) == "True"

    dsc = get_data_science_cluster(client=admin_client)
    if dsc.instance.spec.components.kserve.rawDeploymentServiceConfig != "Headed":
        with ResourceEditor(
            patches={dsc: {"spec": {"components": {"kserve": {"rawDeploymentServiceConfig": "Headed"}}}}}
        ):
            _wait_for_kserve_upgrade(dsc_resource=dsc)
            kserve_controller_manager_deployment.wait_for_replicas()
            _wait_for_kserve_ready(dsc_resource=dsc)
            yield dsc
    else:
        LOGGER.info("DSC already configured for Headed mode")
        yield dsc


def _patched_dsc_garak(admin_client: DynamicClient, components: dict) -> Generator[DataScienceCluster]:
    dsc = get_data_science_cluster(client=admin_client)
    with ResourceEditor(patches={dsc: {"spec": {"components": components}}}):
        wait_for_dsc_status_ready(dsc_resource=dsc)
        yield dsc


@pytest.fixture(scope="class")
def patched_dsc_garak(admin_client: DynamicClient) -> Generator[DataScienceCluster]:
    """Configure DSC for Garak simple mode: KServe Headed + MLflow."""
    yield from _patched_dsc_garak(
        admin_client=admin_client,
        components={
            "kserve": {"rawDeploymentServiceConfig": "Headed"},
            "mlflowoperator": {"managementState": "Managed"},
        },
    )


@pytest.fixture(scope="class")
def patched_dsc_garak_kfp(admin_client: DynamicClient) -> Generator[DataScienceCluster]:
    """Configure DSC for Garak KFP mode: KServe Headed + MLflow + AI Pipelines."""
    yield from _patched_dsc_garak(
        admin_client=admin_client,
        components={
            "kserve": {"rawDeploymentServiceConfig": "Headed"},
            "aipipelines": {"managementState": "Managed"},
            "mlflowoperator": {"managementState": "Managed"},
        },
    )
