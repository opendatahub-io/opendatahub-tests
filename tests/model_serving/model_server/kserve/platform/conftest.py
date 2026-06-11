from collections.abc import Generator
from typing import Any

import pytest
import structlog
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.kserve.platform.utils import FAST_SUFFIXES
from utilities.inference_utils import create_isvc
from utilities.operator_utils import get_rhoai_version_prefix
from utilities.resources.llm_inference_service_config import LLMInferenceServiceConfig

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def invalid_s3_models_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    serving_runtime_from_template: ServingRuntime,
    ci_s3_bucket_name: str,
    model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=request.param["name"],
        namespace=unprivileged_model_namespace.name,
        runtime=serving_runtime_from_template.name,
        storage_uri=f"s3://{ci_s3_bucket_name}/non-existing-path/",
        model_format=serving_runtime_from_template.instance.spec.supportedModelFormats[0].name,
        model_service_account=model_service_account.name,
        deployment_mode=request.param["deployment-mode"],
        wait=False,
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc


@pytest.fixture
def updated_s3_models_inference_service(
    invalid_s3_models_inference_service: InferenceService, s3_models_storage_uri: str
) -> Generator[InferenceService, Any, Any]:
    with ResourceEditor(
        patches={
            invalid_s3_models_inference_service: {
                "spec": {
                    "predictor": {"model": {"storageUri": s3_models_storage_uri}},
                }
            }
        }
    ):
        yield invalid_s3_models_inference_service


@pytest.fixture(scope="module")
def all_llm_configs(admin_client: DynamicClient) -> list[LLMInferenceServiceConfig]:
    """All LLMInferenceServiceConfig resources in the applications namespace."""
    namespace = py_config["applications_namespace"]
    configs = list(LLMInferenceServiceConfig.get(client=admin_client, namespace=namespace))
    LOGGER.info(f"Found {len(configs)} LLMInferenceServiceConfig resources in {namespace}")
    for config in configs:
        LOGGER.info(f"  - {config.name}")
    return configs


@pytest.fixture(scope="module")
def fast_configs(all_llm_configs: list[LLMInferenceServiceConfig]) -> list[LLMInferenceServiceConfig]:
    """Fast overlay configs (-fast-1, -fast-2), skips if none exist."""
    configs = [config for config in all_llm_configs if config.name.endswith(FAST_SUFFIXES)]
    if not configs:
        pytest.skip("No fast LLMInferenceServiceConfig resources found (fast image SHAs may match stable)")
    LOGGER.info(f"Found {len(configs)} fast configs: {[config.name for config in configs]}")
    return configs


@pytest.fixture(scope="module")
def stable_configs_by_name(
    all_llm_configs: list[LLMInferenceServiceConfig],
) -> dict[str, LLMInferenceServiceConfig]:
    """Map of stable config name to resource (non-fast configs)."""
    return {config.name: config for config in all_llm_configs if not config.name.endswith(FAST_SUFFIXES)}


@pytest.fixture(scope="module")
def version_prefix(admin_client: DynamicClient) -> str:
    """RHOAI version prefix for config names."""
    return get_rhoai_version_prefix(client=admin_client, namespace=py_config["operator_namespace"])
