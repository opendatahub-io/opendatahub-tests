"""Fixtures for vLLM-Omni fast template tests."""

from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.template import Template
from pytest import FixtureRequest
from pytest_testconfig import config as py_config

from utilities.constants import KServeDeploymentType
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def vllm_omni_serving_runtime(  # noqa: UFN001 — intentionally shadows parent conftest fixture
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    vllm_omni_runtime_image: str | None,
) -> Generator[ServingRuntime, Any, Any]:
    """ServingRuntime from a fast template; skips if the template is absent.

    Shadows the parent conftest's vllm_omni_serving_runtime so that
    vllm_omni_inference_service picks up the fast runtime automatically.
    """
    template_name: str = request.param["template_name"]
    template = Template(
        client=admin_client,
        name=template_name,
        namespace=py_config["applications_namespace"],
    )
    if not template.exists:
        pytest.fail(
            f"Fast template '{template_name}' not found in namespace '{py_config['applications_namespace']}'. "
            "This should have been deselected by pytest_collection_modifyitems."
        )
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="vllm-omni-fast-runtime",
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param.get("deployment_type", KServeDeploymentType.STANDARD),
        runtime_image=vllm_omni_runtime_image,
    ) as runtime:
        yield runtime
