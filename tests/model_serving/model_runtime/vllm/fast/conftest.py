from collections.abc import Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.template import Template
from pytest import FixtureRequest
from pytest_testconfig import config as py_config

from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def serving_runtime(  # noqa: UFN001
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime]:
    """ServingRuntime from a fast template; fails if the template is absent."""
    template_name: str = request.param["template_name"]
    template = Template(
        client=admin_client,
        name=template_name,
        namespace=py_config["applications_namespace"],
    )
    if not template.exists:
        pytest.fail(
            f"Fast template {template_name} not present on cluster "
            "(fast image SHAs may match stable; see RHOAIENG-68181)"
        )

    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="vllm-fast-runtime",
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param["deployment_type"],
        runtime_image=None,
        support_tgis_open_ai_endpoints=True,
    ) as model_runtime:
        yield model_runtime
