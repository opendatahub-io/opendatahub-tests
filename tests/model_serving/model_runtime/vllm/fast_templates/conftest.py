"""Fixtures for fast template overlay validation tests."""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.template import Template
from pytest_testconfig import config as py_config


@pytest.fixture(scope="class")
def fast_template_config(request: pytest.FixtureRequest) -> dict[str, str]:
    """Parametrized configuration for the current fast template test."""
    return request.param


@pytest.fixture(scope="class")
def stable_template(fast_template_config: dict[str, str], admin_client: DynamicClient) -> Template:
    """Stable (base) vLLM ServingRuntime Template from the cluster."""
    name = fast_template_config["base_template"]
    template = Template(
        client=admin_client,
        name=name,
        namespace=py_config["applications_namespace"],
    )
    if not template.exists:
        pytest.skip(f"Stable template {name} not found on cluster")
    return template


@pytest.fixture(scope="class")
def fast_template(fast_template_config: dict[str, str], admin_client: DynamicClient) -> Template:
    """Fast vLLM ServingRuntime Template overlay from the cluster."""
    name = f"{fast_template_config['base_template']}-{fast_template_config['suffix']}"
    template = Template(
        client=admin_client,
        name=name,
        namespace=py_config["applications_namespace"],
    )
    if not template.exists:
        pytest.skip(f"Fast template {name} not found (image SHAs may match stable)")
    return template


@pytest.fixture(scope="class")
def stable_embedded_runtime(stable_template: Template) -> dict[str, Any]:
    """Embedded ServingRuntime dict from the stable template."""
    objects = stable_template.instance.objects or []
    assert objects, f"Stable template {stable_template.name} has no embedded objects"
    return objects[0].to_dict()


@pytest.fixture(scope="class")
def fast_embedded_runtime(fast_template: Template) -> dict[str, Any]:
    """Embedded ServingRuntime dict from the fast template."""
    objects = fast_template.instance.objects or []
    assert objects, f"Fast template {fast_template.name} has no embedded objects"
    return objects[0].to_dict()
