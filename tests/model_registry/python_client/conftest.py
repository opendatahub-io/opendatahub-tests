from typing import Any

import pytest

from ocp_resources.deployment import Deployment
from tests.model_registry.constants import MR_INSTANCE_NAME


@pytest.fixture(scope="class")
def model_registry_deployment_containers(model_registry_namespace: str) -> dict[str, Any]:
    return Deployment(
        name=MR_INSTANCE_NAME, namespace=model_registry_namespace, ensure_exists=True
    ).instance.spec.template.spec.containers
