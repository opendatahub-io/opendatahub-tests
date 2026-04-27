from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod

from tests.ogx.constants import OGX_CORE_POD_FILTER
from utilities.general import wait_for_pods_by_labels


@pytest.fixture(scope="class")
def ogx_server_pods(admin_client: DynamicClient, unprivileged_model_namespace: Namespace) -> Generator[Pod, Any, Any]:
    """Returns the OgxServer pods running in the namespace."""
    yield wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=unprivileged_model_namespace.name,
        label_selector=OGX_CORE_POD_FILTER,
        expected_num_pods=1,
    )[0]
