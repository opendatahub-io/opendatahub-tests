from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from simple_logger.logger import get_logger

from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import Timeout
from utilities.resources.evalhub import EvalHub

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def evalhub_cr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[EvalHub, Any, Any]:
    """Create an EvalHub custom resource and wait for it to be ready."""
    with EvalHub(
        client=admin_client,
        name="evalhub",
        namespace=model_namespace.name,
        wait_for_resource=True,
    ) as evalhub:
        yield evalhub


@pytest.fixture(scope="class")
def evalhub_deployment(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_cr: EvalHub,
) -> Deployment:
    """Wait for the EvalHub deployment to become available."""
    deployment = Deployment(
        client=admin_client,
        name=evalhub_cr.name,
        namespace=model_namespace.name,
    )
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
    return deployment


@pytest.fixture(scope="class")
def evalhub_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_deployment: Deployment,
) -> Route:
    """Get the Route created by the operator for the EvalHub service."""
    return Route(
        client=admin_client,
        name=evalhub_deployment.name,
        namespace=model_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def evalhub_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    """Create a CA bundle file for verifying the EvalHub route TLS certificate."""
    return create_ca_bundle_file(client=admin_client, ca_type="openshift")
