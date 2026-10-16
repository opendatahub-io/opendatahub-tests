from collections.abc import Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret

from tests.model_serving.model_server.utils import create_oci_connection_secret


@pytest.fixture(scope="class")
def oci_connection_secret(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[Secret]:
    """`oci`-typed ConnectionsAPI connection Secret for the reused MNIST OCI modelcar image.

    The referenced image (`ModelCarImage.MNIST_8_1`) is a public quay.io modelcar, so an empty
    docker config is sufficient — the webhook still injects `imagePullSecrets` referencing this
    Secret regardless of its credential content.
    """
    with create_oci_connection_secret(
        client=admin_client,
        name="kserve-storage-oci-connection-secret",
        namespace=unprivileged_model_namespace.name,
    ) as secret:
        yield secret
