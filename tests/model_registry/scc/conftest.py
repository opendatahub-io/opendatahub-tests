import pytest
import structlog
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod

from tests.model_registry.constants import MR_INSTANCE_NAME
from tests.model_registry.scc.utils import get_pod_by_deployment_name

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def model_registry_scc_namespace(admin_client: DynamicClient, model_registry_namespace: str):
    mr_annotations = Namespace(client=admin_client, name=model_registry_namespace).instance.metadata.annotations
    return {
        "seLinuxOptions": mr_annotations.get("openshift.io/sa.scc.mcs"),
        "uid-range": mr_annotations.get("openshift.io/sa.scc.uid-range"),
    }


@pytest.fixture(scope="function")
def deployment_model_registry_ns(
    request: FixtureRequest, admin_client: DynamicClient, model_registry_namespace: str
) -> Deployment:
    return Deployment(
        client=admin_client,
        name=request.param.get("deployment_name", MR_INSTANCE_NAME),
        namespace=model_registry_namespace,
        ensure_exists=True,
    )


@pytest.fixture(scope="function")
def pod_model_registry_ns(request: FixtureRequest, admin_client: DynamicClient, model_registry_namespace: str) -> Pod:
    return get_pod_by_deployment_name(
        admin_client=admin_client,
        namespace=model_registry_namespace,
        deployment_name=request.param.get("deployment_name", MR_INSTANCE_NAME),
    )
