import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

from tests.model_registry.constants import MR_INSTANCE_NAME, MR_POSTGRES_DEPLOYMENT_NAME_STR
from tests.model_registry.scc.utils import get_pod_by_deployment_name

LOGGER = get_logger(name=__name__)


@pytest.fixture()
def skip_if_not_valid_check(request) -> None:
    """
    Fixture that skips the test if deployment name is model-registry-postgres
    and db_name is not set to 'default'
    """
    deployment_name = request.node.callspec.params.get("deployment_model_registry_ns", {}).get(
        "deployment_name"
    ) or request.node.callspec.params.get("pod_model_registry_ns", {}).get("deployment_name")
    db_name = request.node.callspec.params.get("model_registry_metadata_db_resources", {}).get("db_name", "mysql")
    LOGGER.info(
        f"Deployment name:{deployment_name}, db selection: {db_name}",
    )

    if deployment_name == MR_POSTGRES_DEPLOYMENT_NAME_STR and db_name != "default":
        pytest.skip(reason=f"{MR_POSTGRES_DEPLOYMENT_NAME_STR} deployment only valid when db_name is 'default'")


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
