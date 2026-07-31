import pytest
import structlog
from _pytest.fixtures import FixtureRequest

from tests.ai_hub.constants import MR_INSTANCE_NAME
from tests.ai_hub.scc.utils import get_pod_by_deployment_name
from utilities.openshift_resources.deployment import Deployment
from utilities.openshift_resources.namespace import Namespace
from utilities.openshift_resources.pod import Pod

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def model_registry_scc_namespace(model_registry_namespace: str):
    mr_annotations = Namespace(name=model_registry_namespace).instance.metadata.annotations
    return {
        "seLinuxOptions": mr_annotations.get("openshift.io/sa.scc.mcs"),
        "uid-range": mr_annotations.get("openshift.io/sa.scc.uid-range"),
    }


@pytest.fixture(scope="function")
def deployment_model_registry_ns(request: FixtureRequest, model_registry_namespace: str) -> Deployment:
    return Deployment(
        name=request.param.get("deployment_name", MR_INSTANCE_NAME),
        namespace=model_registry_namespace,
        ensure_exists=True,
    )


@pytest.fixture(scope="function")
def pod_model_registry_ns(request: FixtureRequest, model_registry_namespace: str) -> Pod:
    return get_pod_by_deployment_name(
        namespace=model_registry_namespace,
        deployment_name=request.param.get("deployment_name", MR_INSTANCE_NAME),
    )
