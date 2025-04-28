import pytest
from typing import Generator, Any
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.deployment import Deployment

from pytest import FixtureRequest
from kubernetes.dynamic import DynamicClient


from tests.model_registry.constants import (
    MODEL_REGISTRY_DB_SECRET_STR_DATA,
    MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
)
from tests.model_registry.negative_tests.constants import CUSTOM_NEGATIVE_NS
from tests.model_registry.utils import get_model_registry_deployment_template_dict, get_model_registry_db_label_dict
from utilities.infra import create_ns

DB_RESOURCES_NAME_NEGATIVE = "db-model-registry-negative"


@pytest.fixture(scope="class")
def model_registry_namespace_for_negative_tests(
    request: FixtureRequest, admin_client: DynamicClient
) -> Generator[Namespace, Any, Any]:
    with create_ns(
        name=request.param.get("namespace_name", CUSTOM_NEGATIVE_NS),
        client=admin_client,
    ) as ns:
        yield ns


@pytest.fixture(scope="class")
def model_registry_db_service_for_negative_tests(
    admin_client: DynamicClient, model_registry_namespace_for_negative_tests: Namespace
) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        name=DB_RESOURCES_NAME_NEGATIVE,
        namespace=model_registry_namespace_for_negative_tests.name,
        ports=[
            {
                "name": "mysql",
                "nodePort": 0,
                "port": 3306,
                "protocol": "TCP",
                "appProtocol": "tcp",
                "targetPort": 3306,
            }
        ],
        selector={
            "name": DB_RESOURCES_NAME_NEGATIVE,
        },
        label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME_NEGATIVE),
        annotations={
            "template.openshift.io/expose-uri": r"mysql://{.spec.clusterIP}:{.spec.ports[?(.name==\mysql\)].port}",
        },
    ) as mr_db_service:
        yield mr_db_service


@pytest.fixture(scope="class")
def model_registry_db_pvc_for_negative_tests(
    admin_client: DynamicClient,
    model_registry_namespace_for_negative_tests: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    with PersistentVolumeClaim(
        accessmodes="ReadWriteOnce",
        name=DB_RESOURCES_NAME_NEGATIVE,
        namespace=model_registry_namespace_for_negative_tests.name,
        client=admin_client,
        size="5Gi",
        label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME_NEGATIVE),
    ) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def model_registry_db_secret_negative_test(
    admin_client: DynamicClient,
    model_registry_namespace_for_negative_tests: Namespace,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        name=DB_RESOURCES_NAME_NEGATIVE,
        namespace=model_registry_namespace_for_negative_tests.name,
        string_data=MODEL_REGISTRY_DB_SECRET_STR_DATA,
        label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME_NEGATIVE),
        annotations=MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
    ) as mr_db_secret:
        yield mr_db_secret


@pytest.fixture(scope="class")
def model_registry_db_deployment_negative_test(
    admin_client: DynamicClient,
    model_registry_namespace_for_negative_tests: Namespace,
    model_registry_db_secret_negative_test: Secret,
    model_registry_db_pvc_for_negative_tests: PersistentVolumeClaim,
    model_registry_db_service_for_negative_tests: Service,
) -> Generator[Deployment, Any, Any]:
    with Deployment(
        name=DB_RESOURCES_NAME_NEGATIVE,
        namespace=model_registry_namespace_for_negative_tests.name,
        annotations={
            "template.alpha.openshift.io/wait-for-ready": "true",
        },
        label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME_NEGATIVE),
        replicas=1,
        revision_history_limit=0,
        selector={"matchLabels": {"name": DB_RESOURCES_NAME_NEGATIVE}},
        strategy={"type": "Recreate"},
        template=get_model_registry_deployment_template_dict(
            secret_name=model_registry_db_secret_negative_test.name, resource_name=DB_RESOURCES_NAME_NEGATIVE
        ),
        wait_for_resource=True,
    ) as mr_db_deployment:
        mr_db_deployment.wait_for_replicas(deployed=True)
        yield mr_db_deployment
