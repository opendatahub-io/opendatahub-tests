from collections.abc import Generator
from typing import Any

import pytest
from _pytest.config import Config
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from pytest_testconfig import config as py_config

from tests.model_registry.constants import (
    DB_RESOURCE_NAME,
    MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
    MODEL_REGISTRY_DB_SECRET_STR_DATA,
    MR_INSTANCE_NAME,
)
from tests.model_registry.model_registry.negative_tests.utils import (
    create_mysql_credentials_file,
    execute_mysql_command,
)
from tests.model_registry.utils import get_model_registry_db_label_dict, get_model_registry_deployment_template_dict
from utilities.constants import MODEL_REGISTRY_CUSTOM_NAMESPACE
from utilities.general import wait_for_pods_by_labels
from utilities.infra import create_ns

DB_RESOURCES_NAME_NEGATIVE = "db-model-registry-negative"


@pytest.fixture(scope="class")
def model_registry_namespace_for_negative_tests(
    dsc_resource: DataScienceCluster,
    admin_client: DynamicClient,
    pytestconfig: Config,
) -> Generator[Namespace, Any, Any]:
    namespace_name = MODEL_REGISTRY_CUSTOM_NAMESPACE
    if pytestconfig.option.custom_namespace:
        namespace_name = "rhoai-model-registries"
    with create_ns(
        name=namespace_name,
        admin_client=admin_client,
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
        client=admin_client,
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
            secret_name=model_registry_db_secret_negative_test.name,
            resource_name=DB_RESOURCES_NAME_NEGATIVE,
            db_backend="mysql",
        ),
        wait_for_resource=True,
    ) as mr_db_deployment:
        mr_db_deployment.wait_for_replicas(deployed=True)
        yield mr_db_deployment


@pytest.fixture()
def set_mr_db_dirty(model_registry_db_instance_pod: Pod) -> int:
    """Set the model registry database dirty and return the latest migration version"""
    create_mysql_credentials_file(model_registry_db_instance_pod=model_registry_db_instance_pod)
    output = execute_mysql_command(
        sql_query="SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;",
        model_registry_db_instance_pod=model_registry_db_instance_pod,
    )
    latest_migration_version = int(output.strip().split()[1])
    execute_mysql_command(
        sql_query=f"UPDATE schema_migrations SET dirty = 1 WHERE version = {latest_migration_version};",
        model_registry_db_instance_pod=model_registry_db_instance_pod,
    )
    return latest_migration_version


@pytest.fixture()
def model_registry_db_instance_pod(admin_client: DynamicClient) -> Generator[Pod, Any, Any]:
    """Get the model registry instance pod."""
    yield wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=py_config["model_registry_namespace"],
        label_selector=f"name={DB_RESOURCE_NAME}",
        expected_num_pods=1,
    )[0]


@pytest.fixture()
def delete_mr_deployment(admin_client: DynamicClient) -> None:
    """Delete the model registry deployment"""
    mr_deployment = Deployment(
        client=admin_client, name=MR_INSTANCE_NAME, namespace=py_config["model_registry_namespace"], ensure_exists=True
    )
    mr_deployment.delete(wait=True)
