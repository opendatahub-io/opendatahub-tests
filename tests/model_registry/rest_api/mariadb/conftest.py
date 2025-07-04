from typing import Any, Generator
from kubernetes.dynamic import DynamicClient
import pytest

from ocp_resources.maria_db import MariaDB
from ocp_resources.mariadb_operator import MariadbOperator
from utilities.constants import OPENSHIFT_OPERATORS, MARIADB
from utilities.general import generate_random_name

from tests.model_registry.constants import (
    OAUTH_PROXY_CONFIG_DICT,
    MODEL_REGISTRY_STANDARD_LABELS,
    MR_INSTANCE_NAME,
)
from ocp_resources.secret import Secret
from simple_logger.logger import get_logger
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from utilities.mariadb_utils import wait_for_mariadb_pods

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def mariadb_str() -> str:
    return generate_random_name(prefix=MARIADB, length=4)


@pytest.fixture(scope="class")
def mariadb_dict(mariadb_str: str) -> dict[str, Any]:
    return {
        "connection": {"secretName": f"{mariadb_str}conn", "secretTemplate": {"key": "dsn"}},
        "database": f"{mariadb_str}db".replace("-", "_"),
        "primary_connection": {"secretName": f"{mariadb_str}-conn-primary", "secretTemplate": {"key": "dsn"}},
        "primary_service": {"type": "ClusterIP"},
        "port": 3306,
        "root_password_secret_key_ref": {"generate": True, "key": "password", "name": f"{mariadb_str}-root"},
        "galera": {"enabled": False},
        "metrics": {"enabled": False},
        "replicas": 1,
        "my_cnf": "[mariadb]\nbind-address=*\ndefault_storage_engine=InnoDB"
        "\nbinlog_format=row\ninnodb_autoinc_lock_mode=2"
        "\ninnodb_buffer_pool_size=1024M\nmax_allowed_packet=256M\n",  # pragma: allowlist secret
        "secondary_connection": {"secretName": f"{mariadb_str}-conn-secondary", "secretTemplate": {"key": "dsn"}},
        "secondary_service": {"type": "ClusterIP"},
        "service": {"type": "ClusterIP"},
        "username": f"{mariadb_str}user",
        "password_secret_key_ref": {"generate": True, "key": "password", "name": f"{mariadb_str}-password"},
        "storage": {"size": "1Gi"},
        "update_strategy": {"type": "ReplicasFirstPrimaryLast"},
    }


@pytest.fixture(scope="class")
def deployed_mariadb(
    admin_client: DynamicClient,
    mariadb_operator_cr: MariadbOperator,
    mariadb_str: str,
    mariadb_dict: dict[str, Any],
) -> Generator[MariaDB, Any, Any]:
    with MariaDB(name=f"{mariadb_str}", namespace=OPENSHIFT_OPERATORS, **mariadb_dict) as mariadb:
        wait_for_mariadb_pods(client=admin_client, mariadb=mariadb)
        yield mariadb
    for secret_name in [
        f"{mariadb_str}-root",
        f"{mariadb_str}-password",
        "mariadb-operator-webhook-ca",
        "mariadb-operator-webhook-cert",
    ]:
        secret = Secret(name=secret_name, namespace=OPENSHIFT_OPERATORS)
        if secret.exists:
            secret.clean_up()


@pytest.fixture(scope="class")
def mariadb_secret(deployed_mariadb: MariaDB, model_registry_namespace: str) -> Generator[Secret, Any, Any]:
    mariadb_spec = deployed_mariadb.instance.spec
    secret_name = mariadb_spec.passwordSecretKeyRef.name
    secret_data = Secret(name=secret_name, namespace=OPENSHIFT_OPERATORS, ensure_exists=True).instance.data[
        mariadb_spec.passwordSecretKeyRef.key
    ]
    with Secret(
        name=secret_name,
        namespace=model_registry_namespace,
        data_dict={mariadb_spec.passwordSecretKeyRef.key: secret_data},
    ) as mr_secret:
        yield mr_secret


@pytest.fixture(scope="class")
def model_registry_with_mariadb(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mariadb_mysql_config: dict[str, Any],
    is_model_registry_oauth: bool,
) -> Generator[ModelRegistry, Any, Any]:
    with ModelRegistry(
        name=MR_INSTANCE_NAME,
        namespace=model_registry_namespace,
        label=MODEL_REGISTRY_STANDARD_LABELS,
        grpc={},
        rest={},
        oauth_proxy=OAUTH_PROXY_CONFIG_DICT,
        mysql=mariadb_mysql_config,
        wait_for_resource=True,
        teardown=True,
    ) as mr:
        mr.wait_for_condition(condition="Available", status="True")
        mr.wait_for_condition(condition="OAuthProxyAvailable", status="True")
        yield mr


@pytest.fixture(scope="class")
def mariadb_mysql_config(deployed_mariadb: MariaDB, mariadb_secret: Secret) -> dict[str, Any]:
    mariadb_spec = deployed_mariadb.instance.spec
    return {
        "host": f"{deployed_mariadb.name}.{deployed_mariadb.namespace}.svc.cluster.local",
        "database": mariadb_spec.database,
        "passwordSecret": {"key": mariadb_spec.passwordSecretKeyRef.key, "name": mariadb_secret.name},
        "port": 3306,
        "skipDBCreation": False,
        "username": mariadb_spec.username,
    }
