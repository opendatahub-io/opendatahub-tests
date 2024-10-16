import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.maria_db import MariaDB
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_utilities.operators import install_operator, uninstall_operator

from tests.trustyai.utils import wait_for_mariadb_operator_deployments, wait_for_mariadb_pods

OPENSHIFT_OPERATORS = "openshift-operators"

TIMEOUT_10_MIN = 60 * 10

MINIO: str = "minio"
OPENDATAHUB_IO: str = "opendatahub.io"
MARIADB: str = "mariadb"
QUARKUS: str = "quarkus"


@pytest.fixture(scope="class")
def minio_pod(admin_client: DynamicClient, model_namespace: Namespace) -> Pod:
    with Pod(
        client=admin_client,
        name=MINIO,
        namespace=model_namespace.name,
        containers=[
            {
                "args": [
                    "server",
                    "/data1",
                ],
                "env": [
                    {
                        "name": "MINIO_ACCESS_KEY",
                        "value": "THEACCESSKEY",
                    },
                    {
                        "name": "MINIO_SECRET_KEY",
                        "value": "THESECRETKEY",
                    },
                ],
                "image": "quay.io/trustyai/modelmesh-minio-examples@"
                "sha256:e8360ec33837b347c76d2ea45cd4fea0b40209f77520181b15e534b101b1f323",
                "name": MINIO,
            }
        ],
        label={"app": "minio", "maistra.io/expose-route": "true"},
        annotations={"sidecar.istio.io/inject": "true"},
    ) as minio_pod:
        yield minio_pod


@pytest.fixture(scope="class")
def minio_service(admin_client: DynamicClient, model_namespace: Namespace) -> Service:
    with Service(
        client=admin_client,
        name=MINIO,
        namespace=model_namespace.name,
        ports=[
            {
                "name": "minio-client-port",
                "port": 9000,
                "protocol": "TCP",
                "targetPort": 9000,
            }
        ],
        selector={
            "app": MINIO,
        },
    ) as minio_service:
        yield minio_service


@pytest.fixture(scope="class")
def minio_data_connection(
    admin_client: DynamicClient, model_namespace: Namespace, minio_pod: Pod, minio_service: Service
) -> Secret:
    with Secret(
        client=admin_client,
        name="aws-connection-minio-data-connection",
        namespace=model_namespace.name,
        data_dict={
            "AWS_ACCESS_KEY_ID": "VEhFQUNDRVNTS0VZ",
            "AWS_DEFAULT_REGION": "dXMtc291dGg=",
            "AWS_S3_BUCKET": "bW9kZWxtZXNoLWV4YW1wbGUtbW9kZWxz",
            "AWS_S3_ENDPOINT": "aHR0cDovL21pbmlvOjkwMDA=",
            "AWS_SECRET_ACCESS_KEY": "VEhFU0VDUkVUS0VZ",  # pragma: allowlist secret
        },
        label={
            f"{OPENDATAHUB_IO}/dashboard": "true",
            f"{OPENDATAHUB_IO}/managed": "true",
        },
        annotations={
            f"{OPENDATAHUB_IO}/connection-type": "s3",
            "openshift.io/display-name": "Minio Data Connection",
        },
    ) as minio_secret:
        yield minio_secret


@pytest.fixture(scope="class")
def db_credentials(admin_client: DynamicClient, model_namespace: Namespace) -> Secret:
    with Secret(
        client=admin_client,
        name="db-credentials",
        namespace=model_namespace.name,
        string_data={
            "databaseKind": MARIADB,
            "databaseName": "trustyai_database",
            "databaseUsername": QUARKUS,
            "databasePassword": QUARKUS,
            "databaseService": MARIADB,
            "databasePort": "3306",
            "databaseGeneration": "update",
        },
    ) as db_credentials:
        yield db_credentials


@pytest.fixture(scope="session")
def mariadb_operator(admin_client: DynamicClient) -> None:
    name = "mariadb-operator"
    namespace = "openshift-operators"
    install_operator(
        admin_client=admin_client,
        target_namespaces=[namespace],
        name=name,
        channel="alpha",
        source="community-operators",
        operator_namespace=namespace,
        timeout=TIMEOUT_10_MIN,
        install_plan_approval="Manual",
    )
    yield
    uninstall_operator(admin_client=admin_client, name=name, operator_namespace=namespace, clean_up_namespace=False)


@pytest.fixture(scope="session")
def mariadb_operator_cr(admin_client: DynamicClient) -> MariadbOperator:
    with MariadbOperator(
        client=admin_client,
        name=f"{MARIADB}-operator",
        namespace=OPENSHIFT_OPERATORS,
        cert_controller={
            "enabled": True,
            "caValidity": "35064h",
            "certValidity": "8766h",
            "ha": {"enabled": False, "replicas": 3},
            "image": {"pullPolicy": "IfNotPresent", "repository": "ghcr.io/mariadb-operator/mariadb-operator"},
            "lookaheadValidity": "2160h",
            "requeueDuration": "5m",
            "serviceAccount": {"automount": True, "enabled": True},
            "serviceMonitor": {"enabled": True, "interval": "30s", "scrapeTimeout": "25s"},
        },
        cluster_name="cluster.local",
        image={"pullPolicy": "IfNotPresent", "repository": "ghcr.io/mariadb-operator/mariadb-operator"},
        log_level="INFO",
        metrics={"enabled": False, "serviceMonitor": {"enabled": True, "interval": "30s", "scrapeTimeout": "25s"}},
        rbac={"enabled": True},
        service_account={"automount": True, "enabled": True},
        webhook={
            "cert": {
                "caPath": "/tmp/k8s-webhook-server/certificate-authority",
                "path": "/tmp/k8s-webhook-server/serving-certs",
            },
            "ha": {"enabled": False, "replicas": 3},
            "hostNetwork": False,
            "image": {"pullPolicy": "IfNotPresent", "repository": "ghcr.io/mariadb-operator/mariadb-operator"},
            "port": 10250,
            "serviceAccount": {"automount": True, "enabled": True},
            "serviceMonitor": {"enabled": True, "interval": "30s", "scrapeTimeout": "25s"},
        },
    ) as mariadb_operator:
        mariadb_operator.wait_for_condition(
            condition="Deployed", status=mariadb_operator.Condition.Status.TRUE, timeout=10 * 60
        )
        wait_for_mariadb_operator_deployments(mariadb_operator=mariadb_operator)
        yield mariadb_operator


@pytest.fixture(scope="class")
def mariadb(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    db_credentials: Secret,
    mariadb_operator_cr: MariadbOperator,
) -> MariaDB:
    with MariaDB(
        client=admin_client,
        name=MARIADB,
        namespace=model_namespace.name,
        connection={"secretName": "mariadb-conn", "secretTemplate": {"key": "dsn"}}, # pragma: allowlist secret
        database="trustyai_database",
        galera={"enabled": False},
        metrics={
            "enabled": False,
            "passwordSecretKeyRef": {
                "generate": True,
                "key": "password",
                "name": "mariadb-metrics",
            },
        },
        my_cnf="""
            [mariadb]
            bind-address=*
            default_storage_engine=InnoDB
            binlog_format=row
            innodb_autoinc_lock_mode=2
            innodb_buffer_pool_size=1024M
            max_allowed_packet=256M
            """,
        password_secret_key_ref={
            "generate": False,
            "key": "databasePassword",
            "name": "db-credentials",
        },
        primary_connection={
            "secretName": "mariadb-conn-primary",  # pragma: allowlist secret
            "secretTemplate": {"key": "dsn"},
        },
        primary_service={"type": "ClusterIP"},
        replicas=1,
        root_password_secret_key_ref={
            "generate": False,
            "key": "databasePassword",
            "name": "db-credentials",
        },
        secondary_connection={
            "secretName": "mariadb-conn-secondary",  # pragma: allowlist secret
            "secretTemplate": {"key": "dsn"},
        },
        secondary_service={"type": "ClusterIP"},
        service={"type": "ClusterIP"},
        storage={"size": "1Gi"},
        update_strategy={"type": "ReplicasFirstPrimaryLast"},
        username="quarkus",
    ) as mariadb:
        wait_for_mariadb_pods(mariadb=mariadb)
        yield mariadb
