import json
import secrets
from collections.abc import Generator
from typing import Any

import pytest
import yaml
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.maria_db import MariaDB
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.trustyai_service import TrustyAIService
from pytest_testconfig import py_config

from tests.ai_safety.trustyai_service.constants import (
    GAUSSIAN_CREDIT_MODEL,
    GAUSSIAN_CREDIT_MODEL_RESOURCES,
    GAUSSIAN_CREDIT_MODEL_STORAGE_PATH,
    ISVC_GETTER,
    KSERVE_MLSERVER,
    KSERVE_MLSERVER_ANNOTATIONS,
    KSERVE_MLSERVER_CONTAINERS,
    KSERVE_MLSERVER_SUPPORTED_MODEL_FORMATS,
    TAI_DATA_CONFIG,
    TAI_DB_STORAGE_CONFIG,
    TAI_METRICS_CONFIG,
    TAI_PVC_STORAGE_CONFIG,
    XGBOOST,
)
from tests.ai_safety.trustyai_service.trustyai_service_utils import (
    wait_for_isvc_deployment_registered_by_trustyai_service,
)
from tests.ai_safety.trustyai_service.utils import (
    create_isvc_getter_role,
    create_isvc_getter_role_binding,
    create_isvc_getter_service_account,
    create_isvc_getter_token_secret,
    create_trustyai_service,
    wait_for_mariadb_pods,
)
from utilities.constants import (
    MARIADB,
    OPENSHIFT_OPERATORS,
    TRUSTYAI_SERVICE_NAME,
    Annotations,
    KServeDeploymentType,
    Labels,
)
from utilities.inference_utils import create_isvc
from utilities.infra import create_inference_token, get_kserve_storage_initialize_image, update_configmap_data
from utilities.logger import RedactedString
from utilities.operator_utils import get_cluster_service_version

DB_CREDENTIALS_SECRET_NAME: str = "db-credentials"
DB_NAME: str = "trustyai_db"
DB_USERNAME: str = "trustyai_user"
DB_PASSWORD: str = "trustyai_password"


@pytest.fixture(scope="class")
def trustyai_service(
    request: FixtureRequest,
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    cluster_monitoring_config: ConfigMap,
    user_workload_monitoring_config: ConfigMap,
    teardown_resources: bool,
) -> Generator[TrustyAIService, Any, Any]:
    """Provides a TrustyAIService instance for testing.

    In post-upgrade mode, references the existing TrustyAIService created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new TrustyAIService with either
    PVC or DB storage based on the test parametrization, and manages cleanup via teardown_resources.
    """
    tais_kwargs = {"client": admin_client, "namespace": model_namespace.name, "name": TRUSTYAI_SERVICE_NAME}

    if pytestconfig.option.post_upgrade:
        trustyai_service = TrustyAIService(**tais_kwargs)
        yield trustyai_service
        trustyai_service.clean_up()
    else:
        if request.param["storage"] == "pvc":
            tais_kwargs["storage"] = TAI_PVC_STORAGE_CONFIG
            tais_kwargs["data"] = TAI_DATA_CONFIG
        elif request.param["storage"] == "db":
            request.getfixturevalue("mariadb")
            request.getfixturevalue("trustyai_db_ca_secret")
            tais_kwargs["storage"] = TAI_DB_STORAGE_CONFIG
        else:
            raise ValueError("TrustyAI storage can only be 'pvc' or 'db'")

        with create_trustyai_service(
            **tais_kwargs,
            metrics=TAI_METRICS_CONFIG,
            wait_for_replicas=True,
            teardown=teardown_resources,
        ) as trustyai_service:
            yield trustyai_service


@pytest.fixture(scope="session")
def kserve_raw_config(admin_client: DynamicClient) -> Generator[ConfigMap, Any, Any]:
    """Configure KServe for KServeRaw support by adding logger configuration."""

    storage_initializer_image = get_kserve_storage_initialize_image(client=admin_client)
    logger_config = {
        "image": storage_initializer_image,
        "memoryRequest": "100Mi",
        "memoryLimit": "1Gi",
        "cpuRequest": "100m",
        "cpuLimit": "1",
        "defaultUrl": "http://default-broker",
        "caBundle": "kserve-logger-ca-bundle",
        "caCertFile": "service-ca.crt",
        "tlsSkipVerify": False,
    }

    data = {"logger": json.dumps(obj=logger_config)}

    cm = ConfigMap(
        client=admin_client,
        name="inferenceservice-config",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )

    with ResourceEditor(
        patches={
            cm: {
                "metadata": {"annotations": {Annotations.OpenDataHubIo.MANAGED: "false"}},
                "data": data,
            }
        }
    ):
        yield cm


@pytest.fixture(scope="class")
def kserve_logger_ca_bundle(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[ConfigMap, Any, Any]:
    """Create CA certificate ConfigMap required for KServeRaw logger."""
    with ConfigMap(
        client=admin_client,
        name="kserve-logger-ca-bundle",
        namespace=model_namespace.name,
        annotations={"service.beta.openshift.io/inject-cabundle": "true"},
        data={},
    ) as ca_bundle:
        yield ca_bundle


@pytest.fixture(scope="session")
def user_workload_monitoring_config(admin_client: DynamicClient) -> Generator[ConfigMap, Any, Any]:
    data = {"config.yaml": yaml.dump({"prometheus": {"logLevel": "debug", "retention": "15d"}})}
    with update_configmap_data(
        client=admin_client,
        name="user-workload-monitoring-config",
        namespace="openshift-user-workload-monitoring",
        data=data,
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def db_credentials_secret(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """Provides database credentials secret for MariaDB connection.

    In post-upgrade mode, references the existing secret created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new secret with MariaDB
    connection details and manages cleanup via teardown_resources.
    """
    if pytestconfig.option.post_upgrade:
        secret = Secret(
            client=admin_client,
            name=DB_CREDENTIALS_SECRET_NAME,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield secret
        secret.clean_up()
    else:
        db_password = secrets.token_urlsafe(nbytes=24)
        with Secret(
            client=admin_client,
            name=DB_CREDENTIALS_SECRET_NAME,
            namespace=model_namespace.name,
            string_data={
                "databaseKind": MARIADB,
                "databaseName": DB_NAME,
                "databaseUsername": DB_USERNAME,
                "databasePassword": db_password,
                "databaseService": MARIADB,
                "databasePort": "3306",
                "databaseGeneration": "update",
            },
            teardown=teardown_resources,
        ) as db_credentials:
            yield db_credentials


def _generate_mariadb_tls_certs(namespace_name: str) -> tuple[str, str, str]:
    """Generate self-signed TLS certificates for MariaDB using cryptography library.

    Returns:
        tuple: (ca_cert_pem, server_cert_pem, server_key_pem)
    """
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    import datetime

    # Generate CA private key
    ca_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )

    # Generate CA certificate
    ca_subject = ca_issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, f"mariadb-ca-{namespace_name}"),
        ]
    )
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_issuer)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256(), backend=default_backend())
    )

    # Generate server private key
    server_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )

    # Generate server certificate
    server_subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, f"mariadb.{namespace_name}.svc.cluster.local"),
        ]
    )
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_subject)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("mariadb"),
                    x509.DNSName(f"mariadb.{namespace_name}.svc"),
                    x509.DNSName(f"mariadb.{namespace_name}.svc.cluster.local"),
                ]
            ),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256(), backend=default_backend())
    )

    # Serialize to PEM format
    ca_cert_pem = ca_cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
    server_cert_pem = server_cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
    server_key_pem = server_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    return ca_cert_pem, server_cert_pem, server_key_pem


@pytest.fixture(scope="class")
def mariadb(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    db_credentials_secret: Secret,
    teardown_resources: bool,
) -> Generator[Deployment, Any, Any]:
    """Provides a MariaDB instance using direct Deployment with TLS enabled.

    Uses Red Hat MariaDB image deployed via Deployment to avoid Docker Hub rate limits.
    Generates self-signed TLS certificates to match operator behavior.
    """
    # Red Hat MariaDB image - using tag for multi-arch support (amd64, arm64, s390x, ppc64le)
    MARIA_DB_IMAGE = "registry.redhat.io/rhel9/mariadb-1011:latest"

    if pytestconfig.option.post_upgrade:
        deployment = Deployment(
            client=admin_client,
            name="mariadb",
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield deployment
        deployment.clean_up()
    else:
        # Generate TLS certificates for MariaDB
        ca_cert, server_cert, server_key = _generate_mariadb_tls_certs(model_namespace.name)

        # Create secrets for TLS certificates
        with Secret(
            client=admin_client,
            name="mariadb-ca",
            namespace=model_namespace.name,
            string_data={"ca.crt": ca_cert},
            teardown=teardown_resources,
        ) as ca_secret, Secret(
            client=admin_client,
            name="mariadb-server-cert",
            namespace=model_namespace.name,
            string_data={"tls.crt": server_cert},
            teardown=teardown_resources,
        ) as server_cert_secret, Secret(
            client=admin_client,
            name="mariadb-server-key",
            namespace=model_namespace.name,
            string_data={"tls.key": server_key},
            teardown=teardown_resources,
        ) as server_key_secret:
            # Create PVC for MariaDB data
            with PersistentVolumeClaim(
                accessmodes="ReadWriteOnce",
                name="mariadb",
                namespace=model_namespace.name,
                client=admin_client,
                size="1Gi",
                teardown=teardown_resources,
            ) as pvc:
                # Create Service for MariaDB
                mariadb_service_dict = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": "mariadb",
                    "namespace": model_namespace.name,
                },
                "spec": {
                    "selector": {"name": "mariadb"},
                    "ports": [{
                        "port": 3306,
                        "targetPort": 3306,
                        "name": "mysql",
                        "protocol": "TCP",
                    }],
                    }
                }

                with Service(kind_dict=mariadb_service_dict, teardown=teardown_resources) as svc:
                    # Create Deployment (following ai_hub pattern)
                    deployment_template = {
                    "metadata": {
                        "labels": {
                            "name": "mariadb",
                            "app": "mariadb",
                            "component": "database",
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "mariadb",
                            "image": MARIA_DB_IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "env": [
                                {
                                    "name": "MYSQL_USER",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": DB_CREDENTIALS_SECRET_NAME,
                                            "key": "databaseUsername",
                                        }
                                    },
                                },
                                {
                                    "name": "MYSQL_PASSWORD",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": DB_CREDENTIALS_SECRET_NAME,
                                            "key": "databasePassword",
                                        }
                                    },
                                },
                                {
                                    "name": "MYSQL_ROOT_PASSWORD",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": DB_CREDENTIALS_SECRET_NAME,
                                            "key": "databasePassword",
                                        }
                                    },
                                },
                                {
                                    "name": "MYSQL_DATABASE",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": DB_CREDENTIALS_SECRET_NAME,
                                            "key": "databaseName",
                                        }
                                    },
                                },
                                {
                                    "name": "MARIADB_ROOT_PASSWORD",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": DB_CREDENTIALS_SECRET_NAME,
                                            "key": "databasePassword",
                                        }
                                    },
                                },
                            ],
                            "ports": [{"containerPort": 3306, "protocol": "TCP"}],
                            "livenessProbe": {
                                "exec": {
                                    "command": [
                                        "/bin/bash",
                                        "-c",
                                        "mysqladmin -u${MYSQL_USER} -p${MYSQL_ROOT_PASSWORD} ping",
                                    ]
                                },
                                "initialDelaySeconds": 15,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                            },
                            "readinessProbe": {
                                "exec": {
                                    "command": [
                                        "/bin/bash",
                                        "-c",
                                        'mysql -D ${MYSQL_DATABASE} -u${MYSQL_USER} -p${MYSQL_ROOT_PASSWORD} -e "SELECT 1"',
                                    ]
                                },
                                "initialDelaySeconds": 10,
                                "timeoutSeconds": 5,
                            },
                            "command": [
                                "run-mysqld",
                                "--ssl-ca=/etc/mysql/certs/ca.crt",
                                "--ssl-cert=/etc/mysql/certs/tls.crt",
                                "--ssl-key=/etc/mysql/certs/tls.key",
                                "--require-secure-transport=ON"
                            ],
                            "securityContext": {"capabilities": {}, "privileged": False},
                            "terminationMessagePath": "/dev/termination-log",
                            "volumeMounts": [
                                {
                                    "mountPath": "/var/lib/mysql",
                                    "name": "mariadb-data",
                                },
                                {
                                    "mountPath": "/etc/mysql/certs/ca.crt",
                                    "name": "ca-cert",
                                    "subPath": "ca.crt",
                                    "readOnly": True,
                                },
                                {
                                    "mountPath": "/etc/mysql/certs/tls.crt",
                                    "name": "server-cert",
                                    "subPath": "tls.crt",
                                    "readOnly": True,
                                },
                                {
                                    "mountPath": "/etc/mysql/certs/tls.key",
                                    "name": "server-key",
                                    "subPath": "tls.key",
                                    "readOnly": True,
                                },
                            ],
                        }],
                        "dnsPolicy": "ClusterFirst",
                        "restartPolicy": "Always",
                        "volumes": [
                            {
                                "name": "mariadb-data",
                                "persistentVolumeClaim": {"claimName": "mariadb"},
                            },
                            {
                                "name": "ca-cert",
                                "secret": {"secretName": "mariadb-ca"},
                            },
                            {
                                "name": "server-cert",
                                "secret": {"secretName": "mariadb-server-cert"},
                            },
                            {
                                "name": "server-key",
                                "secret": {"secretName": "mariadb-server-key"},
                            },
                        ],
                    }
                }

                with Deployment(
                    name="mariadb",
                    client=admin_client,
                    namespace=model_namespace.name,
                    label={"name": "mariadb"},
                    replicas=1,
                    selector={"matchLabels": {"name": "mariadb", "app": "mariadb"}},
                    template=deployment_template,
                    wait_for_resource=True,
                    teardown=teardown_resources,
                ) as deployment:
                    yield deployment


@pytest.fixture(scope="class")
def trustyai_db_ca_secret(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mariadb: Deployment,
    teardown_resources: bool,
) -> Generator[Secret, Any]:
    """Provides TLS CA certificate secret for TrustyAI to connect to MariaDB.

    Copies the CA certificate from MariaDB's CA secret for TrustyAI to use.
    """
    if pytestconfig.option.post_upgrade:
        secret = Secret(
            client=admin_client,
            name=f"{TRUSTYAI_SERVICE_NAME}-db-ca",
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield secret
        secret.clean_up()
    else:
        # Get the MariaDB CA secret
        mariadb_ca_secret = Secret(
            client=admin_client,
            name="mariadb-ca",
            namespace=model_namespace.name,
            ensure_exists=True,
        )

        # Create TrustyAI's copy of the CA secret
        with Secret(
            client=admin_client,
            name=f"{TRUSTYAI_SERVICE_NAME}-db-ca",
            namespace=model_namespace.name,
            data_dict={"ca.crt": mariadb_ca_secret.instance.data["ca.crt"]},
            teardown=teardown_resources,
        ) as secret:
            yield secret


@pytest.fixture(scope="class")
def mlserver_runtime(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    minio_data_connection: Secret,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    mlserver_runtime_kwargs = {
        "client": admin_client,
        "namespace": model_namespace.name,
        "name": KSERVE_MLSERVER,
    }

    if pytestconfig.option.post_upgrade:
        serving_runtime = ServingRuntime(**mlserver_runtime_kwargs)
        yield serving_runtime
        serving_runtime.clean_up()

    else:
        with ServingRuntime(
            containers=KSERVE_MLSERVER_CONTAINERS,
            supported_model_formats=KSERVE_MLSERVER_SUPPORTED_MODEL_FORMATS,
            protocol_versions=["v2"],
            annotations=KSERVE_MLSERVER_ANNOTATIONS,
            label={Labels.OpenDataHub.DASHBOARD: "true"},
            teardown=teardown_resources,
            **mlserver_runtime_kwargs,
        ) as mlserver:
            yield mlserver


@pytest.fixture(scope="class")
def gaussian_credit_model(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_pod: Pod,
    minio_service: Service,
    minio_data_connection: Secret,
    mlserver_runtime: ServingRuntime,
    kserve_raw_config: ConfigMap,
    kserve_logger_ca_bundle: ConfigMap,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    gaussian_credit_model_kwargs = {
        "client": admin_client,
        "namespace": model_namespace.name,
        "name": GAUSSIAN_CREDIT_MODEL,
    }

    if pytestconfig.option.post_upgrade:
        isvc = InferenceService(**gaussian_credit_model_kwargs)
        yield isvc
        isvc.clean_up()
    else:
        with create_isvc(
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            model_format=XGBOOST,
            runtime=mlserver_runtime.name,
            storage_key=minio_data_connection.name,
            storage_path=GAUSSIAN_CREDIT_MODEL_STORAGE_PATH,
            enable_auth=True,
            external_route=True,
            wait_for_predictor_pods=False,
            resources=GAUSSIAN_CREDIT_MODEL_RESOURCES,
            teardown=teardown_resources,
            **gaussian_credit_model_kwargs,
        ) as isvc:
            wait_for_isvc_deployment_registered_by_trustyai_service(
                client=admin_client,
                isvc=isvc,
                runtime_name=mlserver_runtime.name,
            )
            yield isvc


@pytest.fixture(scope="class")
def isvc_getter_service_account(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[ServiceAccount, Any, Any]:
    """Provides a ServiceAccount for fetching InferenceServices.

    In post-upgrade mode, references the existing ServiceAccount created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new ServiceAccount and manages
    cleanup.
    """
    if pytestconfig.option.post_upgrade:
        service_account = ServiceAccount(
            client=admin_client,
            name=ISVC_GETTER,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield service_account
        service_account.clean_up()
    else:
        with create_isvc_getter_service_account(
            client=admin_client, namespace=model_namespace, name=ISVC_GETTER, teardown=teardown_resources
        ) as service_account:
            yield service_account


@pytest.fixture(scope="class")
def isvc_getter_role(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[Role, Any, Any]:
    """Provides a Role with permissions to get, list, and watch InferenceServices.

    In post-upgrade mode, references the existing Role created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new Role with InferenceService
    access permissions and manages cleanup.
    """
    if pytestconfig.option.post_upgrade:
        role = Role(
            client=admin_client,
            name=ISVC_GETTER,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield role
        role.clean_up()
    else:
        with create_isvc_getter_role(
            client=admin_client, namespace=model_namespace, name=ISVC_GETTER, teardown=teardown_resources
        ) as role:
            yield role


@pytest.fixture(scope="class")
def isvc_getter_role_binding(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    isvc_getter_role: Role,
    isvc_getter_service_account: ServiceAccount,
    teardown_resources: bool,
) -> Generator[RoleBinding, Any, Any]:
    """Provides a RoleBinding to link the ServiceAccount to the InferenceService getter Role.

    In post-upgrade mode, references the existing RoleBinding created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new RoleBinding
    and manages cleanup.
    """
    if pytestconfig.option.post_upgrade:
        role_binding = RoleBinding(
            client=admin_client,
            name=ISVC_GETTER,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield role_binding
        role_binding.clean_up()
    else:
        with create_isvc_getter_role_binding(
            client=admin_client,
            namespace=model_namespace,
            role=isvc_getter_role,
            service_account=isvc_getter_service_account,
            name=ISVC_GETTER,
            teardown=teardown_resources,
        ) as role_binding:
            yield role_binding


@pytest.fixture(scope="class")
def isvc_getter_token_secret(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    isvc_getter_service_account: ServiceAccount,
    isvc_getter_role_binding: RoleBinding,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """Provides a token Secret for the InferenceService getter ServiceAccount.

    In post-upgrade mode, references the existing token Secret created during pre-upgrade tests
    and cleans it up after post-upgrade tests complete.

    In pre-upgrade mode (or when no upgrade flag is set), creates a new token Secret and manages
    cleanup via teardown_resources.
    """
    if pytestconfig.option.post_upgrade:
        secret = Secret(
            client=admin_client,
            name="sa-token",
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        yield secret
        secret.clean_up()
    else:
        with create_isvc_getter_token_secret(
            client=admin_client,
            name="sa-token",
            namespace=model_namespace,
            service_account=isvc_getter_service_account,
            teardown=teardown_resources,
        ) as secret:
            yield secret


@pytest.fixture(scope="class")
def isvc_getter_token(isvc_getter_service_account: ServiceAccount, isvc_getter_token_secret: Secret) -> str:
    return RedactedString(value=create_inference_token(model_service_account=isvc_getter_service_account))
