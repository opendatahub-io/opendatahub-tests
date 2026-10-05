import datetime
import re
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.trustyai_service import TrustyAIService
from timeout_sampler import TimeoutSampler, retry

from utilities.constants import MARIA_DB_IMAGE, TRUSTYAI_SERVICE_NAME
from utilities.exceptions import TooManyPodsError, UnexpectedFailureError
from utilities.general import validate_container_images, wait_for_pods_by_labels

LOGGER = structlog.get_logger(name=__name__)


def wait_for_mariadb_pods(client: DynamicClient, deployment_name: str, namespace: str, timeout: int = 900) -> None:
    def _get_mariadb_pods() -> list[Pod]:
        return list(
            Pod.get(
                client=client,
                namespace=namespace,
                label_selector=f"name={deployment_name}",
            )
        )

    sampler = TimeoutSampler(wait_timeout=timeout, sleep=1, func=lambda: bool(_get_mariadb_pods()))

    for sample in sampler:
        if sample:
            break

    pods = _get_mariadb_pods()
    for pod in pods:
        pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status="True",
        )


def _generate_mariadb_tls_certs(namespace_name: str) -> tuple[str, str, str]:
    """Generate self-signed TLS certificates for MariaDB.

    Returns:
        tuple: (ca_cert_pem, server_cert_pem, server_key_pem)
    """
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    ca_subject = ca_issuer = x509.Name(
        attributes=[x509.NameAttribute(oid=NameOID.COMMON_NAME, value=f"mariadb-ca-{namespace_name}")]
    )
    ca_cert = (
        x509
        .CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_issuer)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(private_key=ca_key, algorithm=hashes.SHA256())
    )

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    server_subject = x509.Name(
        attributes=[x509.NameAttribute(oid=NameOID.COMMON_NAME, value=f"mariadb.{namespace_name}.svc.cluster.local")]
    )
    server_cert = (
        x509
        .CertificateBuilder()
        .subject_name(server_subject)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("mariadb"),
                x509.DNSName(f"mariadb.{namespace_name}.svc"),
                x509.DNSName(f"mariadb.{namespace_name}.svc.cluster.local"),
            ]),
            critical=False,
        )
        .sign(private_key=ca_key, algorithm=hashes.SHA256())
    )

    ca_cert_pem = ca_cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
    server_cert_pem = server_cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
    server_key_pem = server_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    return ca_cert_pem, server_cert_pem, server_key_pem


@contextmanager
def create_standalone_mariadb(
    client: DynamicClient,
    namespace_name: str,
    name: str,
    db_credentials_secret_name: str,
    teardown: bool = True,
) -> Generator[Deployment, Any, Any]:
    """Create a standalone MariaDB deployment with TLS enabled.

    Creates TLS secrets, PVC, Service, and Deployment for MariaDB.
    Uses Red Hat registry image to avoid Docker Hub rate limits.
    """
    ca_cert, server_cert, server_key = _generate_mariadb_tls_certs(namespace_name=namespace_name)

    with (
        Secret(
            client=client,
            name=f"{name}-ca",
            namespace=namespace_name,
            string_data={"ca.crt": ca_cert},
            teardown=teardown,
        ),
        Secret(
            client=client,
            name=f"{name}-server-cert",
            namespace=namespace_name,
            string_data={"tls.crt": server_cert},
            teardown=teardown,
        ),
        Secret(
            client=client,
            name=f"{name}-server-key",
            namespace=namespace_name,
            string_data={"tls.key": server_key},
            teardown=teardown,
        ),
        PersistentVolumeClaim(
            accessmodes="ReadWriteOnce",
            name=name,
            namespace=namespace_name,
            client=client,
            size="1Gi",
            teardown=teardown,
        ),
        Service(
            kind_dict={
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": name, "namespace": namespace_name},
                "spec": {
                    "selector": {"name": name},
                    "ports": [{"port": 3306, "targetPort": 3306, "name": "mysql", "protocol": "TCP"}],
                },
            },
            teardown=teardown,
        ),
    ):
        deployment_template = {
            "metadata": {
                "labels": {"name": name, "app": "mariadb", "component": "database"},
            },
            "spec": {
                "containers": [
                    {
                        "name": "mariadb",
                        "image": MARIA_DB_IMAGE,
                        "imagePullPolicy": "IfNotPresent",
                        "env": [
                            {
                                "name": "MYSQL_USER",
                                "valueFrom": {
                                    "secretKeyRef": {"name": db_credentials_secret_name, "key": "databaseUsername"}
                                },
                            },
                            {
                                "name": "MYSQL_PASSWORD",
                                "valueFrom": {
                                    "secretKeyRef": {"name": db_credentials_secret_name, "key": "databasePassword"}
                                },
                            },
                            {
                                "name": "MYSQL_ROOT_PASSWORD",
                                "valueFrom": {
                                    "secretKeyRef": {"name": db_credentials_secret_name, "key": "databasePassword"}
                                },
                            },
                            {
                                "name": "MYSQL_DATABASE",
                                "valueFrom": {
                                    "secretKeyRef": {"name": db_credentials_secret_name, "key": "databaseName"}
                                },
                            },
                            {
                                "name": "MARIADB_ROOT_PASSWORD",
                                "valueFrom": {
                                    "secretKeyRef": {"name": db_credentials_secret_name, "key": "databasePassword"}
                                },
                            },
                        ],
                        "ports": [{"containerPort": 3306, "protocol": "TCP"}],
                        "command": [
                            "run-mysqld",
                            "--ssl-ca=/etc/mysql/certs/ca.crt",
                            "--ssl-cert=/etc/mysql/certs/tls.crt",
                            "--ssl-key=/etc/mysql/certs/tls.key",
                            "--require-secure-transport=ON",
                        ],
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
                        "securityContext": {"capabilities": {}, "privileged": False},
                        "terminationMessagePath": "/dev/termination-log",
                        "volumeMounts": [
                            {"mountPath": "/var/lib/mysql", "name": "mariadb-data"},
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
                    }
                ],
                "dnsPolicy": "ClusterFirst",
                "restartPolicy": "Always",
                "volumes": [
                    {"name": "mariadb-data", "persistentVolumeClaim": {"claimName": name}},
                    {"name": "ca-cert", "secret": {"secretName": f"{name}-ca"}},
                    {"name": "server-cert", "secret": {"secretName": f"{name}-server-cert"}},
                    {"name": "server-key", "secret": {"secretName": f"{name}-server-key"}},
                ],
            },
        }

        with Deployment(
            name=name,
            client=client,
            namespace=namespace_name,
            label={"name": name},
            replicas=1,
            selector={"matchLabels": {"name": name, "app": "mariadb"}},
            template=deployment_template,
            wait_for_resource=True,
            teardown=teardown,
        ) as deployment:
            wait_for_mariadb_pods(client=client, deployment_name=name, namespace=namespace_name)
            yield deployment


@retry(
    wait_timeout=120,
    sleep=5,
    exceptions_dict={TooManyPodsError: [], UnexpectedFailureError: []},
)
def validate_trustyai_service_db_conn_failure(
    client: DynamicClient, namespace: Namespace, label_selector: str | None
) -> bool:
    """Validate if invalid DB Certificate leads to pod crash loop.

    Waits for TrustyAIService pod to fail and checks if the pod is in a CrashLoopBackOff state and
    the LastState is in terminated state and the cause was a MariaDB TLS certificate exception.
    Also checks if there are more than one pod for the service.

    Args:
        client: The OpenShift client.
        namespace: Namespace under which the pod is created.
        label_selector: The label selector used to select the correct pod(s) to monitor.

    Returns:
        bool: True if pod failure is of expected state else False.

    Raises:
        TimeoutExpiredError: if the method takes longer than `wait_timeout` to return a value.
        TooManyPodsError: if the number of pods exceeds 1.
        UnexpectedFailureError: if the pod failure is different from the expected failure mode.

    """
    pods = list(Pod.get(client=client, namespace=namespace.name, label_selector=label_selector))
    mariadb_conn_failure_regex = (
        r"^.+ERROR.+Could not connect to mariadb:.+"
        r"(PKIX path.*failed|SSL|socket|Connection refused)"
    )
    if pods:
        if len(pods) > 1:
            raise TooManyPodsError("More than one pod found in TrustyAIService.")
        for container_status in pods[0].instance.status.containerStatuses:
            if (terminate_state := container_status.lastState.terminated) and terminate_state.reason in (
                pods[0].Status.ERROR,
                pods[0].Status.CRASH_LOOPBACK_OFF,
            ):
                if not re.search(mariadb_conn_failure_regex, terminate_state.message):
                    raise UnexpectedFailureError(
                        f"Service {TRUSTYAI_SERVICE_NAME} did not fail with a mariadb connection failure as expected.\
                                  \nExpected format: {mariadb_conn_failure_regex}\
                                  \nGot: {terminate_state.message}"
                    )
                return True
    return False


@contextmanager
def create_trustyai_service(
    client: DynamicClient,
    namespace: str,
    storage: dict[str, str],
    metrics: dict[str, str],
    name: str = TRUSTYAI_SERVICE_NAME,
    data: dict[str, str] | None = None,
    wait_for_replicas: bool = True,
    teardown: bool = True,
) -> Generator[TrustyAIService, Any, Any]:
    """Creates TrustyAIService and TrustyAI deployment.

    Args:
         client: the client.
         namespace: Namespace to create the service in.
         storage: Dict with storage configuration.
         metrics: Dict with metrics configuration.
         name: Name of the TrustyAI service and deployment (default "trustyai-service").
         data: An optional dict with data.
         wait_for_replicas: Wait until replicas are available (default True).
         teardown: Teardown the service (default True).

    Yields:
         Generator[TrustyAIService, Any, Any]: The TrustyAI service.
    """
    with TrustyAIService(
        client=client,
        name=name,
        namespace=namespace,
        storage=storage,
        metrics=metrics,
        data=data,
        teardown=teardown,
    ) as trustyai_service:
        trustyai_deployment = Deployment(namespace=namespace, name=name, wait_for_resource=True)
        if wait_for_replicas:
            trustyai_deployment.wait_for_replicas()
        yield trustyai_service


@contextmanager
def create_isvc_getter_service_account(
    client: DynamicClient, namespace: Namespace, name: str, teardown: bool = True
) -> Generator[ServiceAccount, Any, Any]:
    """Creates a ServiceAccount for fetching InferenceServices.

    Args:
        client: The Kubernetes dynamic client.
        namespace: Namespace: The Namespace object where the ServiceAccount will be created.
        name: str: The name of the ServiceAccount.
        teardown: bool: Decides if the ServiceAccount should be deleted when the context exits.

    Yields:
        Generator[ServiceAccount, Any, Any]: The created ServiceAccount object.
    """
    with ServiceAccount(client=client, name=name, namespace=namespace.name, teardown=teardown) as sa:
        yield sa


@contextmanager
def create_isvc_getter_role(
    client: DynamicClient, namespace: Namespace, name: str, teardown: bool = True
) -> Generator[Role, Any, Any]:
    """Creates a Role with permissions to get, list, and watch InferenceServices.

    Args:
        client: DynamicClient: The Kubernetes dynamic client.
        namespace: Namespace: The Namespace object where the Role will be created.
        name: str: The name of the Role.
        teardown: bool: Decide if the Role should be deleted when the context exits.

    Yields:
        Generator[Role, Any, Any]: The created Role object.
    """
    with Role(
        client=client,
        name=name,
        namespace=namespace.name,
        rules=[
            {
                "apiGroups": ["serving.kserve.io"],
                "resources": ["inferenceservices"],
                "verbs": ["get", "list", "watch"],
            }
        ],
        teardown=teardown,
    ) as role:
        yield role


@contextmanager
def create_isvc_getter_role_binding(
    client: DynamicClient,
    namespace: Namespace,
    role: Role,
    service_account: ServiceAccount,
    name: str,
    teardown: bool = True,
) -> Generator[RoleBinding, Any, Any]:
    """Creates a RoleBinding to link a ServiceAccount to the InferenceService getter Role.

    Args:
        client: DynamicClient: The Kubernetes dynamic client.
        namespace: Namespace: The Namespace object where the RoleBinding will be created.
        role: Role: The Role object to bind.
        service_account: ServiceAccount: The ServiceAccount object to bind.
        name: str: The name of the RoleBinding.
        teardown: bool: Whether to delete the RoleBinding when the context exits.

    Yields:
        Generator[RoleBinding, Any, Any]: The created RoleBinding object.
    """
    with RoleBinding(
        client=client,
        name=name,
        namespace=namespace.name,
        subjects_kind="ServiceAccount",
        subjects_name=service_account.name,
        role_ref_kind="Role",
        role_ref_name=role.name,
        teardown=teardown,
    ) as rb:
        yield rb


@contextmanager
def create_isvc_getter_token_secret(
    client: DynamicClient, namespace: Namespace, service_account: ServiceAccount, name: str, teardown: bool = True
) -> Generator[Secret, Any, Any]:
    """Creates a Secret of type 'kubernetes.io/service-account-token' for a given ServiceAccount.

    Args:
        client: DynamicClient: The Kubernetes dynamic client.
        namespace: Namespace: The Namespace object where the Secret will be created.
        service_account: ServiceAccount: The ServiceAccount object for which the token Secret is created.
        name: str: The name of the Secret.
        teardown: bool: Whether to delete the Secret when the context exits.

    Yields:
        Generator[Secret, Any, Any]: The created Secret object.
    """
    with Secret(
        client=client,
        namespace=namespace.name,
        name=name,
        annotations={"kubernetes.io/service-account.name": service_account.name},
        type="kubernetes.io/service-account-token",
        teardown=teardown,
    ) as secret:
        yield secret


def validate_db_credentials_secret(secret: Secret, namespace_name: str) -> None:
    """Validates that a DB credentials secret has all required keys with non-empty values.

    Args:
        secret: Secret: The database credentials secret to validate.
        namespace_name: str: The namespace name for error messages.

    Raises:
        AssertionError: If the secret is missing required keys, has no data, or has empty values.
    """
    required_keys = {
        "databaseKind",
        "databaseName",
        "databaseUsername",
        "databasePassword",
        "databaseService",
        "databasePort",
        "databaseGeneration",
    }

    secret_data = secret.instance.data
    assert secret_data is not None, f"db-credentials secret has no data in namespace {namespace_name}"

    actual_keys = set(secret_data.keys())
    missing_keys = required_keys - actual_keys

    assert not missing_keys, (
        f"db-credentials secret is missing required keys: {missing_keys} in namespace {namespace_name}. "
        f"Available keys: {actual_keys}"
    )

    empty_keys = {key for key in required_keys if not secret_data.get(key)}
    assert not empty_keys, (
        f"db-credentials secret has empty required values: {empty_keys} in namespace {namespace_name}"
    )


def validate_trustyai_service_images(
    client: DynamicClient,
    related_images_refs: set[str],
    model_namespace: Namespace,
    label_selector: str,
    trustyai_operator_configmap: ConfigMap,
) -> None:
    """Validates trustyai service images against a set of related images.

    Args:
        client: DynamicClient: The Kubernetes dynamic client.
        related_images_refs: list[str]: Related images references from RHOAI CSV.
        model_namespace: Namespace: namespace to run the test against.
        label_selector: str: Label selector string to get the trustyai pod.
        trustyai_operator_configmap: ConfigMap: The trustyai operator configmap.

    Returns:
        None

    Raises:
        AssertionError: If any of the related images references are not present or invalid.
    """
    tai_image_refs = {
        value
        for key, value in trustyai_operator_configmap.instance.data.items()
        if key in ["kube-rbac-proxy", "trustyaiServiceImage"]
    }
    trustyai_service_pod = wait_for_pods_by_labels(
        admin_client=client, namespace=model_namespace.name, label_selector=label_selector, expected_num_pods=1
    )[0]
    validation_errors = validate_container_images(pod=trustyai_service_pod, valid_image_refs=tai_image_refs)
    assert len(validation_errors) == 0, validation_errors
    assert tai_image_refs.issubset(related_images_refs), "TrustyAI service container images are not present in CSV."
