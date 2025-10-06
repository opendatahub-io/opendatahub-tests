import pytest
from typing import Generator, Any

from _pytest.config import Config
from _pytest.fixtures import FixtureRequest

from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.deployment import Deployment
from ocp_resources.model_registry import ModelRegistry
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient
from model_registry import ModelRegistry as ModelRegistryClient

from tests.model_registry.constants import DB_RESOURCES_NAME, MR_INSTANCE_NAME, MR_OPERATOR_NAME, DEFAULT_LABEL_DICT_DB
from tests.model_registry.utils import (
    get_endpoint_from_mr_service,
    get_mr_service_by_label,
    ModelRegistryV1Alpha1,
    wait_for_pods_running,
)
from utilities.constants import Annotations, Protocols, DscComponents
from pytest_testconfig import config as py_config

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session")
def teardown_resources(pytestconfig: pytest.Config) -> bool:
    delete_resources = True

    if pytestconfig.option.pre_upgrade:
        if delete_resources := pytestconfig.option.delete_pre_upgrade_resources:
            LOGGER.warning("Upgrade resources will be deleted")

    return delete_resources


@pytest.fixture(scope="class")
def model_registry_namespace(updated_dsc_component_state_scope_class: DataScienceCluster) -> str:
    return updated_dsc_component_state_scope_class.instance.spec.components.modelregistry.registriesNamespace


@pytest.fixture(scope="class")
def model_registry_db_service(
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    teardown_resources: bool,
) -> Generator[Service, Any, Any]:
    if pytestconfig.option.post_upgrade:
        mr_db_service = Service(name=DB_RESOURCES_NAME, namespace=model_registry_namespace, ensure_exists=True)
        yield mr_db_service
        mr_db_service.delete(wait=True)
    else:
        with Service(
            client=admin_client,
            name=DB_RESOURCES_NAME,
            namespace=model_registry_namespace,
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
                "name": DB_RESOURCES_NAME,
            },
            label=DEFAULT_LABEL_DICT_DB,
            annotations={
                "template.openshift.io/expose-uri": r"mysql://{.spec.clusterIP}:{.spec.ports[?(.name==\mysql\)].port}",
            },
            teardown=teardown_resources,
        ) as mr_db_service:
            yield mr_db_service


@pytest.fixture(scope="class")
def model_registry_db_pvc(
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    if pytestconfig.option.post_upgrade:
        mr_db_pvc = PersistentVolumeClaim(
            name=DB_RESOURCES_NAME, namespace=model_registry_namespace, ensure_exists=True
        )
        yield mr_db_pvc
        mr_db_pvc.delete(wait=True)
    else:
        with PersistentVolumeClaim(
            accessmodes="ReadWriteOnce",
            name=DB_RESOURCES_NAME,
            namespace=model_registry_namespace,
            client=admin_client,
            size="5Gi",
            label=DEFAULT_LABEL_DICT_DB,
            teardown=teardown_resources,
        ) as pvc:
            yield pvc


@pytest.fixture(scope="class")
def model_registry_db_secret(
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    if pytestconfig.option.post_upgrade:
        mr_db_secret = Secret(name=DB_RESOURCES_NAME, namespace=model_registry_namespace, ensure_exists=True)
        yield mr_db_secret
        mr_db_secret.delete(wait=True)
    else:
        with Secret(
            client=admin_client,
            name=DB_RESOURCES_NAME,
            namespace=model_registry_namespace,
            string_data={
                "database-name": "model_registry",
                "database-password": "TheBlurstOfTimes",  # pragma: allowlist secret
                "database-user": "mlmduser",  # pragma: allowlist secret
            },
            label=DEFAULT_LABEL_DICT_DB,
            annotations={
                "template.openshift.io/expose-database_name": "'{.data[''database-name'']}'",
                "template.openshift.io/expose-password": "'{.data[''database-password'']}'",
                "template.openshift.io/expose-username": "'{.data[''database-user'']}'",
            },
            teardown=teardown_resources,
        ) as mr_db_secret:
            yield mr_db_secret


@pytest.fixture(scope="class")
def model_registry_db_deployment(
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_db_secret: Secret,
    model_registry_db_pvc: PersistentVolumeClaim,
    model_registry_db_service: Service,
    teardown_resources: bool,
) -> Generator[Deployment, Any, Any]:
    if pytestconfig.option.post_upgrade:
        db_deployment = Deployment(name=DB_RESOURCES_NAME, namespace=model_registry_namespace, ensure_exists=True)
        yield db_deployment
        db_deployment.delete(wait=True)
    else:
        with Deployment(
            name=DB_RESOURCES_NAME,
            namespace=model_registry_namespace,
            annotations={
                "template.alpha.openshift.io/wait-for-ready": "true",
            },
            label=DEFAULT_LABEL_DICT_DB,
            replicas=1,
            revision_history_limit=0,
            selector={"matchLabels": {"name": DB_RESOURCES_NAME}},
            strategy={"type": "Recreate"},
            template={
                "metadata": {
                    "labels": {
                        "name": DB_RESOURCES_NAME,
                        "sidecar.istio.io/inject": "false",
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "env": [
                                {
                                    "name": "MYSQL_USER",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "key": "database-user",
                                            "name": f"{model_registry_db_secret.name}",
                                        }
                                    },
                                },
                                {
                                    "name": "MYSQL_PASSWORD",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "key": "database-password",
                                            "name": f"{model_registry_db_secret.name}",
                                        }
                                    },
                                },
                                {
                                    "name": "MYSQL_ROOT_PASSWORD",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "key": "database-password",
                                            "name": f"{model_registry_db_secret.name}",
                                        }
                                    },
                                },
                                {
                                    "name": "MYSQL_DATABASE",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "key": "database-name",
                                            "name": f"{model_registry_db_secret.name}",
                                        }
                                    },
                                },
                            ],
                            "args": [
                                "--datadir",
                                "/var/lib/mysql/datadir",
                                "--default-authentication-plugin=mysql_native_password",
                            ],
                            "image": "public.ecr.aws/docker/library/mysql@sha256:9de9d54fecee6253130e65154b930978b1fcc336bcc86dfd06e89b72a2588ebe",  # noqa: E501
                            "imagePullPolicy": "IfNotPresent",
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
                            "name": "mysql",
                            "ports": [{"containerPort": 3306, "protocol": "TCP"}],
                            "readinessProbe": {
                                "exec": {
                                    "command": [
                                        "/bin/bash",
                                        "-c",
                                        'mysql -D ${MYSQL_DATABASE} -u${MYSQL_USER} -p${MYSQL_ROOT_PASSWORD} -e "SELECT 1"',  # noqa: E501
                                    ]
                                },
                                "initialDelaySeconds": 10,
                                "timeoutSeconds": 5,
                            },
                            "securityContext": {"capabilities": {}, "privileged": False},
                            "terminationMessagePath": "/dev/termination-log",
                            "volumeMounts": [
                                {
                                    "mountPath": "/var/lib/mysql",
                                    "name": f"{DB_RESOURCES_NAME}-data",
                                }
                            ],
                        }
                    ],
                    "dnsPolicy": "ClusterFirst",
                    "restartPolicy": "Always",
                    "volumes": [
                        {
                            "name": f"{DB_RESOURCES_NAME}-data",
                            "persistentVolumeClaim": {"claimName": DB_RESOURCES_NAME},
                        }
                    ],
                },
            },
            wait_for_resource=True,
            teardown=teardown_resources,
        ) as mr_db_deployment:
            mr_db_deployment.wait_for_replicas(deployed=True)
            yield mr_db_deployment


@pytest.fixture(scope="class")
def model_registry_instance(
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_db_deployment: Deployment,
    model_registry_db_secret: Secret,
    model_registry_db_service: Service,
    teardown_resources: bool,
) -> Generator[ModelRegistry, Any, Any]:
    host = f"{model_registry_db_deployment.name}.{model_registry_db_deployment.namespace}.svc.cluster.local"
    if pytestconfig.option.post_upgrade:
        mr_instance = ModelRegistryV1Alpha1(
            name=MR_INSTANCE_NAME, namespace=model_registry_namespace, ensure_exists=True
        )
        yield mr_instance
        mr_instance.delete(wait=True)
    else:
        with ModelRegistryV1Alpha1(
            name=MR_INSTANCE_NAME,
            namespace=model_registry_namespace,
            label={
                Annotations.KubernetesIo.NAME: MR_INSTANCE_NAME,
                Annotations.KubernetesIo.INSTANCE: MR_INSTANCE_NAME,
                Annotations.KubernetesIo.PART_OF: MR_OPERATOR_NAME,
                Annotations.KubernetesIo.CREATED_BY: MR_OPERATOR_NAME,
            },
            grpc={},
            rest={},
            istio={
                "authProvider": "redhat-ods-applications-auth-provider",
                "gateway": {"grpc": {"tls": {}}, "rest": {"tls": {}}},
            },
            mysql={
                "host": host,
                "database": model_registry_db_secret.string_data["database-name"],
                "passwordSecret": {"key": "database-password", "name": DB_RESOURCES_NAME},
                "port": 3306,
                "skipDBCreation": False,
                "username": model_registry_db_secret.string_data["database-user"],
            },
            wait_for_resource=True,
            teardown=teardown_resources,
        ) as mr:
            mr.wait_for_condition(condition="Available", status="True")
            wait_for_pods_running(
                admin_client=admin_client, namespace_name=model_registry_namespace, number_of_consecutive_checks=6
            )
            yield mr


@pytest.fixture(scope="class")
def model_registry_instance_service(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_instance: ModelRegistry,
) -> Service:
    return get_mr_service_by_label(
        client=admin_client, ns=model_registry_namespace, mr_instance=model_registry_instance
    )


@pytest.fixture(scope="class")
def model_registry_instance_rest_endpoint(
    model_registry_instance_service: Service,
) -> str:
    return get_endpoint_from_mr_service(svc=model_registry_instance_service, protocol=Protocols.REST)


@pytest.fixture(scope="class")
def updated_dsc_component_state_scope_class(
    pytestconfig: Config,
    admin_client: DynamicClient,
    request: FixtureRequest,
    dsc_resource: DataScienceCluster,
    teardown_resources: bool,
) -> Generator[DataScienceCluster, Any, Any]:
    if not teardown_resources or pytestconfig.option.post_upgrade:
        # if we are not tearing down resources or we are in post upgrade, we don't need to do anything
        # the pre_upgrade/post_upgrade fixtures will handle the rest
        yield dsc_resource
    else:
        original_components = dsc_resource.instance.spec.components
        component_patch = request.param["component_patch"]

        with ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}}):
            for component_name in component_patch:
                dsc_resource.wait_for_condition(
                    condition=DscComponents.COMPONENT_MAPPING[component_name], status="True"
                )
            if component_patch.get(DscComponents.MODELREGISTRY):
                namespace = Namespace(
                    name=dsc_resource.instance.spec.components.modelregistry.registriesNamespace, ensure_exists=True
                )
                namespace.wait_for_status(status=Namespace.Status.ACTIVE)
            wait_for_pods_running(
                admin_client=admin_client,
                namespace_name=py_config["applications_namespace"],
                number_of_consecutive_checks=6,
            )
            yield dsc_resource

        for component_name, value in component_patch.items():
            LOGGER.info(f"Waiting for component {component_name} to be updated.")
            if original_components[component_name]["managementState"] == DscComponents.ManagementState.MANAGED:
                dsc_resource.wait_for_condition(
                    condition=DscComponents.COMPONENT_MAPPING[component_name], status="True"
                )
            if (
                component_name == DscComponents.MODELREGISTRY
                and value.get("managementState") == DscComponents.ManagementState.MANAGED
            ):
                # Since namespace specified in registriesNamespace is automatically created after setting
                # managementStateto Managed. We need to explicitly delete it on clean up.
                namespace = Namespace(name=value["registriesNamespace"], ensure_exists=True)
                if namespace:
                    namespace.delete(wait=True)


@pytest.fixture(scope="class")
def pre_upgrade_dsc_patch(
    dsc_resource: DataScienceCluster,
    admin_client: DynamicClient,
) -> DataScienceCluster:
    original_components = dsc_resource.instance.spec.components
    component_patch = {DscComponents.MODELREGISTRY: {"managementState": DscComponents.ManagementState.MANAGED}}
    if (
        original_components.get(DscComponents.MODELREGISTRY).get("managementState")
        == DscComponents.ManagementState.MANAGED
    ):
        pytest.fail("Model Registry is already set to Managed before upgrade - was this intentional?")
    else:
        editor = ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}})
        editor.update()
        dsc_resource.wait_for_condition(condition=DscComponents.COMPONENT_MAPPING["modelregistry"], status="True")
        namespace = Namespace(
            name=dsc_resource.instance.spec.components.modelregistry.registriesNamespace, ensure_exists=True
        )
        namespace.wait_for_status(status=Namespace.Status.ACTIVE)
        wait_for_pods_running(
            admin_client=admin_client,
            namespace_name=py_config["applications_namespace"],
            number_of_consecutive_checks=6,
        )
        return dsc_resource


@pytest.fixture(scope="class")
def post_upgrade_dsc_patch(
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    # yield right away so that the rest of the fixture is executed at teardown time
    yield dsc_resource

    # the state we found after the upgrade
    original_components = dsc_resource.instance.spec.components
    # We don't have an easy way to figure out the state of the components before the upgrade at runtime
    # For now we know that MR has to go back to Removed after post upgrade tests are run
    component_patch = {DscComponents.MODELREGISTRY: {"managementState": DscComponents.ManagementState.REMOVED}}
    if (
        original_components.get(DscComponents.MODELREGISTRY).get("managementState")
        == DscComponents.ManagementState.REMOVED
    ):
        pytest.fail("Model Registry is already set to Removed after upgrade - was this intentional?")
    else:
        editor = ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}})
        editor.update()
    ns = original_components.get(DscComponents.MODELREGISTRY).get("registriesNamespace")
    namespace = Namespace(name=ns, ensure_exists=True)
    if namespace:
        namespace.delete(wait=True)


@pytest.fixture(scope="class")
def model_registry_client(
    current_client_token: str,
    model_registry_instance_rest_endpoint: str,
) -> ModelRegistryClient:
    server, port = model_registry_instance_rest_endpoint.split(":")
    return ModelRegistryClient(
        server_address=f"{Protocols.HTTPS}://{server}",
        port=int(port),
        author="opendatahub-test",
        user_token=current_client_token,
        is_secure=False,
    )