import pytest
from pytest import Config
from typing import Generator, Any

from ocp_resources.infrastructure import Infrastructure
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment

from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from ocp_resources.resource import ResourceEditor

from pytest import FixtureRequest
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config
from model_registry.types import RegisteredModel
import uuid

from tests.model_registry.constants import (
    MR_OPERATOR_NAME,
    MR_INSTANCE_NAME,
    DB_RESOURCES_NAME,
    MODEL_REGISTRY_DB_SECRET_STR_DATA,
    MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
    OAUTH_PROXY_CONFIG_DICT,
    MODEL_REGISTRY_STANDARD_LABELS,
    ISTIO_CONFIG_DICT,
)
from tests.model_registry.rest_api.utils import ModelRegistryV1Alpha1
from utilities.constants import Labels, Protocols
from tests.model_registry.utils import (
    get_endpoint_from_mr_service,
    get_mr_service_by_label,
    get_model_registry_deployment_template_dict,
    get_model_registry_db_label_dict,
    wait_for_pods_running,
)
from utilities.constants import DscComponents
from model_registry import ModelRegistry as ModelRegistryClient
from semver import Version
from utilities.general import wait_for_pods_by_labels

LOGGER = get_logger(name=__name__)

MIN_MR_VERSION = Version.parse(version="2.20.0")


@pytest.fixture(scope="class")
def model_registry_namespace(updated_dsc_component_state_scope_class: DataScienceCluster) -> str:
    return updated_dsc_component_state_scope_class.instance.spec.components.modelregistry.registriesNamespace


@pytest.fixture(scope="class")
def model_registry_db_service(
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    teardown_resources: bool,
    is_model_registry_oauth: bool,
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
            label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME),
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
    is_model_registry_oauth: bool,
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
            label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME),
            teardown=teardown_resources,
        ) as pvc:
            yield pvc


@pytest.fixture(scope="class")
def model_registry_db_secret(
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    teardown_resources: bool,
    is_model_registry_oauth: bool,
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
            string_data=MODEL_REGISTRY_DB_SECRET_STR_DATA,
            label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME),
            annotations=MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
            teardown=teardown_resources,
        ) as mr_db_secret:
            yield mr_db_secret


@pytest.fixture(scope="class")
def model_registry_db_deployment(
    pytestconfig: Config,
    model_registry_namespace: str,
    model_registry_db_secret: Secret,
    teardown_resources: bool,
    is_model_registry_oauth: bool,
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
            label=get_model_registry_db_label_dict(db_resource_name=DB_RESOURCES_NAME),
            replicas=1,
            revision_history_limit=0,
            selector={"matchLabels": {"name": DB_RESOURCES_NAME}},
            strategy={"type": "Recreate"},
            template=get_model_registry_deployment_template_dict(
                secret_name=model_registry_db_secret.name, resource_name=DB_RESOURCES_NAME
            ),
            wait_for_resource=True,
            teardown=teardown_resources,
        ) as mr_db_deployment:
            mr_db_deployment.wait_for_replicas(deployed=True)
            yield mr_db_deployment


@pytest.fixture(scope="class")
def model_registry_mysql_metadata_db(
    model_registry_db_secret: Secret,
    model_registry_db_pvc: PersistentVolumeClaim,
    model_registry_db_service: Service,
    model_registry_db_deployment: Deployment,
) -> Deployment:
    return model_registry_db_deployment


@pytest.fixture(scope="class")
def model_registry_instance_mysql(
    pytestconfig: Config,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_mysql_config: dict[str, Any],
    teardown_resources: bool,
    is_model_registry_oauth: bool,
) -> Generator[ModelRegistry, Any, Any]:
    """Creates a model registry instance with oauth proxy configuration."""
    if pytestconfig.option.post_upgrade:
        mr_instance = ModelRegistry(name=MR_INSTANCE_NAME, namespace=model_registry_namespace, ensure_exists=True)
        yield mr_instance
        mr_instance.delete(wait=True)
    else:
        istio_config = None
        oauth_config = None
        mr_class_name = ModelRegistry
        if is_model_registry_oauth:
            LOGGER.warning("Requested Ouath Proxy configuration:")
            oauth_config = OAUTH_PROXY_CONFIG_DICT
        else:
            LOGGER.warning("Requested OSSM configuration:")
            istio_config = ISTIO_CONFIG_DICT
            mr_class_name = ModelRegistryV1Alpha1
        with mr_class_name(
            name=MR_INSTANCE_NAME,
            namespace=model_registry_namespace,
            label=MODEL_REGISTRY_STANDARD_LABELS,
            grpc={},
            rest={},
            istio=istio_config,
            oauth_proxy=oauth_config,
            mysql=model_registry_mysql_config,
            wait_for_resource=True,
            teardown=teardown_resources,
        ) as mr:
            mr.wait_for_condition(condition="Available", status="True")
            mr.wait_for_condition(condition="OAuthProxyAvailable", status="True")
            wait_for_pods_running(
                admin_client=admin_client, namespace_name=model_registry_namespace, number_of_consecutive_checks=6
            )
            yield mr


@pytest.fixture(scope="class")
def model_registry_mysql_config(request: FixtureRequest, model_registry_namespace: str) -> dict[str, Any]:
    """
    Fixture to build the MySQL config dictionary for Model Registry.
    Expects request.param to be a dict. If 'sslRootCertificateConfigMap' is not present, it defaults to None.
    If 'sslRootCertificateConfigMap' is present, it will be used to configure the MySQL connection.
    Args:
        request: The pytest request object
        model_registry_namespace: The model registry namespoce
    Returns:
        dict[str, Any]: The MySQL config dictionary
    """
    param = request.param if hasattr(request, "param") else {}
    config = {
        "host": f"{DB_RESOURCES_NAME}.{model_registry_namespace}.svc.cluster.local",
        "database": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
        "passwordSecret": {"key": "database-password", "name": DB_RESOURCES_NAME},
        "port": param.get("port", 3306),
        "skipDBCreation": False,
        "username": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
    }
    if "sslRootCertificateConfigMap" in param:
        config["sslRootCertificateConfigMap"] = param["sslRootCertificateConfigMap"]

    return config


@pytest.fixture(scope="class")
def model_registry_instance_rest_endpoint(admin_client: DynamicClient, model_registry_namespace: str) -> str:
    return get_endpoint_from_mr_service(
        svc=get_mr_service_by_label(
            client=admin_client,
            namespace_name=model_registry_namespace,
            mr_instance=ModelRegistry(name=MR_INSTANCE_NAME, namespace=model_registry_namespace),
        ),
        protocol=Protocols.REST,
    )


@pytest.fixture(scope="class")
def updated_dsc_component_state_scope_class(
    pytestconfig: Config,
    admin_client: DynamicClient,
    request: FixtureRequest,
    dsc_resource: DataScienceCluster,
    teardown_resources: bool,
    is_model_registry_oauth: bool,
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
    """
    Get a client for the model registry instance.
    Args:
        request: The pytest request object
        current_client_token: The current client token
    Returns:
        ModelRegistryClient: A client for the model registry instance
    """
    server, port = model_registry_instance_rest_endpoint.split(":")
    return ModelRegistryClient(
        server_address=f"{Protocols.HTTPS}://{server}",
        port=int(port),
        author="opendatahub-test",
        user_token=current_client_token,
        is_secure=False,
    )


@pytest.fixture(scope="class")
def registered_model(request: FixtureRequest, model_registry_client: ModelRegistryClient) -> RegisteredModel:
    return model_registry_client.register_model(
        name=request.param.get("model_name"),
        uri=request.param.get("model_uri"),
        version=request.param.get("model_version"),
        description=request.param.get("model_description"),
        model_format_name=request.param.get("model_format"),
        model_format_version=request.param.get("model_format_version"),
        storage_key=request.param.get("model_storage_key"),
        storage_path=request.param.get("model_storage_path"),
        metadata=request.param.get("model_metadata"),
    )


@pytest.fixture()
def model_registry_operator_pod(admin_client: DynamicClient) -> Generator[Pod, Any, Any]:
    """Get the model registry operator pod."""
    yield wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=py_config["applications_namespace"],
        label_selector=f"{Labels.OpenDataHubIo.NAME}={MR_OPERATOR_NAME}",
        expected_num_pods=1,
    )[0]


@pytest.fixture()
def model_registry_instance_pod(admin_client: DynamicClient) -> Generator[Pod, Any, Any]:
    """Get the model registry instance pod."""
    yield wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=py_config["model_registry_namespace"],
        label_selector=f"app={MR_INSTANCE_NAME}",
        expected_num_pods=1,
    )[0]


@pytest.fixture()
def model_registry_db_instance_pod(admin_client: DynamicClient) -> Generator[Pod, Any, Any]:
    """Get the model registry instance pod."""
    yield wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=py_config["model_registry_namespace"],
        label_selector=f"name={DB_RESOURCES_NAME}",
        expected_num_pods=1,
    )[0]


@pytest.fixture()
def set_mr_db_dirty(model_registry_db_instance_pod: Pod) -> int:
    """Set the model registry database dirty and return the latest migration version"""
    output = model_registry_db_instance_pod.execute(
        command=[
            "mysql",
            "-u",
            MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
            f"-p{MODEL_REGISTRY_DB_SECRET_STR_DATA['database-password']}",
            "-e",
            "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;",
            MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
        ]
    )
    latest_migration_version = int(output.strip().split()[1])
    model_registry_db_instance_pod.execute(
        command=[
            "mysql",
            "-u",
            MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
            f"-p{MODEL_REGISTRY_DB_SECRET_STR_DATA['database-password']}",
            "-e",
            f"UPDATE schema_migrations SET dirty = 1 WHERE version = {latest_migration_version};",
            MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
        ]
    )
    return latest_migration_version


@pytest.fixture()
def delete_mr_deployment() -> None:
    """Delete the model registry deployment"""
    mr_deployment = Deployment(
        name=MR_INSTANCE_NAME, namespace=py_config["model_registry_namespace"], ensure_exists=True
    )
    mr_deployment.delete(wait=True)


@pytest.fixture(scope="class")
def is_model_registry_oauth(request: FixtureRequest) -> bool:
    return getattr(request, "param", {}).get("use_oauth_proxy", True)


@pytest.fixture(scope="session")
def api_server_url(admin_client: DynamicClient) -> str:
    infrastructure = Infrastructure(client=admin_client, name="cluster", ensure_exists=True)
    return infrastructure.instance.status.apiServerURL


@pytest.fixture(scope="class")
def model_registry_rest_url(model_registry_instance_rest_endpoint: str) -> str:
    # address and port need to be split in the client instantiation
    return f"{Protocols.HTTPS}://{model_registry_instance_rest_endpoint}"


@pytest.fixture(scope="class")
def model_registry_rest_headers(current_client_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {current_client_token}",
        "accept": "application/json",
        "Content-Type": "application/json",
    }


@pytest.fixture(scope="class")
def model_registry_deployment_containers(model_registry_namespace: str) -> list[dict[str, Any]]:
    return Deployment(
        name=MR_INSTANCE_NAME, namespace=model_registry_namespace, ensure_exists=True
    ).instance.spec.template.spec.containers


@pytest.fixture(scope="class")
def model_registry_pod(admin_client: DynamicClient, model_registry_namespace: str) -> Pod:
    mr_pod = list(
        Pod.get(
            dyn_client=admin_client,
            namespace=model_registry_namespace,
            label_selector=f"app={MR_INSTANCE_NAME}",
        )
    )
    assert len(mr_pod) == 1
    return mr_pod[0]


# =============================================================================
# DUPLICATED RESOURCE FIXTURES
# =============================================================================


@pytest.fixture(scope="class")
def db_name_1() -> str:
    """Generate a unique name for the first DB instance."""
    return f"db-instance-1-{str(uuid.uuid4())[:8]}"


@pytest.fixture(scope="class")
def db_secret_1(model_registry_namespace: str, db_name_1: str, teardown_resources: bool) -> Generator[Secret, Any, Any]:
    """Create the first DB secret."""
    with Secret(
        name=db_name_1,
        namespace=model_registry_namespace,
        string_data=MODEL_REGISTRY_DB_SECRET_STR_DATA,
        label=get_model_registry_db_label_dict(db_resource_name=db_name_1),
        annotations=MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
        teardown=teardown_resources,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def db_pvc_1(
    model_registry_namespace: str, db_name_1: str, teardown_resources: bool
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """Create the first DB PVC."""
    with PersistentVolumeClaim(
        name=db_name_1,
        namespace=model_registry_namespace,
        accessmodes="ReadWriteOnce",
        size="5Gi",
        label=get_model_registry_db_label_dict(db_resource_name=db_name_1),
        teardown=teardown_resources,
    ) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def db_service_1(
    model_registry_namespace: str, db_name_1: str, teardown_resources: bool
) -> Generator[Service, Any, Any]:
    """Create the first DB service."""
    with Service(
        name=db_name_1,
        namespace=model_registry_namespace,
        ports=[{"name": "mysql", "port": 3306, "protocol": "TCP", "targetPort": 3306}],
        selector={"name": db_name_1},
        label=get_model_registry_db_label_dict(db_resource_name=db_name_1),
        teardown=teardown_resources,
    ) as service:
        yield service


@pytest.fixture(scope="class")
def db_deployment_1(
    model_registry_namespace: str, db_name_1: str, teardown_resources: bool
) -> Generator[Deployment, Any, Any]:
    """Create the first DB deployment."""
    with Deployment(
        name=db_name_1,
        namespace=model_registry_namespace,
        template=get_model_registry_deployment_template_dict(secret_name=db_name_1, resource_name=db_name_1),
        label=get_model_registry_db_label_dict(db_resource_name=db_name_1),
        replicas=1,
        revision_history_limit=0,
        selector={"matchLabels": {"name": db_name_1}},
        strategy={"type": "Recreate"},
        wait_for_resource=True,
        teardown=teardown_resources,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True)
        yield deployment


@pytest.fixture(scope="class")
def model_registry_instance_1(
    db_name_1: str, model_registry_namespace: str, teardown_resources: bool, is_model_registry_oauth: bool
) -> Generator[ModelRegistry, Any, Any]:
    """Create the first Model Registry instance (default/oauth)."""
    mysql_config = {
        "host": f"{db_name_1}.{model_registry_namespace}.svc.cluster.local",
        "database": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
        "passwordSecret": {"key": "database-password", "name": db_name_1},
        "port": 3306,
        "skipDBCreation": False,
        "username": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
    }
    istio_config = None
    oauth_config = None
    mr_class_name = ModelRegistry
    if is_model_registry_oauth:
        LOGGER.warning("Requested Ouath Proxy configuration:")
        oauth_config = OAUTH_PROXY_CONFIG_DICT
    else:
        LOGGER.warning("Requested OSSM configuration:")
        istio_config = ISTIO_CONFIG_DICT
        mr_class_name = ModelRegistryV1Alpha1
    mr_name = f"mr-instance-1-{str(uuid.uuid4())[:8]}"
    with mr_class_name(
        name=mr_name,
        namespace=model_registry_namespace,
        grpc={},
        rest={},
        label=MODEL_REGISTRY_STANDARD_LABELS,
        istio=istio_config,
        oauth_proxy=oauth_config,
        mysql=mysql_config,
        wait_for_resource=True,
        teardown=teardown_resources,
    ) as mr_instance:
        mr_instance.wait_for_condition(condition="Available", status="True")
        mr_instance.wait_for_condition(condition="OAuthProxyAvailable", status="True")
        yield mr_instance


# --- Instance 2 Resources ---


@pytest.fixture(scope="class")
def db_name_2() -> str:
    """Generate a unique name for the second DB instance."""
    return f"db-instance-2-{str(uuid.uuid4())[:8]}"


@pytest.fixture(scope="class")
def db_secret_2(model_registry_namespace: str, db_name_2: str, teardown_resources: bool) -> Generator[Secret, Any, Any]:
    """Create the second DB secret."""
    with Secret(
        name=db_name_2,
        namespace=model_registry_namespace,
        string_data=MODEL_REGISTRY_DB_SECRET_STR_DATA,
        label=get_model_registry_db_label_dict(db_resource_name=db_name_2),
        annotations=MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
        teardown=teardown_resources,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def db_pvc_2(
    model_registry_namespace: str, db_name_2: str, teardown_resources: bool
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """Create the second DB PVC."""
    with PersistentVolumeClaim(
        name=db_name_2,
        namespace=model_registry_namespace,
        accessmodes="ReadWriteOnce",
        size="5Gi",
        label=get_model_registry_db_label_dict(db_resource_name=db_name_2),
        teardown=teardown_resources,
    ) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def db_service_2(
    model_registry_namespace: str, db_name_2: str, teardown_resources: bool
) -> Generator[Service, Any, Any]:
    """Create the second DB service."""
    with Service(
        name=db_name_2,
        namespace=model_registry_namespace,
        ports=[{"name": "mysql", "port": 3306, "protocol": "TCP", "targetPort": 3306}],
        selector={"name": db_name_2},
        label=get_model_registry_db_label_dict(db_resource_name=db_name_2),
        teardown=teardown_resources,
    ) as service:
        yield service


@pytest.fixture(scope="class")
def db_deployment_2(
    model_registry_namespace: str, db_name_2: str, teardown_resources: bool
) -> Generator[Deployment, Any, Any]:
    """Create the second DB deployment."""
    with Deployment(
        name=db_name_2,
        namespace=model_registry_namespace,
        template=get_model_registry_deployment_template_dict(secret_name=db_name_2, resource_name=db_name_2),
        label=get_model_registry_db_label_dict(db_resource_name=db_name_2),
        replicas=1,
        revision_history_limit=0,
        selector={"matchLabels": {"name": db_name_2}},
        strategy={"type": "Recreate"},
        wait_for_resource=True,
        teardown=teardown_resources,
    ) as deployment:
        deployment.wait_for_replicas(deployed=True)
        yield deployment


@pytest.fixture(scope="class")
def model_registry_instance_2(
    model_registry_namespace: str, db_name_2: str, teardown_resources: bool, is_model_registry_oauth: bool
) -> Generator[ModelRegistry, Any, Any]:
    """Create the second Model Registry instance (istio)."""
    mysql_config = {
        "host": f"{db_name_2}.{model_registry_namespace}.svc.cluster.local",
        "database": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
        "passwordSecret": {"key": "database-password", "name": db_name_2},
        "port": 3306,
        "skipDBCreation": False,
        "username": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
    }
    istio_config = None
    oauth_config = None
    mr_class_name = ModelRegistry
    if is_model_registry_oauth:
        LOGGER.warning("Requested Ouath Proxy configuration:")
        oauth_config = OAUTH_PROXY_CONFIG_DICT
    else:
        LOGGER.warning("Requested OSSM configuration:")
        istio_config = ISTIO_CONFIG_DICT
        mr_class_name = ModelRegistryV1Alpha1
    mr_name = f"mr-instance-2-{str(uuid.uuid4())[:8]}"
    with mr_class_name(
        name=mr_name,
        namespace=model_registry_namespace,
        grpc={},
        rest={},
        label=MODEL_REGISTRY_STANDARD_LABELS,
        istio=istio_config,
        oauth_proxy=oauth_config,
        mysql=mysql_config,
        wait_for_resource=True,
        teardown=teardown_resources,
    ) as mr_instance:
        mr_instance.wait_for_condition(condition="Available", status="True")
        mr_instance.wait_for_condition(condition="OAuthProxyAvailable", status="True")
        yield mr_instance
