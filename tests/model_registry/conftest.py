import os
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.infrastructure import Infrastructure
from ocp_resources.namespace import Namespace
from ocp_resources.oauth import OAuth
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from pytest import Config, FixtureRequest
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from tests.model_registry.constants import (
    DB_BASE_RESOURCES_NAME,
    DB_RESOURCE_NAME,
    KUBERBACPROXY_STR,
    MR_INSTANCE_BASE_NAME,
    MR_INSTANCE_NAME,
    MR_OPERATOR_NAME,
)
from tests.model_registry.utils import (
    generate_namespace_name,
    get_byoidc_user_credentials,
    get_model_registry_metadata_resources,
    get_model_registry_objects,
    get_rest_headers,
    wait_for_default_resource_cleanedup,
)
from utilities.constants import DscComponents, Labels
from utilities.general import (
    generate_random_name,
    wait_for_oauth_openshift_deployment,
    wait_for_pods_by_labels,
    wait_for_pods_running,
)
from utilities.infra import get_data_science_cluster, login_with_user_password, wait_for_dsc_status_ready
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from utilities.user_utils import UserTestSession, create_htpasswd_file, wait_for_user_creation

DEFAULT_TOKEN_DURATION = "10m"
LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session")
def model_registry_namespace(updated_dsc_component_state_scope_session: DataScienceCluster) -> str:
    return updated_dsc_component_state_scope_session.instance.spec.components.modelregistry.registriesNamespace


@pytest.fixture(scope="session")
def updated_dsc_component_state_scope_session(
    pytestconfig: Config,
    admin_client: DynamicClient,
) -> Generator[DataScienceCluster, Any, Any]:
    dsc_resource = get_data_science_cluster(client=admin_client)
    original_namespace_name = dsc_resource.instance.spec.components.modelregistry.registriesNamespace
    if pytestconfig.option.custom_namespace:
        resource_editor = ResourceEditor(
            patches={
                dsc_resource: {
                    "spec": {
                        "components": {
                            DscComponents.MODELREGISTRY: {
                                "managementState": DscComponents.ManagementState.REMOVED,
                                "registriesNamespace": original_namespace_name,
                            },
                        }
                    }
                }
            }
        )
        try:
            # first disable MR
            resource_editor.update(backup_resources=True)
            wait_for_dsc_status_ready(dsc_resource=dsc_resource)
            # now delete the original namespace:
            original_namespace = Namespace(client=admin_client, name=original_namespace_name, wait_for_resource=True)
            original_namespace.delete(wait=True)
            # Now enable it with the custom namespace
            with ResourceEditor(
                patches={
                    dsc_resource: {
                        "spec": {
                            "components": {
                                DscComponents.MODELREGISTRY: {
                                    "managementState": DscComponents.ManagementState.MANAGED,
                                    "registriesNamespace": py_config["model_registry_namespace"],
                                },
                            }
                        }
                    }
                }
            ):
                namespace = Namespace(
                    client=admin_client, name=py_config["model_registry_namespace"], wait_for_resource=True
                )
                namespace.wait_for_status(status=Namespace.Status.ACTIVE)
                wait_for_pods_running(
                    admin_client=admin_client,
                    namespace_name=py_config["applications_namespace"],
                    number_of_consecutive_checks=6,
                )
                wait_for_pods_running(
                    admin_client=admin_client,
                    namespace_name=py_config["model_registry_namespace"],
                    number_of_consecutive_checks=6,
                )
                yield dsc_resource
        finally:
            resource_editor.restore()
            Namespace(client=admin_client, name=py_config["model_registry_namespace"]).delete(wait=True)
            # create the original namespace object again, so that we can wait for it to be created first
            original_namespace = Namespace(client=admin_client, name=original_namespace_name, wait_for_resource=True)
            original_namespace.wait_for_status(status=Namespace.Status.ACTIVE)
            wait_for_pods_running(
                admin_client=admin_client,
                namespace_name=py_config["applications_namespace"],
                number_of_consecutive_checks=6,
            )
    else:
        LOGGER.info("Model Registry is enabled by default and does not require any setup.")
        yield dsc_resource


@pytest.fixture()
def model_registry_operator_pod(admin_client: DynamicClient) -> Generator[Pod, Any, Any]:
    """Get the model registry operator pod."""
    yield wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=py_config["applications_namespace"],
        label_selector=f"{Labels.OpenDataHubIo.NAME}={MR_OPERATOR_NAME}",
        expected_num_pods=1,
    )[0]


@pytest.fixture(scope="module")
def test_idp_user(
    request: pytest.FixtureRequest,
    original_user: str,
    api_server_url: str,
    is_byoidc: bool,
    admin_client: DynamicClient,
) -> Generator[UserTestSession | None]:
    """
    Session-scoped fixture that creates a test IDP user and cleans it up after all tests.
    Returns a UserTestSession object that contains all necessary credentials and contexts.
    """
    if is_byoidc:
        # For BYOIDC, we would be using a preconfigured group and username for actual api calls.
        yield
    else:
        user_credentials_rbac = request.getfixturevalue(argname="user_credentials_rbac")
        _ = request.getfixturevalue(argname="created_htpasswd_secret")
        _ = request.getfixturevalue(argname="updated_oauth_config")
        idp_session = None
        try:
            if wait_for_user_creation(
                username=user_credentials_rbac["username"],
                password=user_credentials_rbac["password"],
                cluster_url=api_server_url,
            ):
                # undo the login as test user if we were successful in logging in as test user
                LOGGER.info(f"Undoing login as test user and logging in as {original_user}")
                login_with_user_password(api_address=api_server_url, user=original_user)

            idp_session = UserTestSession(
                idp_name=user_credentials_rbac["idp_name"],
                secret_name=user_credentials_rbac["secret_name"],
                username=user_credentials_rbac["username"],
                password=user_credentials_rbac["password"],
                original_user=original_user,
                api_server_url=api_server_url,
                client=admin_client,
            )
            LOGGER.info(f"Created session test IDP user: {idp_session.username}")

            yield idp_session

        finally:
            if idp_session:
                LOGGER.info(f"Cleaning up test IDP user: {idp_session.username}")
                idp_session.cleanup()


@pytest.fixture(scope="session")
def api_server_url(admin_client: DynamicClient) -> str:
    """
    Get api server url from the cluster.
    """
    infrastructure = Infrastructure(client=admin_client, name="cluster", ensure_exists=True)
    return infrastructure.instance.status.apiServerURL


@pytest.fixture(scope="module")
def created_htpasswd_secret(
    is_byoidc: bool, admin_client: DynamicClient, original_user: str, user_credentials_rbac: dict[str, str]
) -> Generator[UserTestSession | None]:
    """
    Session-scoped fixture that creates a test IDP user and cleans it up after all tests.
    Returns a UserTestSession object that contains all necessary credentials and contexts.
    """
    if is_byoidc:
        yield

    else:
        temp_path, htpasswd_b64 = create_htpasswd_file(
            username=user_credentials_rbac["username"], password=user_credentials_rbac["password"]
        )
        try:
            LOGGER.info(f"Creating secret {user_credentials_rbac['secret_name']} in openshift-config namespace")
            with Secret(
                client=admin_client,
                name=user_credentials_rbac["secret_name"],
                namespace="openshift-config",
                htpasswd=htpasswd_b64,
                type="Opaque",
                wait_for_resource=True,
            ) as secret:
                yield secret
        finally:
            # Clean up the temporary file
            temp_path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def updated_oauth_config(
    is_byoidc: bool, admin_client: DynamicClient, original_user: str, user_credentials_rbac: dict[str, str]
) -> Generator[Any]:
    if is_byoidc:
        yield
    else:
        # Get current providers and add the new one
        oauth = OAuth(client=admin_client, name="cluster")
        identity_providers = oauth.instance.spec.identityProviders

        new_idp = {
            "name": user_credentials_rbac["idp_name"],
            "mappingMethod": "claim",
            "type": "HTPasswd",
            "htpasswd": {"fileData": {"name": user_credentials_rbac["secret_name"]}},
        }
        updated_providers = identity_providers + [new_idp]

        LOGGER.info("Updating OAuth")
        identity_providers_patch = ResourceEditor(patches={oauth: {"spec": {"identityProviders": updated_providers}}})
        identity_providers_patch.update(backup_resources=True)
        # Wait for OAuth server to be ready
        wait_for_oauth_openshift_deployment(client=admin_client)
        LOGGER.info(f"Added IDP {user_credentials_rbac['idp_name']} to OAuth configuration")
        yield
        identity_providers_patch.restore()
        wait_for_oauth_openshift_deployment(client=admin_client)


@pytest.fixture(scope="module")
def user_credentials_rbac(
    is_byoidc: bool,
    admin_client: DynamicClient,
) -> dict[str, str]:
    if is_byoidc:
        byoidc_creds = get_byoidc_user_credentials(client=admin_client, username="mr-non-admin")
        return {
            "username": byoidc_creds["username"],
            "password": byoidc_creds["password"],
            "idp_name": "byoidc",
            "secret_name": None,
        }
    else:
        random_str = generate_random_name()
        return {
            "username": f"test-user-{random_str}",
            "password": f"test-password-{random_str}",
            "idp_name": f"test-htpasswd-idp-{random_str}",
            "secret_name": f"test-htpasswd-secret-{random_str}",
        }


@pytest.fixture(scope="class")
def model_registry_instance(
    request: pytest.FixtureRequest,
    pytestconfig: Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
    model_registry_metadata_db_resources: dict[Any, Any],
    model_registry_namespace: str,
) -> Generator[list[Any], Any, Any]:
    param = getattr(request, "param", {})
    if pytestconfig.option.post_upgrade:
        mr_instance = ModelRegistry(
            client=admin_client, name=MR_INSTANCE_NAME, namespace=model_registry_namespace, ensure_exists=True
        )
        yield [mr_instance]
        mr_instance.delete(wait=True)
    else:
        db_name = param.get("db_name", "mysql")
        mr_objects = get_model_registry_objects(
            client=admin_client,
            namespace=model_registry_namespace,
            base_name=MR_INSTANCE_BASE_NAME,
            num=param.get("num_resources", 1),
            teardown_resources=teardown_resources,
            params=param,
            db_backend=db_name,
        )
        with ExitStack() as stack:
            mr_instances = [stack.enter_context(mr_obj) for mr_obj in mr_objects]
            for mr_instance in mr_instances:
                mr_instance.wait_for_condition(condition="Available", status="True")
                mr_instance.wait_for_condition(condition=KUBERBACPROXY_STR, status="True")
                wait_for_pods_running(
                    admin_client=admin_client, namespace_name=model_registry_namespace, number_of_consecutive_checks=6
                )
            yield mr_instances
        if db_name == "default":
            wait_for_default_resource_cleanedup(admin_client=admin_client, namespace_name=model_registry_namespace)


@pytest.fixture(scope="class")
def model_registry_metadata_db_resources(
    request: FixtureRequest,
    admin_client: DynamicClient,
    pytestconfig: Config,
    teardown_resources: bool,
    model_registry_namespace: str,
) -> Generator[dict[Any, Any]]:
    num_resources = getattr(request, "param", {}).get("num_resources", 1)
    db_backend = getattr(request, "param", {}).get("db_name", "mysql")

    if pytestconfig.option.post_upgrade:
        resources = {
            Secret: [
                Secret(
                    client=admin_client, name=DB_RESOURCE_NAME, namespace=model_registry_namespace, ensure_exists=True
                )
            ],
            PersistentVolumeClaim: [
                PersistentVolumeClaim(
                    client=admin_client, name=DB_RESOURCE_NAME, namespace=model_registry_namespace, ensure_exists=True
                )
            ],
            Service: [
                Service(
                    client=admin_client, name=DB_RESOURCE_NAME, namespace=model_registry_namespace, ensure_exists=True
                )
            ],
            ConfigMap: [
                ConfigMap(
                    client=admin_client, name=DB_RESOURCE_NAME, namespace=model_registry_namespace, ensure_exists=True
                )
            ]
            if db_backend == "mariadb"
            else [],
            Deployment: [
                Deployment(
                    client=admin_client, name=DB_RESOURCE_NAME, namespace=model_registry_namespace, ensure_exists=True
                )
            ],
        }
        yield resources
        for kind in [Deployment, ConfigMap, Service, PersistentVolumeClaim, Secret]:
            for resource in resources[kind]:
                resource.delete(wait=True)
    else:
        resources_instances = {}
        if db_backend == "default":
            yield resources_instances
        else:
            resources = get_model_registry_metadata_resources(
                base_name=DB_BASE_RESOURCES_NAME,
                namespace=model_registry_namespace,
                num_resources=num_resources,
                db_backend=db_backend,
                teardown_resources=teardown_resources,
                client=admin_client,
            )
            with ExitStack() as stack:
                for kind_name in [Secret, PersistentVolumeClaim, Service, ConfigMap, Deployment]:
                    if resources[kind_name]:
                        LOGGER.info(f"Creating {num_resources} {kind_name} resources")
                        resources_instances[kind_name] = [
                            stack.enter_context(resource_obj) for resource_obj in resources[kind_name]
                        ]
                for deployment in resources_instances[Deployment]:
                    deployment.wait_for_replicas(deployed=True)
                yield resources_instances


@pytest.fixture(scope="class")
def model_registry_rest_headers(current_client_token: str) -> dict[str, str]:
    return get_rest_headers(token=current_client_token)


@pytest.fixture(scope="class")
def sa_namespace(request: pytest.FixtureRequest, admin_client: DynamicClient) -> Generator[Namespace]:
    """
    Creates a namespace
    """

    test_file = os.path.relpath(request.fspath.strpath, start=os.path.dirname(__file__))
    ns_name = generate_namespace_name(file_path=test_file)
    LOGGER.info(f"Creating temporary namespace: {ns_name}")
    with Namespace(client=admin_client, name=ns_name) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=120)
        yield ns


@pytest.fixture()
def login_as_test_user(is_byoidc: bool, api_server_url: str, original_user: str, test_idp_user) -> Generator[None]:
    """
    Fixture to log in as a test user and restore original user after test.

    This fixture is used for RBAC tests to switch context to a non-admin test user.
    Used by both model registry and model catalog RBAC tests.
    """
    if is_byoidc:
        yield
    else:
        from utilities.user_utils import UserTestSession

        if isinstance(test_idp_user, UserTestSession):
            username = test_idp_user.username
            password = test_idp_user.password
        else:
            username = test_idp_user
            password = None

        LOGGER.info(f"Logging in as {username}")
        login_with_user_password(
            api_address=api_server_url,
            user=username,
            password=password,
        )
        yield
        LOGGER.info(f"Logging in as {original_user}")
        login_with_user_password(
            api_address=api_server_url,
            user=original_user,
        )


@pytest.fixture(scope="class")
def service_account(admin_client: DynamicClient, sa_namespace: Namespace) -> Generator[Any]:
    """
    Creates a ServiceAccount.
    """

    sa_name = generate_random_name(prefix="mr-test-user")
    LOGGER.info(f"Creating ServiceAccount: {sa_name} in namespace {sa_namespace.name}")
    with ServiceAccount(client=admin_client, name=sa_name, namespace=sa_namespace.name, wait_for_resource=True) as sa:
        yield sa
