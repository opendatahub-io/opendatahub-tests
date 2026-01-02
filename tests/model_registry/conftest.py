import pytest
from pytest import Config
from typing import Generator, Any

from ocp_resources.infrastructure import Infrastructure
from ocp_resources.oauth import OAuth
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.data_science_cluster import DataScienceCluster


from ocp_resources.resource import ResourceEditor

from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config

from utilities.general import wait_for_oauth_openshift_deployment
from tests.model_registry.utils import (
    get_byoidc_user_credentials,
)

from utilities.general import generate_random_name, wait_for_pods_running

from tests.model_registry.constants import (
    MR_OPERATOR_NAME,
)
from utilities.constants import Labels
from utilities.constants import DscComponents
from utilities.general import wait_for_pods_by_labels
from utilities.infra import get_data_science_cluster, wait_for_dsc_status_ready, login_with_user_password
from utilities.user_utils import UserTestSession, wait_for_user_creation, create_htpasswd_file

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
            original_namespace = Namespace(name=original_namespace_name, wait_for_resource=True)
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
                namespace = Namespace(name=py_config["model_registry_namespace"], wait_for_resource=True)
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
            Namespace(name=py_config["model_registry_namespace"]).delete(wait=True)
            # create the original namespace object again, so that we can wait for it to be created first
            original_namespace = Namespace(name=original_namespace_name, wait_for_resource=True)
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
) -> Generator[UserTestSession | None, None, None]:
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
    is_byoidc: bool, original_user: str, user_credentials_rbac: dict[str, str]
) -> Generator[UserTestSession | None, None, None]:
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
) -> Generator[Any, None, None]:
    if is_byoidc:
        yield
    else:
        # Get current providers and add the new one
        oauth = OAuth(name="cluster")
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
        wait_for_oauth_openshift_deployment()
        LOGGER.info(f"Added IDP {user_credentials_rbac['idp_name']} to OAuth configuration")
        yield
        identity_providers_patch.restore()
        wait_for_oauth_openshift_deployment()


@pytest.fixture(scope="module")
def user_credentials_rbac(
    is_byoidc: bool,
) -> dict[str, str]:
    if is_byoidc:
        byoidc_creds = get_byoidc_user_credentials(username="mr-non-admin")
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
