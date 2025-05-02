import pytest
from typing import Self
import shlex
import os
from simple_logger.logger import get_logger
from contextlib import nullcontext
from pyhelper_utils.shell import run_command


from tests.model_registry.constants import MR_INSTANCE_NAME
from tests.model_registry.utils import get_endpoint_from_mr_service, get_model_registry_service
from kubernetes.dynamic import DynamicClient
from utilities.constants import DscComponents, Protocols
from mr_openapi.exceptions import ForbiddenException
from model_registry import ModelRegistry as ModelRegistryClient

LOGGER = get_logger(name=__name__)
TEST_NAMESPACE = "model-registry-test-ns"


def get_token(user_name: str, password: str, admin_client: DynamicClient) -> str:
    current_context = run_command(command=["oc", "config", "current-context"])[1].strip()

    command: str = (
        f"oc login  --insecure-skip-tls-verify=true {admin_client.configuration.host} -u {user_name} -p {password}"
    )
    run_command(command=shlex.split(command), hide_log_command=True)

    token = run_command(command=["oc", "whoami", "-t"])[1].strip()

    run_command(command=["oc", "config", "use-context", current_context])

    return token


def assert_mr_client(user_token, admin_client, context):
    """
    Initiate MR client
    """
    svc = get_model_registry_service(admin_client=admin_client, namespace=TEST_NAMESPACE)
    server, port = get_endpoint_from_mr_service(svc, Protocols.REST).split(":")

    with context:
        ModelRegistryClient(
            server_address=f"https://{server}",
            port=int(port),
            author="opendatahub-test",
            user_token=user_token,
            is_secure=False,
        )


@pytest.fixture(scope="session", autouse=True)
def check_env_vars():
    missing = []
    missing.extend(var for var in ["ADMIN_PASSWORD", "NON_ADMIN_PASSWORD"] if not os.environ.get(var))
    if missing:
        pytest.fail(f"Required environment variables not set: {', '.join(missing)}")


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [
        pytest.param({
            "component_patch": {
                DscComponents.MODELREGISTRY: {
                    "managementState": DscComponents.ManagementState.MANAGED,
                    "registriesNamespace": TEST_NAMESPACE,
                },
            }
        })
    ],
    indirect=True,
)
class TestUserPermission:
    """
    Test Role-based access control
    """

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "user_name, password, context_manager",
        [
            ("htpasswd-cluster-admin-user", os.environ.get("ADMIN_PASSWORD"), nullcontext()),
            ("ldap-user1", os.environ.get("NON_ADMIN_PASSWORD"), pytest.raises(ForbiddenException)),
        ],
    )
    def test_user_permission(
        self: Self,
        updated_dsc_component_state_scope_class,
        model_registry_instance,
        admin_client: DynamicClient,
        user_name,
        password,
        context_manager,
    ):
        """
        Cluster admin user should be able to access the model registry,
        other users should not be able to access the model registry
        """
        assert model_registry_instance.name == MR_INSTANCE_NAME
        user_token = get_token(user_name=user_name, password=password, admin_client=admin_client)

        assert_mr_client(user_token=user_token, admin_client=admin_client, context=context_manager)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "user_name, password",
        [
            ("ldap-user1", os.environ.get("NON_ADMIN_PASSWORD")),
        ],
    )
    def test_user_added_to_group(
        self: Self,
        updated_dsc_component_state_scope_class,
        model_registry_instance,
        admin_client: DynamicClient,
        user_name,
        password,
    ):
        """
        User can initiate MR only when they are added to the model-registry-users group
        """
        assert model_registry_instance.name == MR_INSTANCE_NAME

        user_token = get_token(user_name=user_name, password=password, admin_client=admin_client)

        LOGGER.info("User has no access to MR")
        assert_mr_client(user_token=user_token, admin_client=admin_client, context=pytest.raises(ForbiddenException))

        LOGGER.info("Add user to the model registry users group")
        run_command(command=["oc", "adm", "groups", "add-users", "model-registry-users", user_name])

        user_token = get_token(user_name=user_name, password=password, admin_client=admin_client)

        LOGGER.info("User has access to MR")
        assert_mr_client(user_token=user_token, admin_client=admin_client, context=nullcontext())
