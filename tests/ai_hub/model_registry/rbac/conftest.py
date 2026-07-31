import base64
import json
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import structlog
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient

from tests.ai_hub.constants import (
    KUBERBACPROXY_STR,
    MR_INSTANCE_NAME,
    NUM_MR_INSTANCES,
)
from tests.ai_hub.model_registry.rbac.group_utils import create_group
from tests.ai_hub.model_registry.rbac.utils import create_role_binding, grant_mr_access, revoke_mr_access
from tests.ai_hub.utils import (
    get_byoidc_user_credentials,
    get_endpoint_from_mr_service,
    get_mr_service_by_label,
    get_mr_user_token,
)
from utilities.constants import Protocols
from utilities.infra import get_openshift_token, login_with_user_password
from utilities.openshift_resources.deployment import Deployment
from utilities.openshift_resources.group import Group
from utilities.openshift_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from utilities.openshift_resources.persistent_volume_claim import PersistentVolumeClaim
from utilities.openshift_resources.role import Role
from utilities.openshift_resources.role_binding import RoleBinding
from utilities.openshift_resources.secret import Secret
from utilities.openshift_resources.service import Service
from utilities.user_utils import UserTestSession, get_byoidc_issuer_url

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="function")
def add_user_to_group(
    test_idp_user: UserTestSession,
) -> Generator[str]:
    """
    Fixture to create a group and add a test user to it.
    Uses create_group context manager to ensure proper cleanup.

    Args:
        admin_client: The admin client for accessing the cluster
        test_idp_user_session: The test user session containing user information

    Yields:
        str: The name of the created group
    """
    group_name = "test-model-registry-group"
    with create_group(
        group_name=group_name,
        users=[test_idp_user.username],
    ) as group_name:
        yield group_name


@pytest.fixture(scope="function")
def byoidc_entra_group_role_bindings(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> Generator[list[RoleBinding]]:
    """RoleBindings mapping Entra group UUIDs from mr-user1's token to the model registry role."""
    mr_user1_creds = get_byoidc_user_credentials(client=admin_client, username="mr-user1")
    token = get_mr_user_token(admin_client=admin_client, user_credentials_rbac=mr_user1_creds)

    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    claims = json.loads(base64.b64decode(payload))
    entra_groups: list[str] = claims["groups"]

    role_name = f"registry-user-{MR_INSTANCE_NAME}"
    with ExitStack() as stack:
        role_bindings = [
            stack.enter_context(
                RoleBinding(
                    namespace=model_registry_namespace,
                    name=f"{MR_INSTANCE_NAME}-entra-group-{idx}",
                    role_ref={
                        "kind": "Role",
                        "name": role_name,
                        "apiGroup": "rbac.authorization.k8s.io",
                    },
                    subjects=[
                        {
                            "kind": "Group",
                            "name": group_uuid,
                            "apiGroup": "rbac.authorization.k8s.io",
                        }
                    ],
                )
            )
            for idx, group_uuid in enumerate(entra_groups)
        ]
        LOGGER.info(f"Created {len(role_bindings)} RoleBindings for Entra groups: {entra_groups}")
        yield role_bindings


@pytest.fixture(scope="function")
def model_registry_group_with_user(
    request: FixtureRequest,
    is_byoidc: bool,
    admin_client: DynamicClient,
    test_idp_user: UserTestSession,
) -> Generator[Group | list[RoleBinding]]:
    """
    Fixture to manage a test user in a specified group.
    For Microsoft Entra BYOIDC, creates RoleBindings for Entra group UUIDs.
    For other BYOIDC providers, group membership comes from the OIDC token.
    For non-BYOIDC, adds the user to an OpenShift Group.

    Yields:
        Group or list[RoleBinding] depending on the authentication type.
    """
    if is_byoidc:
        issuer_url = get_byoidc_issuer_url(admin_client=admin_client)
        if "microsoftonline" in issuer_url:
            yield request.getfixturevalue(argname="byoidc_entra_group_role_bindings")
        else:
            yield
    else:
        group_name = f"{MR_INSTANCE_NAME}-users"
        group = Group(
            name=group_name,
            wait_for_resource=True,
        )

        # Add user to group
        with group.patch_and_restore(
            patch={
                "metadata": {"name": group_name},
                "users": [test_idp_user.username],
            }
        ):
            LOGGER.info(f"Added user {test_idp_user.username} to {group_name} group")
            yield group


@pytest.fixture(scope="function")
def created_role_binding_group(
    model_registry_namespace: str,
    mr_access_role: Role,
    test_idp_user: UserTestSession,
    add_user_to_group: str,
) -> Generator[RoleBinding]:
    yield from create_role_binding(
        model_registry_namespace=model_registry_namespace,
        name="test-model-registry-group-edit",
        mr_access_role=mr_access_role,
        subjects_kind="Group",
        subjects_name=add_user_to_group,
    )


@pytest.fixture(scope="function")
def created_role_binding_user(
    is_byoidc: bool,
    model_registry_namespace: str,
    mr_access_role: Role,
    user_credentials_rbac: dict[str, str],
) -> Generator[RoleBinding]:
    # Determine the username to use without mutating the shared fixture
    username = "mr-non-admin" if is_byoidc else user_credentials_rbac["username"]
    LOGGER.info(f"Using user {username}")
    yield from create_role_binding(
        model_registry_namespace=model_registry_namespace,
        name="test-model-registry-access",
        mr_access_role=mr_access_role,
        subjects_kind="User",
        subjects_name=username,
    )


# =============================================================================
# RESOURCE FIXTURES PARMETRIZED
# =============================================================================
@pytest.fixture(scope="class")
def db_secret_parametrized(request: FixtureRequest, teardown_resources: bool) -> Generator[list[Secret], Any, Any]:
    """Create DB Secret parametrized"""
    with ExitStack() as stack:
        secrets = [
            stack.enter_context(
                Secret(
                    **param,
                    teardown=teardown_resources,
                )
            )
            for param in request.param
        ]
        yield secrets


@pytest.fixture(scope="class")
def db_pvc_parametrized(
    request: FixtureRequest, teardown_resources: bool
) -> Generator[list[PersistentVolumeClaim], Any, Any]:
    """Create DB PVC parametrized"""
    with ExitStack() as stack:
        pvc = [
            stack.enter_context(
                PersistentVolumeClaim(
                    **param,
                    teardown=teardown_resources,
                )
            )
            for param in request.param
        ]
        yield pvc


@pytest.fixture(scope="class")
def db_service_parametrized(request: FixtureRequest, teardown_resources: bool) -> Generator[list[Service], Any, Any]:
    """Create DB Service parametrized"""
    with ExitStack() as stack:
        services = [
            stack.enter_context(
                Service(
                    **param,
                    teardown=teardown_resources,
                )
            )
            for param in request.param
        ]
        yield services


@pytest.fixture(scope="class")
def db_deployment_parametrized(
    request: FixtureRequest, teardown_resources: bool
) -> Generator[list[Deployment], Any, Any]:
    """Create DB Deployment parametrized"""
    with ExitStack() as stack:
        deployments = [
            stack.enter_context(
                Deployment(
                    **param,
                    teardown=teardown_resources,
                )
            )
            for param in request.param
        ]

        for deployment in deployments:
            deployment.wait_for_replicas()

        yield deployments


@pytest.fixture(scope="class")
def model_registry_instance_parametrized(
    request: FixtureRequest, teardown_resources: bool
) -> Generator[list[ModelRegistry], Any, Any]:
    """Create Model Registry instance parametrized"""
    if len(request.param) != NUM_MR_INSTANCES:
        raise ValueError(f"Expected {NUM_MR_INSTANCES} MR instances, but got {len(request.param)}")

    with ExitStack() as stack:
        model_registry_instances = []
        mr_instances = [stack.enter_context(ModelRegistry(**param)) for param in request.param]
        for mr_instance in mr_instances:
            mr_instance.wait_for_condition(condition="Available", status="True")
            mr_instance.wait_for_condition(condition=KUBERBACPROXY_STR, status="True")
            model_registry_instances.append(mr_instance)

        LOGGER.info(
            f"Created {len(model_registry_instances)} MR instances: {[mr.name for mr in model_registry_instances]}"
        )
        yield model_registry_instances


@pytest.fixture(scope="class")
def mr_endpoints_parametrized(
    model_registry_namespace: str,
    model_registry_instance_parametrized: list[ModelRegistry],
) -> list[dict[str, Any]]:
    """Collect MR service endpoints as admin before login_as_test_user switches context."""
    mr_data = []
    for mr_instance in model_registry_instance_parametrized:
        service = get_mr_service_by_label(
            namespace_name=model_registry_namespace,
            mr_instance=mr_instance,
        )
        endpoint = get_endpoint_from_mr_service(svc=service, protocol=Protocols.REST)
        mr_data.append({"instance": mr_instance, "endpoint": endpoint, "name": mr_instance.name})
    return mr_data


@pytest.fixture(scope="class")
def test_user_token(
    is_byoidc: bool,
    admin_client: DynamicClient,
    user_credentials_rbac: dict[str, str],
    test_idp_user: UserTestSession,
    api_server_url: str,
    original_user: str,
) -> str:
    """Get the test user's token without permanently switching oc context."""
    if is_byoidc:
        return get_mr_user_token(admin_client=admin_client, user_credentials_rbac=user_credentials_rbac)

    login_with_user_password(api_address=api_server_url, user=test_idp_user.username, password=test_idp_user.password)
    try:
        token = get_openshift_token()
    finally:
        login_with_user_password(api_address=api_server_url, user=original_user)
    return token


@pytest.fixture()
def granted_mr_instance_access(
    is_byoidc: bool,
    mr_endpoints_parametrized: list[dict[str, Any]],
    model_registry_namespace: str,
    user_credentials_rbac: dict[str, str],
) -> Generator[Generator[int], Any, Any]:
    """Yield a generator that grants access to each MR instance one at a time, revoking on each advance."""
    rbac_username = "mr-non-admin" if is_byoidc else user_credentials_rbac["username"]
    granted_instances: list[str] = []

    def _grant_each() -> Generator[int]:
        for idx, mr_data in enumerate(mr_endpoints_parametrized):
            grant_mr_access(
                user=rbac_username,
                mr_instance_name=mr_data["name"],
                model_registry_namespace=model_registry_namespace,
            )
            granted_instances.append(mr_data["name"])
            yield idx
            revoke_mr_access(
                user=rbac_username,
                mr_instance_name=mr_data["name"],
                model_registry_namespace=model_registry_namespace,
            )
            granted_instances.remove(mr_data["name"])

    yield _grant_each()

    for instance_name in granted_instances:
        revoke_mr_access(
            user=rbac_username,
            mr_instance_name=instance_name,
            model_registry_namespace=model_registry_namespace,
        )


@pytest.fixture()
def skip_test_on_byoidc(is_byoidc: bool) -> None:
    if is_byoidc:
        pytest.skip(reason="This test is meant to skip on byoidc.")
