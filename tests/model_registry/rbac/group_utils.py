from contextlib import contextmanager
from typing import Generator
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient
from ocp_resources.group import Group

LOGGER = get_logger(name=__name__)


@contextmanager
def create_group(
    admin_client: DynamicClient,
    group_name: str,
    users: list[str] | None = None,
    wait_for_resource: bool = True,
) -> Generator[str, None, None]:
    """
    Factory function to create an OpenShift group with optional users.
    Uses context manager to ensure proper cleanup.

    Args:
        admin_client: The admin client to use for group operations
        group_name: Name of the group to create
        users: Optional list of usernames to add to the group
        wait_for_resource: Whether to wait for the group to be ready

    Yields:
        The group name
    """
    with Group(
        client=admin_client,
        name=group_name,
        users=users or [],
        wait_for_resource=wait_for_resource,
    ) as _:
        LOGGER.info(f"Group {group_name} created successfully.")
        yield group_name
