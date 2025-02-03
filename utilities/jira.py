import os
from functools import cache

from jira import JIRA
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@cache
def get_jira_connection() -> JIRA:
    """
    Get Jira connection.

    Returns:
        JIRA: Jira connection.

    """
    return JIRA(
        token_auth=os.getenv("PYTEST_JIRA_TOKEN"),
        options={"server": os.getenv("PYTEST_JIRA_URL")},
    )


def is_jira_open(jira_id: str) -> bool:
    """
    Check if Jira issue is open.

    Args:
        jira_id (str): Jira issue id.

    Returns:
        bool: True if Jira issue is open.

    """
    jira_status = get_jira_connection().issue(id=jira_id).fields.status.name.lower()

    if jira_status not in ("testing", "resolved", "closed"):
        LOGGER.info(f"Jira {jira_id}: status is {jira_status}")
        return True

    return False
