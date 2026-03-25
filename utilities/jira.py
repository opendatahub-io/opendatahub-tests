import os
import re
from functools import cache
from typing import Any

import structlog
from jira import JIRA, JIRAError
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.exceptions import MissingResourceError
from packaging.version import Version
from pytest_testconfig import config as py_config
from requests.exceptions import ConnectionError as RequestsConnectionError
from urllib3.exceptions import NewConnectionError

LOGGER = structlog.get_logger(name=__name__)

JIRA_CLOSED_STATUSES = ("closed", "resolved", "testing")


@cache
def get_jira_connection() -> JIRA:
    """
    Get Jira connection.

    Returns:
        JIRA: Jira connection.

    """
    return JIRA(
        server=os.getenv("PYTEST_JIRA_URL"),
        basic_auth=(os.getenv("PYTEST_JIRA_USERNAME"), os.getenv("PYTEST_JIRA_TOKEN")),
    )


@cache
def get_jira_issue_fields(jira_id: str) -> Any:
    """
    Get Jira issue fields (status and fixVersions).

    Args:
        jira_id: Jira issue id (e.g. "RHOAIENG-52129").

    Returns:
        Jira issue fields object with status and fixVersions.
    """
    return get_jira_connection().issue(id=jira_id, fields="status, fixVersions").fields


def is_jira_open(jira_id: str, admin_client: DynamicClient) -> bool:  # skip-unused-code
    """
    Check if Jira issue is open.

    Args:
        jira_id (str): Jira issue id.
        admin_client (DynamicClient): DynamicClient object

    Returns:
        bool: True if Jira issue is open.

    """
    if is_jira_issue_open(jira_id=jira_id):
        return True

    else:
        jira_fields = get_jira_issue_fields(jira_id=jira_id)
        jira_fix_versions: list[Version] = [
            Version(_fix_version.group())
            for fix_version in jira_fields.fixVersions
            if (_fix_version := re.search(r"\d+\.\d+(?:\.\d+)?", fix_version.name))
        ]

        if not jira_fix_versions:
            raise ValueError(f"Jira {jira_id}: closed/resolved but does not have fix version(s)")

        operator_version: str = ""
        for csv in ClusterServiceVersion.get(client=admin_client, namespace=py_config["applications_namespace"]):
            if re.match("rhods|opendatahub", csv.name):
                operator_version = csv.instance.spec.version
                break

        if not operator_version:
            raise MissingResourceError("Operator ClusterServiceVersion not found")

        csv_version = Version(version=operator_version)
        if all(csv_version < fix_version for fix_version in jira_fix_versions):
            LOGGER.info(
                f"Bug is open: Jira {jira_id}: fix versions {jira_fix_versions}, operator version is {operator_version}"
            )
            return True

    return False


@cache
def is_jira_issue_open(jira_id: str) -> bool:  # skip-unused-code
    """
    Check if a Jira issue is open based on its status.

    Args:
        jira_id: Jira issue id (e.g. "RHOAIENG-52129").

    Returns:
        True if the issue status is not in closed/resolved/testing.
        True if Jira is unreachable (assumes issue is open).
    """
    try:
        jira_status = get_jira_issue_fields(jira_id=jira_id).status.name.lower()
    except NewConnectionError, JIRAError, RequestsConnectionError:
        LOGGER.warning(f"Failed to get Jira issue {jira_id}, assuming it is open")
        return True

    is_open = jira_status not in JIRA_CLOSED_STATUSES
    LOGGER.info(f"Jira {jira_id}: status={jira_status}, open={is_open}")
    return is_open
