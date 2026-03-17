import os
import re
from functools import cache

from jira import JIRA
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.exceptions import MissingResourceError
from packaging.version import Version
from pytest_testconfig import config as py_config
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


def is_jira_open(jira_id: str, admin_client: DynamicClient) -> bool:
    """
    Check if Jira issue is open.

    Args:
        jira_id (str): Jira issue id.
        admin_client (DynamicClient): DynamicClient object

    Returns:
        bool: True if Jira issue is open.

    """
    jira_fields = get_jira_connection().issue(id=jira_id, fields="status, fixVersions").fields

    jira_status = jira_fields.status.name.lower()

    if jira_status not in ("testing", "resolved", "closed"):
        LOGGER.info(f"Jira {jira_id}: status is {jira_status}")
        return True

    else:
        # Check if the operator version in ClusterServiceVersion is greater than the jira fix version
        jira_fix_versions: list[Version] = [
            Version(_fix_version.group())
            for fix_version in jira_fields.fixVersions
            if (_fix_version := re.search(r"\d+\.\d+(?:\.\d+)?", fix_version.name))
        ]

        if not jira_fix_versions:
            raise ValueError(f"Jira {jira_id}: status is {jira_status} but does not have fix version(s)")

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
                f"Bug is open: Jira {jira_id}: status is {jira_status}, "
                f"fix versions {jira_fix_versions}, operator version is {operator_version}"
            )
            return True

    return False
