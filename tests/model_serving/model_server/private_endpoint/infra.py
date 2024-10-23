from typing import Generator, Dict, Optional

from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.project_project_openshift_io import Project
from ocp_resources.project_request import ProjectRequest

TIMEOUT_2MIN = 2 * 10
TIMEOUT_6MIN = 6 * 10


def create_ns(
    name: str,
    unprivileged_client: DynamicClient = None,
    labels: Optional[Dict[str, str]] = None,
    admin_client: DynamicClient = None,
    teardown: bool = True,
    delete_timeout: int = TIMEOUT_6MIN,
) -> Generator[Namespace] | Generator[Project]:
    if not unprivileged_client:
        with Namespace(
            client=admin_client,
            name=name,
            label=labels,
            teardown=teardown,
            delete_timeout=delete_timeout,
        ) as ns:
            ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=TIMEOUT_2MIN)
            yield ns
    else:
        with ProjectRequest(name=name, client=unprivileged_client, teardown=teardown):
            project = Project(
                name=name,
                client=unprivileged_client,
                label=labels,
                teardown=teardown,
                delete_timeout=delete_timeout,
            )
            project.wait_for_status(project.Status.ACTIVE, timeout=TIMEOUT_2MIN)
            yield project
