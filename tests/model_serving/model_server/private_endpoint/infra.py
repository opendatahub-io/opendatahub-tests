from typing import Generator, Dict, Optional

from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.project_project_openshift_io import Project


def create_ns(
    name: str,
    labels: Optional[Dict[str, str]] = None,
    admin_client: DynamicClient = None,
    teardown: bool = True,
    delete_timeout: int = 6 * 10,
) -> Generator[Namespace,None,None]:
    with Namespace(
        client=admin_client,
        name=name,
        label=labels,
        teardown=teardown,
        delete_timeout=delete_timeout,
    ) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=2 * 10)
        yield ns