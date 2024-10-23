from ocp_resources.namespace import Namespace
from ocp_resources.project_project_openshift_io import Project
from ocp_resources.project_request import ProjectRequest

TIMEOUT_2MIN = 2 * 10
TIMEOUT_6MIN = 6 * 10


def create_ns(
    name,
    unprivileged_client=None,
    labels=None,
    admin_client=None,
    teardown=True,
    delete_timeout=TIMEOUT_6MIN,
):
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