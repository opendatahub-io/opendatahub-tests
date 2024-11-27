from typing import Generator, Dict, Optional
from contextlib import contextmanager

from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace


@contextmanager
def create_ns(
    name: str,
    admin_client: DynamicClient,
    teardown: bool = True,
    delete_timeout: int = 6 * 10,
    labels: Optional[Dict[str, str]] = None,
) -> Generator[Namespace, None, None]:
    with Namespace(
        client=admin_client,
        name=name,
        label=labels,
        teardown=teardown,
        delete_timeout=delete_timeout,
    ) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=2 * 10)
        yield ns
