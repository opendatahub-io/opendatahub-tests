from contextlib import contextmanager
from typing import Any, Dict, Optional

from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

TIMEOUT_2MIN = 2 * 60


@contextmanager
def create_ns(
    client: DynamicClient,
    name: str,
    labels: Optional[Dict[str, Any]] = None,
    wait_for_resource: bool = True,
) -> Namespace:
    with Namespace(
        client=client,
        name=name,
        label=labels,
        wait_for_resource=wait_for_resource,
    ) as ns:
        yield ns
