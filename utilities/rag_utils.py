from contextlib import contextmanager
from ocp_resources.resource import NamespacedResource
from kubernetes.dynamic import DynamicClient
from typing import Any, Dict, Generator


class LlamaStackDistribution(NamespacedResource):
    api_group: str = "llamastack.io"

    def __init__(self, replicas: int, server: Dict[str, Any], **kwargs: Any):
        """
        Args:
            kwargs: Keyword arguments to pass to the LlamaStackDistribution constructor
        """
        super().__init__(
            **kwargs,
        )
        self.replicas = replicas
        self.server = server

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]
            _spec["replicas"] = self.replicas
            _spec["server"] = self.server


@contextmanager
def create_llama_stack_distribution(
    client: DynamicClient,
    name: str,
    namespace: str,
    replicas: int,
    server: Dict[str, Any],
    teardown: bool = True,
) -> Generator[LlamaStackDistribution, Any, Any]:
    """
    Context manager to create and optionally delete a LLama Stack Distribution
    """
    with LlamaStackDistribution(
        client=client,
        name=name,
        namespace=namespace,
        replicas=replicas,
        server=server,
        teardown=teardown,
    ) as llama_stack_distribution:
        yield llama_stack_distribution
