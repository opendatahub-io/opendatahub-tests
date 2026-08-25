# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md

from typing import Any

from ocp_resources.resource import NamespacedResource


class MCPServer(NamespacedResource):
    """MCPServer is the Schema for the mcpservers API."""

    api_group: str = "mcp.x-k8s.io"
    kind: str = "MCPServer"

    def __init__(
        self,
        source: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.source = source
        self.config = config
        self.runtime = runtime

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.source is not None:
                _spec["source"] = self.source

            if self.config is not None:
                _spec["config"] = self.config

            if self.runtime is not None:
                _spec["runtime"] = self.runtime

    # End of generated code
