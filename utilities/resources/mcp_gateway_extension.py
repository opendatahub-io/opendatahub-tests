# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md

from ocp_resources.resource import NamespacedResource

from utilities.constants import ApiGroups


class MCPGatewayExtension(NamespacedResource):
    """
    MCPGatewayExtension is the Schema for the mcpgatewayextensions API (mcp.kuadrant.io/v1alpha1).
    """

    api_group: str = ApiGroups.MCP_KUADRANT_IO

    # End of generated code
