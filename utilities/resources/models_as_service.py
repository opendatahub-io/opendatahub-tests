# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.resource import Resource


class ModelsAsService(Resource):
    """
    ModelsAsService is the Schema for the modelsasservice API
    """

    api_group: str = Resource.ApiGroup.COMPONENTS_PLATFORM_OPENDATAHUB_IO

    def __init__(
        self,
        gateway_ref: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            gateway_ref (dict[str, Any]): GatewayRef specifies which Gateway (Gateway API) to use for exposing
              model endpoints. If omitted, defaults to openshift-ingress/maas-
              default-gateway.

        """
        super().__init__(**kwargs)

        self.gateway_ref = gateway_ref

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.gateway_ref is not None:
                _spec["gatewayRef"] = self.gateway_ref

    # End of generated code
