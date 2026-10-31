# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource

from utilities.constants import ApiGroups


class ExternalModel(NamespacedResource):
    """
    ExternalModel defines a client-facing model that maps to one or more
    external providers. The model name clients use in requests is determined
    by spec.modelName (if set) or metadata.name (default).
    """

    api_group: str = ApiGroups.INFERENCE_OPENDATAHUB_IO

    def __init__(
        self,
        external_provider_refs: list[Any] | None = None,
        gateway_refs: list[dict[str, Any]] | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            external_provider_refs (list[Any]): ExternalProviderRefs maps this model to one or more external
              providers. Each entry specifies the provider specific details.

            gateway_refs (list[dict[str, Any]]): GatewayRefs attaches this model to explicit tenant Gateways
              (ai-gateway-controller contract). Each entry is a ``{"name": ..., "namespace": ...}``
              reference; None keeps the legacy shape (tenant-local resolution).

            model_name (str): ModelName is the client-facing model name used in inference request
              bodies. Clients send this value in the "model" field of chat
              completion requests. Defaults to metadata.name if not set. Use
              this field when the desired model name contains characters not
              allowed in Kubernetes resource names (dots, colons, slashes,
              uppercase).

        """
        super().__init__(**kwargs)

        self.external_provider_refs = external_provider_refs
        self.gateway_refs = gateway_refs
        self.model_name = model_name

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.external_provider_refs is None:
                raise MissingRequiredArgumentError(argument="self.external_provider_refs")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["externalProviderRefs"] = self.external_provider_refs

            if self.gateway_refs is not None:
                _spec["gatewayRefs"] = self.gateway_refs

            if self.model_name is not None:
                _spec["modelName"] = self.model_name

    # End of generated code
