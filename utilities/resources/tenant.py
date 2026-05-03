# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.resource import NamespacedResource


class Tenant(NamespacedResource):
    """
    Tenant is the namespace-scoped API for the MaaS platform tenant.
    """

    api_group: str = NamespacedResource.ApiGroup.MAAS_OPENDATAHUB_IO

    def __init__(
        self,
        api_keys: dict[str, Any] | None = None,
        external_oidc: dict[str, Any] | None = None,
        gateway_ref: dict[str, Any] | None = None,
        telemetry: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            api_keys (dict[str, Any]): APIKeys contains configuration for API key management.
            external_oidc (dict[str, Any]): ExternalOIDC configures an external OIDC identity provider.
            gateway_ref (dict[str, Any]): GatewayRef specifies which Gateway to use for model endpoints.
            telemetry (dict[str, Any]): Telemetry contains configuration for metrics collection.
        """
        super().__init__(**kwargs)

        self.api_keys = api_keys
        self.external_oidc = external_oidc
        self.gateway_ref = gateway_ref
        self.telemetry = telemetry

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.api_keys is not None:
                _spec["apiKeys"] = self.api_keys

            if self.external_oidc is not None:
                _spec["externalOIDC"] = self.external_oidc

            if self.gateway_ref is not None:
                _spec["gatewayRef"] = self.gateway_ref

            if self.telemetry is not None:
                _spec["telemetry"] = self.telemetry
