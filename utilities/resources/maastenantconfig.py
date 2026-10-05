# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.resource import NamespacedResource


class MaasTenantConfig(NamespacedResource):
    """
    MaasTenantConfig is the namespace-scoped MaaS-owned tenant configuration.

    AITenant bootstrap creates this object (e.g. default-tenant) instead of the
    legacy Tenant CR. Conditions such as Ready, DependenciesAvailable,
    MaaSPrerequisitesAvailable, and DeploymentsAvailable live here.
    """

    api_group: str = NamespacedResource.ApiGroup.MAAS_OPENDATAHUB_IO

    def __init__(
        self,
        api_keys: dict[str, Any] | None = None,
        telemetry: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            api_keys (dict[str, Any]): APIKeys contains configuration for API key management.
            telemetry (dict[str, Any]): Telemetry contains configuration for metrics collection.
        """
        super().__init__(**kwargs)

        self.api_keys = api_keys
        self.telemetry = telemetry

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.api_keys is not None:
                _spec["apiKeys"] = self.api_keys

            if self.telemetry is not None:
                _spec["telemetry"] = self.telemetry
