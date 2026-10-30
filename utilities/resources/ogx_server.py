# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/class_generator/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource


class OgxServer(NamespacedResource):
    """
    OGXServer is the Schema for the ogxservers API.
    """

    api_group: str = "ogx.io"
    kind: str = "OGXServer"

    def __init__(
        self,
        base_config: dict[str, Any] | None = None,
        disabled_apis: list[Any] | None = None,
        distribution: dict[str, Any] | None = None,
        monitoring: dict[str, Any] | None = None,
        network: dict[str, Any] | None = None,
        override_config: dict[str, Any] | None = None,
        praxis_mode: dict[str, Any] | None = None,
        providers: dict[str, Any] | None = None,
        registry_refresh_interval_seconds: int | None = None,
        resources: dict[str, Any] | None = None,
        storage: dict[str, Any] | None = None,
        tls: dict[str, Any] | None = None,
        workload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            base_config (dict[str, Any]): BaseConfig references a ConfigMap key containing the base config.yaml
              used as the starting point for declarative config generation. When
              set, this takes precedence over OCI label resolution. Mutually
              exclusive with overrideConfig. The ConfigMap must be in the same
              namespace as the OGXServer and must have the label ogx.io/watch:
              "true".

            disabled_apis (list[Any]): DisabledAPIs lists API names to remove from the generated config.
              Mutually exclusive with overrideConfig.

            distribution (dict[str, Any]): Distribution identifies the OGX distribution to deploy.

            monitoring (dict[str, Any]): Monitoring configures Prometheus monitoring and observability.

            network (dict[str, Any]): Network defines network access controls.

            override_config (dict[str, Any]): OverrideConfig references a ConfigMap key containing a full
              config.yaml override. Mutually exclusive with providers,
              resources, storage, disabledAPIs, and baseConfig. The ConfigMap
              must be in the same namespace as the OGXServer and must have the
              label ogx.io/watch: "true".

            praxis_mode (dict[str, Any]): PraxisMode configures integration with an existing Praxis instance
              that acts as gateway for this OGX server.

            providers (dict[str, Any]): Providers configures providers by API type. Mutually exclusive with
              overrideConfig.

            registry_refresh_interval_seconds (int): RegistryRefreshIntervalSeconds configures how often the server
              refreshes its model registry, in seconds. When omitted, the
              server's built-in default is used.

            resources (dict[str, Any]): Resources declares models to register. Mutually exclusive with
              overrideConfig.

            storage (dict[str, Any]): Storage configures state storage backends (KV and SQL). Mutually
              exclusive with overrideConfig.

            tls (dict[str, Any]): TLS configures outbound TLS trust anchors and client identity for
              connections to providers and backends.

            workload (dict[str, Any]): Workload consolidates Kubernetes deployment settings.

        """
        super().__init__(**kwargs)

        self.base_config = base_config
        self.disabled_apis = disabled_apis
        self.distribution = distribution
        self.monitoring = monitoring
        self.network = network
        self.override_config = override_config
        self.praxis_mode = praxis_mode
        self.providers = providers
        self.registry_refresh_interval_seconds = registry_refresh_interval_seconds
        self.resources = resources
        self.storage = storage
        self.tls = tls
        self.workload = workload

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.distribution is None:
                raise MissingRequiredArgumentError(argument="self.distribution")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["distribution"] = self.distribution

            if self.base_config is not None:
                _spec["baseConfig"] = self.base_config

            if self.disabled_apis is not None:
                _spec["disabledAPIs"] = self.disabled_apis

            if self.monitoring is not None:
                _spec["monitoring"] = self.monitoring

            if self.network is not None:
                _spec["network"] = self.network

            if self.override_config is not None:
                _spec["overrideConfig"] = self.override_config

            if self.praxis_mode is not None:
                _spec["praxisMode"] = self.praxis_mode

            if self.providers is not None:
                _spec["providers"] = self.providers

            if self.registry_refresh_interval_seconds is not None:
                _spec["registryRefreshIntervalSeconds"] = self.registry_refresh_interval_seconds

            if self.resources is not None:
                _spec["resources"] = self.resources

            if self.storage is not None:
                _spec["storage"] = self.storage

            if self.tls is not None:
                _spec["tls"] = self.tls

            if self.workload is not None:
                _spec["workload"] = self.workload

    # End of generated code
