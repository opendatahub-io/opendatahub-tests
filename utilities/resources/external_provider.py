# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource

from utilities.constants import ApiGroups


class ExternalProvider(NamespacedResource):
    """
    ExternalProvider defines a connection to an external LLM provider (endpoint + credentials).
    Multiple ExternalModel resources can reference the same ExternalProvider.
    """

    api_group: str = ApiGroups.INFERENCE_OPENDATAHUB_IO

    def __init__(
        self,
        auth: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        endpoint: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            auth (dict[str, Any]): Auth configures how to authenticate with the provider.

            config (dict[str, Any]): Config holds provider-specific configuration as key-value pairs. e.g.,
              Vertex AI: {"project": "my-project", "location": "us-central1"}.

            endpoint (str): Endpoint is the FQDN of the external provider (no scheme or path).
              e.g. "api.openai.com", "bedrock.amazonaws.com".

            provider (str): Provider identifies the API type for this provider. e.g. "openai",
              "anthropic", "azure", "aws-bedrock", "vertex".

        """
        super().__init__(**kwargs)

        self.auth = auth
        self.config = config
        self.endpoint = endpoint
        self.provider = provider

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.auth is None:
                raise MissingRequiredArgumentError(argument="self.auth")

            if self.endpoint is None:
                raise MissingRequiredArgumentError(argument="self.endpoint")

            if self.provider is None:
                raise MissingRequiredArgumentError(argument="self.provider")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["auth"] = self.auth
            _spec["endpoint"] = self.endpoint
            _spec["provider"] = self.provider

            if self.config is not None:
                _spec["config"] = self.config

    # End of generated code
