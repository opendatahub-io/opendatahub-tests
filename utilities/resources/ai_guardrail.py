# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource

from utilities.constants import ApiGroups


class AIGuardrail(NamespacedResource):
    """
    AIGuardrail is a reusable guardrail policy that MaaS resources attach by name.
    It binds an ordered list of checks to a NemoGuardrails provider.
    """

    api_group: str = ApiGroups.AIGATEWAY_OPENDATAHUB_IO

    def __init__(
        self,
        checks: list[Any] | None = None,
        provider: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            checks (list[Any]): Checks is the ordered list of checks this policy offers. Each check
              names a configId loaded by the provider and the phases (Input,
              Output) it runs in.

            provider (dict[str, Any]): Provider selects the NemoGuardrails server that evaluates the
              checks and the per-call timeout.

        """
        super().__init__(**kwargs)

        self.checks = checks
        self.provider = provider

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.checks is None:
                raise MissingRequiredArgumentError(argument="self.checks")

            if self.provider is None:
                raise MissingRequiredArgumentError(argument="self.provider")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["checks"] = self.checks
            _spec["provider"] = self.provider

    # End of generated code
