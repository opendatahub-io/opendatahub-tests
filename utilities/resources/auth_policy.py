# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource


class AuthPolicy(NamespacedResource):
    """
    AuthPolicy enables authentication and authorization for service workloads in a Gateway API network
    """

    api_group: str = NamespacedResource.ApiGroup.KUADRANT_IO

    def __init__(
        self,
        defaults: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
        patterns: dict[str, Any] | None = None,
        rules: dict[str, Any] | None = None,
        target_ref: dict[str, Any] | None = None,
        when: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            defaults (dict[str, Any]): Rules to apply as defaults. Can be overridden by more specific policiy
              rules lower in the hierarchy and by less specific policy
              overrides. Use one of: defaults, overrides, or bare set of policy
              rules (implicit defaults).

            overrides (dict[str, Any]): Rules to apply as overrides. Override all policy rules lower in the
              hierarchy. Can be overridden by less specific policy overrides.
              Use one of: defaults, overrides, or bare set of policy rules
              (implicit defaults).

            patterns (dict[str, Any]): Named sets of patterns that can be referred in `when` conditions and
              in pattern-matching authorization policy rules.

            rules (dict[str, Any]): The auth rules of the policy. See Authorino's AuthConfig CRD for more
              details.

            target_ref (dict[str, Any]): Reference to the object to which this policy applies.

            when (list[Any]): Overall conditions for the policy to be enforced. If omitted, the
              policy will be enforced at all requests to the protected routes.
              If present, all conditions must match for the policy to be
              enforced.

        """
        super().__init__(**kwargs)

        self.defaults = defaults
        self.overrides = overrides
        self.patterns = patterns
        self.rules = rules
        self.target_ref = target_ref
        self.when = when

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.target_ref is None:
                raise MissingRequiredArgumentError(argument="self.target_ref")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["targetRef"] = self.target_ref

            if self.defaults is not None:
                _spec["defaults"] = self.defaults

            if self.overrides is not None:
                _spec["overrides"] = self.overrides

            if self.patterns is not None:
                _spec["patterns"] = self.patterns

            if self.rules is not None:
                _spec["rules"] = self.rules

            if self.when is not None:
                _spec["when"] = self.when

    # End of generated code
