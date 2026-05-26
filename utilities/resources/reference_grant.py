# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource


class ReferenceGrant(NamespacedResource):
    """
        ReferenceGrant identifies kinds of resources in other namespaces that are
    trusted to reference the specified kinds of resources in the same namespace
    as the policy.

    Each ReferenceGrant can be used to represent a unique trust relationship.
    Additional Reference Grants can be used to add to the set of trusted
    sources of inbound references for the namespace they are defined within.

    All cross-namespace references in Gateway API (with the exception of cross-namespace
    Gateway-route attachment) require a ReferenceGrant.

    ReferenceGrant is a form of runtime verification allowing users to assert
    which cross-namespace object references are permitted. Implementations that
    support ReferenceGrant MUST NOT permit cross-namespace references which have
    no grant, and MUST respond to the removal of a grant by revoking the access
    that the grant allowed.
    """

    api_group: str = NamespacedResource.ApiGroup.GATEWAY_NETWORKING_K8S_IO

    def __init__(
        self,
        from_: list[Any] | None = None,
        to: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            from_ (list[Any]): From describes the trusted namespaces and kinds that can reference the
              resources described in "To". Each entry in this list MUST be
              considered to be an additional place that references can be valid
              from, or to put this another way, entries MUST be combined using
              OR.  Support: Core

                Note: Parameter renamed from &#39;from&#39; to avoid Python keyword conflict.
            to (list[Any]): To describes the resources that may be referenced by the resources
              described in "From". Each entry in this list MUST be considered to
              be an additional place that references can be valid to, or to put
              this another way, entries MUST be combined using OR.  Support:
              Core

        """
        super().__init__(**kwargs)

        self.from_ = from_
        self.to = to

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.from_ is None:
                raise MissingRequiredArgumentError(argument="self.from_")

            if self.to is None:
                raise MissingRequiredArgumentError(argument="self.to")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["from"] = self.from_
            _spec["to"] = self.to

    # End of generated code
