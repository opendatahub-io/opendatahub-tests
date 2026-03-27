# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import Resource


class Auth(Resource):
    """
    Auth is the Schema for the auths API
    """

    api_group: str = Resource.ApiGroup.SERVICES_PLATFORM_OPENDATAHUB_IO

    def __init__(
        self,
        admin_groups: list[Any] | None = None,
        allowed_groups: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            admin_groups (list[Any]): No field description from API

            allowed_groups (list[Any]): No field description from API

        """
        super().__init__(**kwargs)

        self.admin_groups = admin_groups
        self.allowed_groups = allowed_groups

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.admin_groups is None:
                raise MissingRequiredArgumentError(argument="self.admin_groups")

            if self.allowed_groups is None:
                raise MissingRequiredArgumentError(argument="self.allowed_groups")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["adminGroups"] = self.admin_groups
            _spec["allowedGroups"] = self.allowed_groups

    # End of generated code
