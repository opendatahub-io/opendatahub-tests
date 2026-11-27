# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/class_generator/README.md


from typing import Any

from ocp_resources.resource import Resource


class Monitoring(Resource):
    """
    Monitoring is the Schema for the monitorings API
    """

    api_group: str = Resource.ApiGroup.SERVICES_PLATFORM_OPENDATAHUB_IO

    def __init__(
        self,
        namespace: str | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            namespace (str): monitoring spec exposed to DSCI api Namespace for monitoring if it is
              enabled

        """
        super().__init__(**kwargs)

        self.namespace = namespace

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.namespace is not None:
                _spec["namespace"] = self.namespace

    # End of generated code
