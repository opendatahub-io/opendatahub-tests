# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/class_generator/README.md


from typing import Any, Dict, Optional

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import Resource


class AdmissionCheck(Resource):
    """
    AdmissionCheck is the Schema for the admissionchecks API
    """

    api_group: str = "kueue.x-k8s.io"

    def __init__(
        self,
        controller_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            controller_name (str): controllerName identifies the controller that processes the
              AdmissionCheck, not necessarily a Kubernetes Pod or Deployment
              name. Cannot be empty.

            parameters (dict[str, Any]): parameters identifies a configuration with additional parameters for
              the check.

        """
        super().__init__(**kwargs)

        self.controller_name = controller_name
        self.parameters = parameters

    def to_dict(self) -> None:
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.controller_name is None:
                raise MissingRequiredArgumentError(argument="self.controller_name")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["controllerName"] = self.controller_name

            if self.parameters is not None:
                _spec["parameters"] = self.parameters

    # End of generated code
