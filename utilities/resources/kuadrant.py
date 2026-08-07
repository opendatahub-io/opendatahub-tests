# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md

from ocp_resources.resource import NamespacedResource

from utilities.constants import ApiGroups


class Kuadrant(NamespacedResource):
    """
    Kuadrant is the Schema for the kuadrants API.
    """

    api_group: str = ApiGroups.KUADRANT_IO

    # End of generated code
