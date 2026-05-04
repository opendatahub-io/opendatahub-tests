# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from ocp_resources.resource import NamespacedResource


class Workflow(NamespacedResource):
    """
    Argo Workflow resource used by Data Science Pipelines.

    API Group: argoproj.io
    API Version: v1alpha1
    """

    api_group: str = "argoproj.io"
    api_version: str = "v1alpha1"

    # End of generated code
