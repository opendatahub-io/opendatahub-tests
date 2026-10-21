# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from ocp_resources.resource import NamespacedResource

from utilities.constants import ApiGroups


class SparkConnect(NamespacedResource):
    """
    SparkConnect is the Schema for the sparkconnects API.

    A SparkConnect is a long-lived Spark Connect server (a driver pod plus executor pods) that
    remote clients query over gRPC. Unlike SparkApplication/ScheduledSparkApplication, its API
    version is ``v1alpha1``.
    """

    api_group: str = ApiGroups.SPARKOPERATOR_K8S_IO
    api_version: str = "v1alpha1"

    # End of generated code
