# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from ocp_resources.resource import NamespacedResource

from utilities.constants import ApiGroups


class ScheduledSparkApplication(NamespacedResource):
    """
    ScheduledSparkApplication is the Schema for the scheduledsparkapplications API.

    A ScheduledSparkApplication is the Spark Operator's cron wrapper around a
    SparkApplication: on each schedule tick the operator mints a child
    SparkApplication from ``spec.template`` and records it in ``status.lastRunName``.
    """

    api_group: str = ApiGroups.SPARKOPERATOR_K8S_IO

    # End of generated code
