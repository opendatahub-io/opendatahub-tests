from ocp_resources.resource import NamespacedResource


class Workload(NamespacedResource):
    """
    Workload is the Schema for the workloads API (kueue.x-k8s.io).
    """

    api_group: str = "kueue.x-k8s.io"
