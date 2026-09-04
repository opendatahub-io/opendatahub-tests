from ocp_resources.resource import NamespacedResource


class ServiceEntry(NamespacedResource):
    """Istio ServiceEntry object."""

    api_group: str = NamespacedResource.ApiGroup.NETWORKING_ISTIO_IO
