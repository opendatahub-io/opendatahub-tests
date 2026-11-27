from ocp_resources.resource import Resource


class Workbenches(Resource):
    """Cluster-scoped Workbenches module API."""

    api_version: str = f"{Resource.ApiGroup.COMPONENTS_PLATFORM_OPENDATAHUB_IO}/v1alpha1"
