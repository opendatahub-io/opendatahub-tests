"""Lightweight wrapper for the LLMInferenceServiceConfig custom resource."""

from ocp_resources.resource import NamespacedResource


class LLMInferenceServiceConfig(NamespacedResource):
    """LLMInferenceServiceConfig custom resource from the serving.kserve.io API group."""

    api_group: str = NamespacedResource.ApiGroup.SERVING_KSERVE_IO
