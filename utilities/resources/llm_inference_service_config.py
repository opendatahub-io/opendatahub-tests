from ocp_resources.resource import NamespacedResource


class LLMInferenceServiceConfig(NamespacedResource):
    api_group: str = "serving.kserve.io"
    api_version: str = "v1alpha1"
