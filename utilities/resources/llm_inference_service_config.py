# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md

from ocp_resources.resource import NamespacedResource


class LLMInferenceServiceConfig(NamespacedResource):
    """LLMInferenceServiceConfig custom resource from the serving.kserve.io API group."""

    api_group: str = NamespacedResource.ApiGroup.SERVING_KSERVE_IO
