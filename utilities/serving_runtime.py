from typing import Any, Dict
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.template import Template


class ServingRuntimeFromTemplate(ServingRuntime):
    def __init__(self, client: DynamicClient, name: str, namespace: str, template_name: str):
        self.client = client
        self.name = name
        self.namespace = namespace
        self.template_name = template_name

        super().__init__(client=self.client, kind_dict=self.get_model_dict_from_template())

    def get_model_template(self) -> Template:
        template = Template(
            client=self.client,
            name=self.template_name,
            namespace="redhat-ods-applications",
        )
        if template.exists:
            return template

        raise ResourceNotFoundError(f"{self.template_name} template not found")

    def get_model_dict_from_template(self) -> Dict[Any, Any]:
        template = self.get_model_template()
        model_dict: Dict[str, Any] = template.instance.objects[0].to_dict()
        model_dict["metadata"]["name"] = self.name
        model_dict["metadata"]["namespace"] = self.namespace

        return model_dict
