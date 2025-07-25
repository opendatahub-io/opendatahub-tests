from ocp_resources.template import Template
from ocp_resources.serving_runtime import ServingRuntime
from kubernetes.dynamic import DynamicClient
from utilities.serving_runtime import ServingRuntimeFromTemplate
from simple_logger.logger import get_logger

from tests.model_serving.model_runtime.rhoai_upgrade.constant import APPLICATIONS_NAMESPACE

LOGGER = get_logger(name=__name__)


def create_serving_runtime_template(client: DynamicClient, yaml_file: str) -> Template:
    """
    Creates a Serving Runtime Template from a YAML file.
    """
    LOGGER.debug(f"Creating Serving Runtime template from yaml_file {yaml_file}")
    template = Template(client=client, yaml_file=yaml_file, namespace=APPLICATIONS_NAMESPACE)
    template.create()
    return template


def create_serving_runtime_instance(
    client: DynamicClient, serving_runtime_template_name: str, serving_runtime_instance_name: str
) -> ServingRuntime:
    """
    Creates a Serving Runtime instance from a given template.
    """
    LOGGER.debug(
        f"Creating Serving Runtime Instance '{serving_runtime_instance_name}' "
        f"from template '{serving_runtime_template_name}'"
    )
    instance = ServingRuntimeFromTemplate(
        client=client,
        name=serving_runtime_instance_name,
        namespace=APPLICATIONS_NAMESPACE,
        template_name=serving_runtime_template_name,
    )
    instance.create()
    return instance


def get_serving_runtime_template_name_list(client: DynamicClient) -> list[str]:
    """
    Retrieves a list of Serving Runtime Template names.
    """
    templates_api = client.resources.get(api_version="template.openshift.io/v1", kind="Template", name="templates")
    templates_items = templates_api.get(namespace=APPLICATIONS_NAMESPACE).items
    return [template.metadata.name for template in templates_items]


def get_serving_runtime_template(client: DynamicClient, serving_runtime_template_name: str) -> Template:
    """
    Retrieves an existing Serving Runtime Template by name.
    """
    return Template(client=client, name=serving_runtime_template_name, namespace=APPLICATIONS_NAMESPACE)


def get_serving_runtime_instance(client: DynamicClient, serving_runtime_instance_name: str) -> ServingRuntime:
    """
    Retrieves an existing Serving Runtime instance by name.
    """
    return ServingRuntime(client=client, name=serving_runtime_instance_name, namespace=APPLICATIONS_NAMESPACE)


def delete_serving_runtime_template(serving_runtime_template: Template) -> None:
    """
    Deletes a Serving Runtime Template.
    """
    LOGGER.debug(f"Deleting Serving Runtime Template '{serving_runtime_template.name}'")
    serving_runtime_template.delete()


def delete_serving_runtime_instance(serving_runtime_instance: ServingRuntime) -> None:
    """
    Deletes a Serving Runtime instance.
    """
    LOGGER.debug(f"Deleting Serving Runtime Instance '{serving_runtime_instance.name}'")
    serving_runtime_instance.delete()
