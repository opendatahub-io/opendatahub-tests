import copy
from typing import Any

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.template import Template
from pytest_testconfig import config as py_config

from utilities.constants import ApiGroups, PortNames, Protocols, vLLM_CONFIG


class ServingRuntimeFromTemplate(ServingRuntime):
    def __init__(
        self,
        client: DynamicClient,
        name: str,
        namespace: str,
        template_name: str,
        multi_model: bool | None = None,
        enable_http: bool | None = None,
        enable_grpc: bool | None = None,
        resources: dict[str, Any] | None = None,
        model_format_name: dict[str, str] | None = None,
        unprivileged_client: DynamicClient | None = None,
        enable_external_route: bool | None = None,
        enable_auth: bool | None = None,
        protocol: str | None = None,
        deployment_type: str | None = None,
        runtime_image: str | None = None,
        models_priorities: dict[str, str] | None = None,
        supported_model_formats: list[dict[str, Any]] | None = None,
        volumes: list[dict[str, Any]] | None = None,
        containers: dict[str, dict[str, Any]] | None = None,
        support_tgis_open_ai_endpoints: bool = False,
        teardown: bool = True,
    ):
        """
        ServingRuntimeFromTemplate class

        Args:
            client (DynamicClient): DynamicClient object
            name (str): Name of the serving runtime
            namespace (str): Namespace of the serving runtime
            template_name (str): Name of the serving runtime template to use
            multi_model (bool): Whether to use multi model (model mesh) or not (kserve)
            enable_http (bool): Whether to enable http or not
            enable_grpc (bool): Whether to enable grpc or not
            resources (dict): Resources to be used for the serving runtime
            model_format_name (dict): Model format name to be used for the serving runtime
            unprivileged_client (DynamicClient): DynamicClient object for unprivileged user
            enable_external_route (bool): Whether to enable external route or not; relevant for model mesh only
            enable_auth (bool): Whether to enable auth or not; relevant for model mesh only
            protocol (str): Protocol to be used for the serving runtime; relevant for model mesh only
            models_priorities (dict[str, str]): Model priority to be used for the serving runtime
            supported_model_formats (list[dict[str, Any]]): Model formats;
                overwrites template's `supportedModelFormats`
            volumes (list[dict[str, Any]]): Volumes to be used with the serving runtime
            containers (dict[str, dict[str, Any]]): Containers configurations to override or add
                to the serving runtime
            support_tgis_open_ai_endpoints (bool): Whether to support TGIS and OpenAI endpoints using
                a single entry point
            teardown (bool): Whether to teardown the serving runtime or not
        """

        self.admin_client = client
        self.name = name
        self.namespace = namespace
        self.template_name = template_name
        self.multi_model = multi_model
        self.enable_http = enable_http
        self.enable_grpc = enable_grpc
        self.resources = resources
        self.model_format_name = model_format_name
        self.unprivileged_client = unprivileged_client
        self.deployment_type = deployment_type
        self.runtime_image = runtime_image
        self.models_priorities = models_priorities
        self.supported_model_formats = supported_model_formats
        self.volumes = volumes
        self.containers = containers
        self.support_tgis_open_ai_endpoints = support_tgis_open_ai_endpoints

        # model mesh attributes
        self.enable_external_route = enable_external_route
        self.enable_auth = enable_auth
        self.protocol = protocol

        self.model_dict = self.update_model_dict()

        super().__init__(
            client=self.unprivileged_client or self.admin_client,
            kind_dict=self.model_dict,
            teardown=teardown,
        )

    def get_model_template(self) -> Template:
        """
        Get the model template from the cluster

        Returns:
            Template: SeringRuntime Template object

        Raises:
            ResourceNotFoundError: If the template is not found

        """
        # Only admin client can get templates from the cluster
        template = Template(
            client=self.admin_client,
            name=self.template_name,
            namespace=py_config["applications_namespace"],
        )
        if template.exists:
            return template

        raise ResourceNotFoundError(f"{self.template_name} template not found")

    def get_model_dict_from_template(self) -> dict[Any, Any]:
        """
        Get the model dictionary from the template

        Returns:
            dict[Any, Any]: Model dict

        """
        template = self.get_model_template()
        model_dict: dict[str, Any] = template.instance.objects[0].to_dict()
        model_dict["metadata"]["name"] = self.name
        model_dict["metadata"]["namespace"] = self.namespace

        return model_dict

    def update_model_dict(self) -> dict[str, Any]:
        """
        Update the model dict with values from init

        Returns:
            dict[str, Any]: Model dict

        """
        _model_dict = self.get_model_dict_from_template()
        _model_metadata = _model_dict.get("metadata", {})
        _model_spec = _model_dict.get("spec", {})

        if self.multi_model is not None:
            _model_spec["multiModel"] = self.multi_model

        if self.enable_external_route:
            _model_metadata.setdefault("annotations", {})["enable-route"] = "true"

        if self.enable_auth:
            _model_metadata.setdefault("annotations", {})["enable-auth"] = "true"

        if self.protocol is not None:
            _model_metadata.setdefault("annotations", {})[f"{ApiGroups.OPENDATAHUB_IO}/apiProtocol"] = self.protocol

        template_containers = _model_spec.get("containers", [])

        containers_to_add = copy.deepcopy(self.containers) if self.containers else {}

        for container in template_containers:
            container_name = container.get("name")

            for env in container.get("env", []):
                if env["name"] == "RUNTIME_HTTP_ENABLED" and self.enable_http is not None:
                    env["value"] = str(self.enable_http).lower()

                if env["name"] == "RUNTIME_GRPC_ENABLED" and self.enable_grpc is not None:
                    env["value"] = str(self.enable_grpc).lower()

                    if self.enable_grpc is True:
                        container["ports"][0] = {
                            "containerPort": 8085,
                            "name": PortNames.GRPC_PORT_NAME,
                            "protocol": Protocols.TCP,
                        }

            if self.resources is not None and (resource_dict := self.resources.get(container_name)):
                container["resources"] = resource_dict

            if self.runtime_image is not None:
                container["image"] = self.runtime_image

            # Support single entrypoint for TGIS and OpenAI
            if self.support_tgis_open_ai_endpoints:
                if "vllm" in self.template_name and self.runtime_image is not None and self.deployment_type is not None:
                    is_grpc = "grpc" in self.deployment_type.lower()
                    is_raw = "raw" in self.deployment_type.lower()
                    # Remove '--model' from the container args, we will pass this using isvc
                    container["args"] = [arg for arg in container["args"] if "--model" not in arg]
                    # Update command if deployment type is grpc
                    if is_grpc or is_raw:
                        container["command"][-1] = vLLM_CONFIG["commands"]["GRPC"]

                    if is_grpc:
                        container["ports"] = vLLM_CONFIG["port_configurations"]["grpc"]
                    elif is_raw:
                        container["ports"] = vLLM_CONFIG["port_configurations"]["raw"]

            if containers_to_add and container_name in containers_to_add:
                container_config = containers_to_add.pop(container_name)
                for key, value in container_config.items():
                    container[key] = value

        if containers_to_add:
            for container_name, container_config in containers_to_add.items():
                new_container = {"name": container_name}
                new_container.update(container_config)
                template_containers.append(new_container)

        _model_spec["containers"] = template_containers

        if self.supported_model_formats:
            _model_spec["supportedModelFormats"] = self.supported_model_formats
        else:
            _model_spec_supported_formats = _model_spec.get("supportedModelFormats", [])
            if self.model_format_name is not None:
                for model in _model_spec_supported_formats:
                    if model["name"] in self.model_format_name:
                        model["version"] = self.model_format_name[model["name"]]

            if self.models_priorities:
                for _model in _model_spec_supported_formats:
                    _model_name = _model["name"]
                    if _model_name in self.models_priorities:
                        _model["priority"] = self.models_priorities[_model_name]

        if self.volumes:
            _model_spec["volumes"] = self.volumes

        return _model_dict
