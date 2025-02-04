from typing import Any, Optional
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.template import Template
from pytest_testconfig import config as py_config


class ServingRuntimeFromTemplate(ServingRuntime):
    def __init__(
        self,
        client: DynamicClient,
        name: str,
        namespace: str,
        template_name: str,
        multi_model: Optional[bool] = None,
        enable_http: Optional[bool] = None,
        enable_grpc: Optional[bool] = None,
        resources: Optional[dict[str, Any]] = None,
        model_format_name: Optional[dict[str, str]] = None,
        unprivileged_client: Optional[DynamicClient] = None,
        enable_external_route: Optional[bool] = None,
        enable_auth: Optional[bool] = None,
        protocol: Optional[str] = None,
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

        # model mesh attributes
        self.enable_external_route = enable_external_route
        self.enable_auth = enable_auth
        self.protocol = protocol

        self.model_dict = self.update_model_dict()

        super().__init__(client=self.unprivileged_client or self.admin_client, kind_dict=self.model_dict)

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

        if self.multi_model is not None:
            _model_dict["spec"]["multiModel"] = self.multi_model

        if self.enable_external_route:
            _model_dict["metadata"].setdefault("annotations", {})["enable-route"] = "true"

        if self.enable_auth:
            _model_dict["metadata"].setdefault("annotations", {})["enable-auth"] = "true"

        if self.protocol is not None:
            _model_dict["metadata"].setdefault("annotations", {})["opendatahub.io/apiProtocol"] = self.protocol

        for container in _model_dict["spec"]["containers"]:
            for env in container.get("env", []):
                if env["name"] == "RUNTIME_HTTP_ENABLED" and self.enable_http is not None:
                    env["value"] = str(self.enable_http).lower()

                if env["name"] == "RUNTIME_GRPC_ENABLED" and self.enable_grpc is not None:
                    env["value"] = str(self.enable_grpc).lower()

                    if self.enable_grpc is True:
                        container["ports"][0] = {"containerPort": 8085, "name": "h2c", "protocol": "TCP"}

            if self.resources is not None and (resource_dict := self.resources.get(container["name"])):
                container["resources"] = resource_dict

        if self.model_format_name is not None:
            for model in _model_dict["spec"]["supportedModelFormats"]:
                if model["name"] in self.model_format_name:
                    model["version"] = self.model_format_name[model["name"]]

        return _model_dict
