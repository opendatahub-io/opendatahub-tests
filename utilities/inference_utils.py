from __future__ import annotations

import json
import re
import shlex
from json import JSONDecodeError
from string import Template
from typing import Any, Optional
from urllib.parse import urlparse

from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import get_client
from ocp_resources.service import Service
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger
from timeout_sampler import retry

from utilities.infra import (
    get_inference_serving_runtime,
    get_model_mesh_route,
    get_pods_by_isvc_label,
    get_services_by_isvc_label,
)
from utilities.certificates_utils import get_ca_bundle
from utilities.constants import (
    KServeDeploymentType,
    Protocols,
    HTTPRequest,
)
import portforward

LOGGER = get_logger(name=__name__)


class Inference:
    ALL_TOKENS: str = "all-tokens"
    STREAMING: str = "streaming"
    INFER: str = "infer"

    def __init__(self, inference_service: InferenceService):
        """
        Args:
            inference_service: InferenceService object
        """
        self.inference_service = inference_service
        self.deployment_mode = self.inference_service.instance.metadata.annotations["serving.kserve.io/deploymentMode"]
        self.runtime = get_inference_serving_runtime(isvc=self.inference_service)
        self.visibility_exposed = self.is_service_exposed()

        self.inference_url = self.get_inference_url()

    def get_inference_url(self) -> str:
        """
        Get inference url

        Returns:
            inference url

        Raises:
            ValueError: If the inference url is not found

        """
        if self.visibility_exposed:
            if self.deployment_mode == KServeDeploymentType.SERVERLESS and (
                url := self.inference_service.instance.status.components.predictor.url
            ):
                return urlparse(url=url).netloc

            elif self.deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT and (
                url := self.inference_service.instance.status.url
            ):
                return urlparse(url=url).netloc

            elif self.deployment_mode == KServeDeploymentType.MODEL_MESH:
                route = get_model_mesh_route(client=self.inference_service.client, isvc=self.inference_service)
                return route.instance.spec.host

            else:
                raise ValueError(f"{self.inference_service.name}: No url found for inference")

        else:
            return "localhost"

    def is_service_exposed(self) -> bool:
        """
        Check if the service is exposed or internal

        Returns:
            bool: True if the service is exposed, False otherwise

        """
        labels = self.inference_service.labels

        if self.deployment_mode in KServeDeploymentType.RAW_DEPLOYMENT:
            return labels and labels.get("networking.kserve.io/visibility") == "exposed"

        if self.deployment_mode == KServeDeploymentType.SERVERLESS:
            if labels and labels.get("networking.knative.dev/visibility") == "cluster-local":
                return False
            else:
                return True

        if self.deployment_mode == KServeDeploymentType.MODEL_MESH:
            if self.runtime:
                _annotations = self.runtime.instance.metadata.annotations
                return _annotations and _annotations.get("enable-route") == "true"

        return False


class UserInference(Inference):
    def __init__(
        self,
        protocol: str,
        inference_type: str,
        inference_config: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """
        User inference object

        Args:
            protocol (str): inference protocol
            inference_type (str): inference type
            inference_config (dict[str, Any]): inference config
            **kwargs ():
        """
        super().__init__(**kwargs)

        self.protocol = protocol
        self.inference_type = inference_type
        self.inference_config = inference_config
        self.runtime_config = self.get_runtime_config()

    def get_runtime_config(self) -> dict[str, Any]:
        """
        Get runtime config from inference config based on inference type and protocol

        Returns:
            dict[str, Any]: runtime config

        Raises:
            ValueError: If the runtime config is not found

        """
        if inference_type := self.inference_config.get(self.inference_type):
            protocol = Protocols.HTTP if self.protocol in Protocols.TCP_PROTOCOLS else self.protocol
            if data := inference_type.get(protocol):
                return data

            else:
                raise ValueError(f"Protocol {protocol} not supported.\nSupported protocols are {self.inference_type}")

        else:
            raise ValueError(
                f"Inference type {inference_type} not supported.\nSupported inference types are {self.inference_config}"
            )

    @property
    def inference_response_text_key_name(self) -> Optional[str]:
        """
        Get inference response text key name from runtime config

        Returns:
            Optional[str]: inference response text key name

        """
        return self.runtime_config["response_fields_map"].get("response_output")

    @property
    def inference_response_key_name(self) -> str:
        """
        Get inference response key name from runtime config

        Returns:
            str: inference response key name

        """
        return self.runtime_config["response_fields_map"].get("response", "output")

    def get_inference_body(
        self,
        model_name: str,
        inference_input: Optional[Any] = None,
        use_default_query: bool = False,
    ) -> str:
        """
        Get inference body from runtime config

        Args:
            model_name (str): inference model name
            inference_input (Any): inference input
            use_default_query (bool): use default query from inference config

        Returns:
            str: inference body

        Raises:
            ValueError: If inference input is not provided

        """
        if not use_default_query and inference_input is None:
            raise ValueError("Either pass `inference_input` or set `use_default_query` to True")

        if use_default_query:
            default_query_config = self.inference_config.get("default_query_model")
            if not default_query_config:
                raise ValueError(f"Missing default query config for {model_name}")

            if self.inference_config.get("support_multi_default_queries"):
                inference_input = default_query_config.get(self.inference_type).get("query_input")
            else:
                inference_input = default_query_config.get("query_input")

            if not inference_input:
                raise ValueError(f"Missing default query dict for {model_name}")

        if isinstance(inference_input, list):
            inference_input = json.dumps(inference_input)

        return Template(self.runtime_config["body"]).safe_substitute(
            model_name=model_name,
            query_input=inference_input,
        )

    def get_inference_endpoint_url(self) -> str:
        """
        Get inference endpoint url from runtime config

        Returns:
            str: inference endpoint url

        Raises:
            ValueError: If the protocol is not supported

        """
        endpoint = Template(self.runtime_config["endpoint"]).safe_substitute(model_name=self.inference_service.name)

        if self.protocol in Protocols.TCP_PROTOCOLS:
            return f"{self.protocol}://{self.inference_url}/{endpoint}"

        elif self.protocol == "grpc":
            return f"{self.inference_url}{':443' if self.visibility_exposed else ''} {endpoint}"

        else:
            raise ValueError(f"Protocol {self.protocol} not supported")

    def generate_command(
        self,
        model_name: str,
        inference_input: Optional[Any] = None,
        use_default_query: bool = False,
        insecure: bool = False,
        token: Optional[str] = None,
    ) -> str:
        """
        Generate command to run inference

        Args:
            model_name (str): inference model name
            inference_input (Any): inference input
            use_default_query (bool): use default query from inference config
            insecure (bool): Use insecure connection
            token (str): Token to use for authentication

        Returns:
            str: inference command

        Raises:
                ValueError: If the protocol is not supported

        """
        body = self.get_inference_body(
            model_name=model_name,
            inference_input=inference_input,
            use_default_query=use_default_query,
        )
        header = f"'{Template(self.runtime_config['header']).safe_substitute(model_name=model_name)}'"
        url = self.get_inference_endpoint_url()

        if self.protocol in Protocols.TCP_PROTOCOLS:
            cmd_exec = "curl -i -s "

        elif self.protocol == "grpc":
            cmd_exec = "grpcurl -connect-timeout 10 "
            if self.deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
                cmd_exec += " --plaintext "

        else:
            raise ValueError(f"Protocol {self.protocol} not supported")

        cmd = f"{cmd_exec} -d '{body}'  -H {header}"

        if token:
            cmd += f" {HTTPRequest.AUTH_HEADER.format(token=token)}"

        if insecure:
            cmd += " --insecure"

        else:
            # admin client is needed to check if cluster is managed
            _client = get_client()
            if ca := get_ca_bundle(client=_client, deployment_mode=self.deployment_mode):
                cmd += f" --cacert {ca} "

            else:
                LOGGER.warning("No CA bundle found, using insecure access")
                cmd += " --insecure"

        if cmd_args := self.runtime_config.get("args"):
            cmd += f" {cmd_args} "

        cmd += f" {url}"

        return cmd

    def run_inference_flow(
        self,
        model_name: str,
        inference_input: Optional[str] = None,
        use_default_query: bool = False,
        insecure: bool = False,
        token: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run inference full flow - generate command and run it

        Args:
            model_name (str): inference model name
            inference_input (str): inference input
            use_default_query (bool): use default query from inference config
            insecure (bool): Use insecure connection
            token (str): Token to use for authentication

        Returns:
            dict: inference response dict with response headers and response output

        """
        cmd = self.generate_command(
            model_name=model_name,
            inference_input=inference_input,
            use_default_query=use_default_query,
            insecure=insecure,
            token=token,
        )

        out = self.run_inference(cmd=cmd)

        try:
            if self.protocol in Protocols.TCP_PROTOCOLS:
                # with curl response headers are also returned
                response_dict: dict[str, Any] = {}
                response_headers: list[str] = []

                if "content-type: application/json" in out.lower():
                    if response_re := re.match(r"(.*)\n\{", out, re.MULTILINE | re.DOTALL):
                        response_headers = response_re.group(1).splitlines()
                        if output_re := re.search(r"(\{.*)(?s:.*)(\})", out, re.MULTILINE | re.DOTALL):
                            output = re.sub(r"\n\s*", "", output_re.group())
                            response_dict["output"] = json.loads(output)

                else:
                    response_headers = out.splitlines()[:-2]
                    response_dict["output"] = json.loads(response_headers[-1])

                for line in response_headers:
                    if line:
                        header_name, header_value = re.split(": | ", line.strip(), maxsplit=1)
                        response_dict[header_name] = header_value

                return response_dict
            else:
                return json.loads(out)

        except JSONDecodeError:
            return {"output": out}

    @retry(wait_timeout=30, sleep=5)
    def run_inference(self, cmd: str) -> str:
        """
        Run inference command

        Args:
            cmd (str): inference command

        Returns:
            str: inference output

        Raises:
            ValueError: If inference fails

        """
        # For internal inference, we need to use port forwarding to the service
        if not self.visibility_exposed:
            svc = get_services_by_isvc_label(
                client=self.inference_service.client,
                isvc=self.inference_service,
                runtime_name=self.runtime.name,
            )[0]
            port = self.get_target_port(svc=svc)
            cmd = cmd.replace("localhost", f"localhost:{port}")

            with portforward.forward(
                pod_or_service=svc.name,
                namespace=svc.namespace,
                from_port=port,
                to_port=port,
            ):
                res, out, err = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)

        else:
            res, out, err = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)

        if not res:
            raise ValueError(f"Inference failed with error: {err}\nOutput: {out}\nCommand: {cmd}")

        return out

    def get_target_port(self, svc: Service) -> int:
        """
        Get target port for inference when using port forwarding

        Args:
            svc (Service): Service object

        Returns:
            int: Target port

        Raises:
                ValueError: If target port is not found in service

        """
        if self.protocol in Protocols.ALL_SUPPORTED_PROTOCOLS:
            svc_protocol = "TCP"
        else:
            svc_protocol = self.protocol

        ports = svc.instance.spec.ports

        # For multi node with headless service, we need to get the pod to get the port
        # TODO: check behavior for both normal and headless service
        if self.inference_service.instance.spec.predictor.workerSpec and not self.visibility_exposed:
            pod = get_pods_by_isvc_label(
                client=self.inference_service.client,
                isvc=self.inference_service,
                runtime_name=self.runtime.name,
            )[0]
            if ports := pod.instance.spec.containers[0].ports:
                return ports[0].containerPort

        if not ports:
            raise ValueError(f"Service {svc.name} has no ports")

        for port in ports:
            svc_port = port.targetPort if isinstance(port.targetPort, int) else port.port

            if (
                self.deployment_mode == KServeDeploymentType.MODEL_MESH
                and port.protocol.lower() == svc_protocol.lower()
                and port.name == self.protocol
            ):
                return svc_port

            elif (
                self.deployment_mode
                in (
                    KServeDeploymentType.RAW_DEPLOYMENT,
                    KServeDeploymentType.SERVERLESS,
                )
                and port.protocol.lower() == svc_protocol.lower()
            ):
                return svc_port

        raise ValueError(f"No port found for protocol {self.protocol} service {svc.instance}")
