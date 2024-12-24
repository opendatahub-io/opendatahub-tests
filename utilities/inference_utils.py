from __future__ import annotations
import json
import re
import shlex
from json import JSONDecodeError
from string import Template
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from ocp_resources.inference_service import InferenceService
from ocp_resources.service import Service
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger

from tests.model_serving.model_server.utils import (
    get_services_by_isvc_label,
)
from utilities.constants import KServeDeploymentType, Protocols
from utilities.manifests.runtime_query_config import RUNTIMES_QUERY_CONFIG
import portforward

LOGGER = get_logger(name=__name__)


class Inference:
    ALL_TOKENS: str = "all-tokens"
    STREAMING: str = "streaming"

    def __init__(self, inference_service: InferenceService, runtime: str):
        """
        Args:
            inference_service: InferenceService object
        """
        self.inference_service = inference_service
        self.runtime = runtime
        self.visibility_exposed = self.is_service_exposed()

        self.inference_url = self.get_inference_url()

    def get_inference_url(self) -> str:
        # TODO: add ModelMesh support
        if self.visibility_exposed:
            if url := self.inference_service.instance.status.components.predictor.url:
                return urlparse(url).netloc
            else:
                raise ValueError(f"{self.inference_service.name}: No url found in InferenceService status")

        else:
            return "localhost"

    def is_service_exposed(self) -> bool:
        labels = self.inference_service.labels
        inference_type = self.inference_service.instance.metadata.annotations["serving.kserve.io/deploymentMode"]

        if inference_type == KServeDeploymentType.RAW_DEPLOYMENT:
            if labels and labels.get("networking.kserve.io/visibility") == "exposed":
                return True
            else:
                return False

        elif inference_type == KServeDeploymentType.SERVERLESS:
            if labels and labels.get("networking.knative.dev/visibility") == "cluster-local":
                return False
            else:
                return True

        else:
            # TODO: add support for ModelMesh
            return False


class LlmInference(Inference):
    def __init__(self, protocol: str, inference_type: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.protocol = protocol
        self.inference_type = inference_type
        self.inference_config = self.get_inference_config()
        self.runtime_config = self.get_runtime_config()

    def get_inference_config(self) -> Dict[str, Any]:
        if runtime_config := RUNTIMES_QUERY_CONFIG.get(self.runtime):
            return runtime_config

        else:
            raise ValueError(f"Runtime {self.runtime} not supported. Supported runtimes are {RUNTIMES_QUERY_CONFIG}")

    def get_runtime_config(self) -> Dict[str, Any]:
        if inference_type := self.inference_config.get(self.inference_type):
            protocol = Protocols.HTTP if self.protocol in Protocols.TCP_PROTOCOLS else self.protocol
            if data := inference_type.get(protocol):
                return data

            else:
                raise ValueError(f"Protocol {protocol} not supported.\nSupported protocols are {inference_type}")

        else:
            raise ValueError(
                f"Inference type {inference_type} not supported.\nSupported inference types are {self.inference_config}"
            )

    @property
    def inference_response_text_key_name(self) -> Optional[str]:
        return self.runtime_config["response_fields_map"].get("response_text")

    def generate_command(
        self,
        model_name: str,
        text: Optional[str] = None,
        use_default_query: bool = False,
        insecure: bool = False,
        token: Optional[str] = None,
        port: Optional[int] = None,
        is_exposed: bool = False,
    ) -> str:
        if use_default_query:
            text = self.inference_config.get("default_query_model", {}).get("text")
            if not text:
                raise ValueError(f"Missing default query dict for {model_name}")

        header = f"'{Template(self.runtime_config['header']).safe_substitute(model_name=model_name)}'"
        body = Template(self.runtime_config["body"]).safe_substitute(
            model_name=model_name,
            query_text=text,
        )

        if self.protocol in Protocols.TCP_PROTOCOLS:
            url = f"{self.protocol}://{self.inference_url}/{self.runtime_config['endpoint']}"
            cmd_exec = "curl -i -s"

        elif self.protocol == "grpc":
            url = f"{self.inference_url}:{port or 443} {self.runtime_config['endpoint']}"
            cmd_exec = "grpcurl -connect-timeout 10"

        else:
            raise ValueError(f"Protocol {self.protocol} not supported")

        cmd = f"{cmd_exec} -d '{body}'  -H {header}"

        if token:
            cmd += f' -H "Authorization: Bearer {token}"'

        if insecure:
            cmd += " --insecure"

        cmd += f" {url}"

        return cmd

    def run_inference(
        self,
        model_name: str,
        text: Optional[str] = None,
        use_default_query: bool = False,
        insecure: bool = False,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        cmd = self.generate_command(
            model_name=model_name,
            text=text,
            use_default_query=use_default_query,
            insecure=insecure,
            token=token,
        )

        # For internal inference, we need to use port forwarding to the service
        if not self.visibility_exposed:
            svc = get_services_by_isvc_label(client=self.inference_service.client, isvc=self.inference_service)[0]

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

        try:
            if self.protocol in Protocols.TCP_PROTOCOLS:
                # with curl response headers are also returned
                response_dict = {}
                response_list = out.splitlines()
                for line in response_list[:-2]:
                    header_name, header_value = re.split(": | ", line.strip(), maxsplit=1)
                    response_dict[header_name] = header_value

                response_dict["output"] = json.loads(response_list[-1])

                return response_dict
            else:
                return json.loads(out)

        except JSONDecodeError:
            return {"output": out}

    def get_target_port(self, svc: Service) -> int:
        if self.protocol in Protocols.TCP_PROTOCOLS:
            svc_protocol = "TCP"
        elif self.protocol == Protocols.GRPC:
            svc_protocol = "h2c"
        else:
            svc_protocol = self.protocol

        if port := [item.targetPort for item in svc.instance.spec.ports if item.protocol == svc_protocol]:
            return port[0]

        raise ValueError(f"No port found for protocol {self.protocol} service {svc.name}")
