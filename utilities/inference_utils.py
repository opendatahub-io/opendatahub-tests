import json
import shlex
from functools import cache
from string import Template
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from ocp_resources.inference_service import InferenceService
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger

from utilities.manifests.runtime_query_formats import RUNTIME_QUERY_FORMATS

LOGGER = get_logger(name=__name__)
INFERENCE_QUERIES: Dict[str, Dict[str, str]] = {
    "nitrogen-boil-temp": {
        "text": "At what temperature does liquid Nitrogen boil?",
        "response_text": "74 degrees F",
    }
}


class Inference:
    def __init__(self, inference_service: InferenceService, runtime: str, protocol: str, inference_type: str):
        """
        Args:
            inference_service: InferenceService object
        """
        self.inference_service = inference_service
        self.runtime = runtime
        self.protocol = protocol
        self.inference_type = inference_type
        self.url = self.get_inference_url()

    def get_inference_url(self) -> str:
        if url := self.inference_service.instance.status.components.predictor.url:
            return urlparse(url).netloc
        else:
            raise ValueError(f"{self.inference_service.name}: No url found in InferenceService status")

    @cache
    def get_inference_config(self) -> Dict[str, Any]:
        if runtime_config := RUNTIME_QUERY_FORMATS.get(self.runtime):
            if inference_type := runtime_config.get(self.inference_type):
                if data := inference_type.get(self.protocol):
                    return data

                else:
                    raise ValueError(
                        f"Protocol {self.protocol} not supported.\n" f"Supported protocols are {inference_type}"
                    )

            else:
                raise ValueError(
                    f"Inference type {inference_type} not supported.\n"
                    f"Supported inference types are {runtime_config['endpoints']}"
                )

        else:
            raise ValueError(f"Runtime {self.runtime} not supported. Supported runtimes are {RUNTIME_QUERY_FORMATS}")

    @property
    def inference_response_text_key_name(self) -> str:
        return self.get_inference_config()["response_fields_map"]["response_text"]

    def generate_command(
        self,
        model_name: str,
        text: str,
        insecure: bool = False,
        token: Optional[str] = None,
    ) -> str:
        data = self.get_inference_config()
        header = Template(data["header"]).safe_substitute(model_name=model_name)
        body = Template(data["body"]).safe_substitute(
            model_name=model_name,
            query_text=text,
        )

        if self.protocol == "http":
            self.url = f"https://{self.url}"

        cmd = f"curl -H {header} -d '{body}' {self.url}/{data['endpoint']}"

        if insecure:
            cmd += " --insecure"

        if token:
            cmd += f' -H "Authorization: Bearer {token}"'

        return cmd

    def run_inference(
        self,
        model_name: str,
        text: str,
        insecure: bool = False,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        cmd = self.generate_command(
            model_name=model_name,
            text=text,
            insecure=insecure,
            token=token,
        )

        res, out, err = run_command(command=shlex.split(cmd), verify_stderr=False)
        if not res:
            raise ValueError(f"Inference failed with error: {err}\n" f"Output: {out}\n" f"Command: {cmd}")

        return json.loads(out)
