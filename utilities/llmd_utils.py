"""Utilities for LLM Deployment (LLMD) resources."""

import json
import re
import shlex
from collections.abc import Generator
from contextlib import contextmanager
from string import Template
from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutWatch, retry

from utilities.certificates_utils import get_ca_bundle
from utilities.constants import HTTPRequest, Timeout
from utilities.exceptions import InferenceResponseError
from utilities.infra import get_services_by_isvc_label
from utilities.llmd_constants import (
    ContainerImages,
    KServeGateway,
    LLMDGateway,
    LLMEndpoint,
)

LOGGER = get_logger(name=__name__)


@contextmanager
def create_llmd_gateway(
    client: DynamicClient,
    name: str = LLMDGateway.DEFAULT_NAME,
    namespace: str = LLMDGateway.DEFAULT_NAMESPACE,
    gateway_class_name: str = LLMDGateway.DEFAULT_CLASS,
    listeners: list[dict[str, Any]] | None = None,
    infrastructure: dict[str, Any] | None = None,
    wait_for_condition: bool = True,
    timeout: int = 300,
    teardown: bool = True,
) -> Generator[Gateway]:
    """
    Context manager to create and manage a Gateway resource using ocp_resources.

    Args:
        client: Kubernetes dynamic client
        name: Gateway name
        namespace: Gateway namespace
        gateway_class_name: The name of the GatewayClass resource
        listeners: List of listener configurations
        infrastructure: Infrastructure configuration
        wait_for_condition: Whether to wait for the gateway to be programmed
        timeout: Timeout in seconds for waiting
        teardown: Whether to clean up the resource

    Yields:
        Gateway: The created Gateway resource
    """
    if listeners is None:
        listeners = [
            {
                "name": "http",
                "port": 80,
                "protocol": "HTTP",
                "allowedRoutes": {"namespaces": {"from": "All"}},
            }
        ]

    if infrastructure is None:
        infrastructure = {"labels": {KServeGateway.LABEL: KServeGateway.INGRESS_GATEWAY}}
    try:
        existing_gateway = Gateway(
            client=client,
            name=name,
            namespace=namespace,
            api_group=KServeGateway.API_GROUP,
        )
        if existing_gateway.exists:
            LOGGER.info(f"Cleaning up existing Gateway {name} in namespace {namespace}")
            existing_gateway.delete(wait=True, timeout=Timeout.TIMEOUT_2MIN)
    except Exception as e:
        LOGGER.debug(f"No existing Gateway to clean up: {e}")
    gateway_body = {
        "apiVersion": f"{KServeGateway.API_GROUP}/v1",
        "kind": "Gateway",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "gatewayClassName": gateway_class_name,
            "listeners": listeners,
            "infrastructure": infrastructure,
        },
    }

    with Gateway(
        client=client,
        teardown=teardown,
        kind_dict=gateway_body,
        api_group="gateway.networking.k8s.io",
    ) as gateway:
        if wait_for_condition:
            LOGGER.info(f"Waiting for Gateway {name} to be programmed...")
            gateway.wait_for_condition(
                condition="Programmed",
                status="True",
                timeout=timeout,
            )
            LOGGER.info(f"Gateway {name} is programmed and ready")

        yield gateway


def _get_llm_config_references(enable_prefill_decode: bool = False, disable_scheduler: bool = False) -> dict[str, str]:
    """
    Get LLMInferenceServiceConfig references based on configuration type.

    Uses existing cluster configs instead of hardcoding complex configurations:
    - kserve-config-llm-template: Standard main template
    - kserve-config-llm-scheduler: Scheduler configuration
    - kserve-config-llm-prefill-template: Prefill template
    - kserve-config-llm-decode-template: Decode template

    Args:
        enable_prefill_decode: Enable prefill/decode pattern
        disable_scheduler: Disable scheduler (no-scheduler pattern)

    Returns:
        Dict with config references for the specified pattern
    """
    base_configs = {
        "template_ref": "kserve-config-llm-template",
    }

    if enable_prefill_decode:
        base_configs.update({
            "prefill_template_ref": "kserve-config-llm-prefill-template",
            "decode_template_ref": "kserve-config-llm-decode-template",
            "scheduler_ref": "kserve-config-llm-scheduler",
        })
    elif not disable_scheduler:
        base_configs["scheduler_ref"] = "kserve-config-llm-scheduler"

    return base_configs


@contextmanager
def create_llmisvc(
    client: DynamicClient,
    name: str,
    namespace: str,
    storage_uri: str | None = None,
    storage_key: str | None = None,
    storage_path: str | None = None,
    replicas: int = 1,
    wait: bool = True,
    enable_auth: bool = False,
    router_config: dict[str, Any] | None = None,
    container_image: str | None = None,
    container_resources: dict[str, Any] | None = None,
    container_env: list[dict[str, str]] | None = None,
    liveness_probe: dict[str, Any] | None = None,
    readiness_probe: dict[str, Any] | None = None,
    image_pull_secrets: list[str] | None = None,
    service_account: str | None = None,
    volumes: list[dict[str, Any]] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
    annotations: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    timeout: int = Timeout.TIMEOUT_15MIN,
    teardown: bool = True,
    model_name: str | None = None,
    prefill_config: dict[str, Any] | None = None,
    disable_scheduler: bool = False,
    enable_prefill_decode: bool = False,
) -> Generator[LLMInferenceService, Any]:
    """
    Create LLMInferenceService object following the pattern of create_isvc.

    Args:
        client: DynamicClient object
        name: LLMInferenceService name
        namespace: Namespace name
        storage_uri: Storage URI (e.g., 'oci://quay.io/user/model:tag') - used if storage_key/storage_path not provided
        storage_key: S3 secret name for authentication (alternative to storage_uri)
        storage_path: S3 path to model (alternative to storage_uri)
        replicas: Number of replicas
        wait: Wait for LLMInferenceService to be ready
        enable_auth: Enable authentication
        router_config: Router configuration (scheduler, route, gateway)
        container_image: Container image
        container_resources: Container resource requirements
        container_env: Container environment variables
        liveness_probe: Liveness probe configuration
        readiness_probe: Readiness probe configuration
        image_pull_secrets: Image pull secrets
        service_account: Service account name
        volumes: Volume configurations
        volume_mounts: Volume mount configurations
        annotations: Additional annotations
        labels: Additional labels
        timeout: Timeout for waiting
        teardown: Whether to clean up on exit
        model_name: Model name (spec.model.name field)
        prefill_config: Prefill configuration for prefill/decode pattern
        disable_scheduler: Disable scheduler in router configuration
        enable_prefill_decode: Enable prefill/decode configuration

    Yields:
        LLMInferenceService: LLMInferenceService object
    """
    if labels is None:
        labels = {}

    if annotations is None:
        annotations = {}

    if storage_uri:
        model_config = {
            "uri": storage_uri,
        }
    elif storage_key and storage_path:
        # LLMInferenceService requires full URI, construct it from bucket + path
        model_config = {
            "uri": f"s3://ods-ci-wisdom/{storage_path}",
        }
    else:
        raise ValueError("Provide either storage_uri or (storage_key and storage_path) for the model")

    if model_name:
        model_config["name"] = model_name

    config_refs = _get_llm_config_references(
        enable_prefill_decode=enable_prefill_decode, disable_scheduler=disable_scheduler
    )

    if router_config is None:
        if disable_scheduler:
            router_config = {"route": {}}
        elif enable_prefill_decode:
            router_config = {"scheduler": {"configRef": config_refs["scheduler_ref"]}, "route": {}, "gateway": {}}
        else:
            router_config = {"scheduler": {"configRef": config_refs["scheduler_ref"]}, "route": {}, "gateway": {}}

    if container_resources is None:
        raise ValueError("container_resources must be provided for LLMInferenceService")

    if container_env is None:
        container_env = [{"name": "VLLM_LOGGING_LEVEL", "value": "DEBUG"}]
        # Add FIPS-compatible env vars for vLLM CPU image
        if container_image == ContainerImages.VLLM_CPU:
            container_env.extend([
                {"name": "VLLM_ADDITIONAL_ARGS", "value": "--ssl-ciphers ECDHE+AESGCM:DHE+AESGCM"},
                {"name": "VLLM_CPU_KVCACHE_SPACE", "value": "4"},
            ])
    template_config: dict[str, Any] = {"configRef": config_refs["template_ref"]}

    if any([
        container_image,
        container_resources,
        container_env,
        liveness_probe,
        readiness_probe,
        volumes,
        service_account,
    ]):
        main_container: dict[str, Any] = {"name": "main"}

        if container_image:
            main_container["image"] = container_image
        if container_resources:
            main_container["resources"] = container_resources
        if container_env:
            main_container["env"] = container_env
        if liveness_probe:
            main_container["livenessProbe"] = liveness_probe
        if readiness_probe:
            main_container["readinessProbe"] = readiness_probe
        if volume_mounts:
            main_container["volumeMounts"] = volume_mounts

        template_config["containers"] = [main_container]

        if volumes:
            template_config["volumes"] = volumes
        if service_account:
            template_config["serviceAccountName"] = service_account
        if image_pull_secrets:
            template_config["imagePullSecrets"] = [{"name": secret} for secret in image_pull_secrets]

    if enable_auth:
        annotations["security.opendatahub.io/enable-auth"] = "true"
    else:
        annotations["security.opendatahub.io/enable-auth"] = "false"

    LOGGER.info(f"Creating LLMInferenceService {name} in namespace {namespace}")

    spec_config = {
        "model": model_config,
        "replicas": replicas,
        "router": router_config,
        "template": template_config,
    }

    if enable_prefill_decode and prefill_config:
        prefill_template = {
            "containers": [
                {
                    "name": "main",
                    "resources": container_resources,
                    "env": container_env + [{"name": "VLLM_PREFILL_MODE", "value": "true"}]
                    if container_env
                    else [{"name": "VLLM_PREFILL_MODE", "value": "true"}],
                }
            ]
        }

        if service_account:
            prefill_template["serviceAccountName"] = service_account  # type: ignore[assignment]

        spec_config["prefill"] = {
            "replicas": prefill_config.get("replicas", 1),
            "template": prefill_template,
        }

    with LLMInferenceService(
        client=client,
        name=name,
        namespace=namespace,
        annotations=annotations,
        label=labels,
        teardown=teardown,
        **spec_config,
    ) as llm_service:
        timeout_watch = TimeoutWatch(timeout=timeout)

        if wait:
            LOGGER.info(f"Waiting for LLMInferenceService {name} to be ready...")
            llm_service.wait_for_condition(
                condition="Ready",
                status="True",
                timeout=timeout_watch.remaining_time(),
            )
            LOGGER.info(f"LLMInferenceService {name} is ready")

        yield llm_service


def get_llm_inference_url(llm_service: LLMInferenceService) -> str:
    """
    Get the inference URL for an LLMInferenceService.

    This function attempts to resolve the URL in the following order:
    1. External URL from service status
    2. Service discovery via labels
    3. Fallback to service name pattern

    Args:
        llm_service: The LLMInferenceService resource

    Returns:
        str: The inference URL (full URL including protocol and path)

    Raises:
        ValueError: If the inference URL cannot be determined
    """
    # Check for external URL from status.addresses first
    if llm_service.instance.status and llm_service.instance.status.get("addresses"):
        addresses = llm_service.instance.status["addresses"]
        if addresses and len(addresses) > 0 and addresses[0].get("url"):
            url = addresses[0]["url"]
            LOGGER.debug(f"Using external URL for {llm_service.name}: {url}")
            return url

    # Fallback to legacy status.url field
    if llm_service.instance.status and llm_service.instance.status.get("url"):
        url = llm_service.instance.status["url"]
        LOGGER.debug(f"Using legacy external URL for {llm_service.name}: {url}")
        return url

    try:
        services = get_services_by_isvc_label(
            client=llm_service.client,
            isvc=llm_service,
            runtime_name=None,
        )
        if services:
            internal_url = f"http://{services[0].name}.{llm_service.namespace}.svc.cluster.local"
            LOGGER.debug(f"Using service discovery URL for {llm_service.name}: {internal_url}")
            return internal_url
    except Exception as e:
        LOGGER.warning(f"Could not get service for LLMInferenceService {llm_service.name}: {e}")
    fallback_url = f"http://{llm_service.name}.{llm_service.namespace}.svc.cluster.local"
    LOGGER.debug(f"Using fallback URL for {llm_service.name}: {fallback_url}")
    return fallback_url


def verify_inference_response_llmd(
    llm_service: LLMInferenceService,
    inference_config: dict[str, Any],
    inference_type: str,
    protocol: str,
    model_name: str | None = None,
    inference_input: Any | None = None,
    use_default_query: bool = False,
    expected_response_text: str | None = None,
    insecure: bool = False,
    token: str | None = None,
    authorized_user: bool | None = None,
) -> None:
    """
    Verify the LLM inference response following the pattern of verify_inference_response.

    Args:
        llm_service: LLMInferenceService resource to test
        inference_config: Inference configuration dictionary
        inference_type: Type of inference ('infer', 'streaming', etc.)
        protocol: Protocol to use ('http', 'grpc')
        model_name: Name of the model (defaults to service name)
        inference_input: Input for inference (optional)
        use_default_query: Whether to use default query from config
        expected_response_text: Expected response text for validation
        insecure: Whether to use insecure connections
        token: Authentication token (optional)
        authorized_user: Whether user should be authorized (optional)

    Raises:
        InferenceResponseError: If inference response is invalid
        ValueError: If inference response validation fails
    """

    model_name = model_name or llm_service.name
    inference = LLMUserInference(
        llm_service=llm_service,
        inference_config=inference_config,
        inference_type=inference_type,
        protocol=protocol,
    )

    res = inference.run_inference_flow(
        model_name=model_name,
        inference_input=inference_input,
        use_default_query=use_default_query,
        token=token,
        insecure=insecure,
    )

    if authorized_user is False:
        _validate_unauthorized_response(res=res, token=token, inference=inference)
    else:
        _validate_authorized_response(
            res=res,
            inference=inference,
            inference_config=inference_config,
            inference_type=inference_type,
            expected_response_text=expected_response_text,
            use_default_query=use_default_query,
            model_name=model_name,
        )


class LLMUserInference:
    """
    LLM-specific inference handler following the pattern of UserInference.
    """

    STREAMING = "streaming"
    INFER = "infer"

    def __init__(
        self,
        llm_service: LLMInferenceService,
        inference_config: dict[str, Any],
        inference_type: str,
        protocol: str,
    ) -> None:
        self.llm_service = llm_service
        self.inference_config = inference_config
        self.inference_type = inference_type
        self.protocol = protocol
        self.runtime_config = self.get_runtime_config()

    def get_runtime_config(self) -> dict[str, Any]:
        """Get runtime config from inference config based on inference type and protocol."""
        if inference_type_config := self.inference_config.get(self.inference_type):
            protocol = "http" if self.protocol.lower() in ["http", "https"] else self.protocol
            if data := inference_type_config.get(protocol):
                return data
            else:
                raise ValueError(f"Protocol {protocol} not supported for inference type {self.inference_type}")
        else:
            raise ValueError(f"Inference type {self.inference_type} not supported in config")

    @property
    def inference_response_text_key_name(self) -> str | None:
        """Get inference response text key name from runtime config."""
        return self.runtime_config.get("response_fields_map", {}).get("response_output")

    @property
    def inference_response_key_name(self) -> str:
        """Get inference response key name from runtime config."""
        return self.runtime_config.get("response_fields_map", {}).get("response", "output")

    def get_inference_body(
        self,
        model_name: str,
        inference_input: Any | None = None,
        use_default_query: bool = False,
    ) -> str:
        """Get inference body for LLM request."""
        if not use_default_query and inference_input is None:
            raise ValueError("Either pass `inference_input` or set `use_default_query` to True")

        if use_default_query:
            default_query_config = self.inference_config.get("default_query_model")
            if not default_query_config:
                raise ValueError(f"Missing default query config for {model_name}")

            if self.inference_config.get("support_multi_default_queries"):
                query_config = default_query_config.get(self.inference_type)
                if not query_config:
                    raise ValueError(f"Missing default query for inference type {self.inference_type}")
                query_input = query_config.get("query_input", "")
            else:
                query_input = default_query_config.get("query_input", "")

            # Use the proper JSON body template from runtime config
            body_template = self.runtime_config.get("body", "")
            if body_template:
                # Use template substitution for both model name and query input
                template = Template(template=body_template)
                body = template.safe_substitute(model_name=model_name, query_input=query_input)
            else:
                # Fallback to plain text (legacy behavior)
                template = Template(template=query_input)
                body = template.safe_substitute(model_name=model_name)
        else:
            # For custom input, create OpenAI-compatible format
            if isinstance(inference_input, str):
                body = json.dumps({
                    "model": model_name,
                    "messages": [{"role": "user", "content": inference_input}],
                    "max_tokens": 100,
                    "temperature": 0.0,
                })
            else:
                body = json.dumps(inference_input)

        return body

    def generate_command(
        self,
        model_name: str,
        inference_input: str | None = None,
        use_default_query: bool = False,
        insecure: bool = False,
        token: str | None = None,
    ) -> str:
        """Generate curl command string for LLM inference."""
        base_url = get_llm_inference_url(llm_service=self.llm_service)
        endpoint_url = f"{base_url}{LLMEndpoint.CHAT_COMPLETIONS}"

        body = self.get_inference_body(
            model_name=model_name,
            inference_input=inference_input,
            use_default_query=use_default_query,
        )

        header = HTTPRequest.CONTENT_JSON.replace("-H ", "")
        cmd_exec = "curl -i -s"
        cmd = f"{cmd_exec} -X POST -d '{body}' -H {header} -H 'Accept: application/json'"

        if token:
            cmd += f" {HTTPRequest.AUTH_HEADER.format(token=token)}"

        if insecure:
            cmd += " --insecure"
        else:
            try:
                from ocp_resources.resource import get_client

                client = get_client()
                ca_bundle = get_ca_bundle(client=client, deployment_mode="raw")
                if ca_bundle:
                    cmd += f" --cacert {ca_bundle}"
                else:
                    cmd += " --insecure"
            except Exception:
                cmd += " --insecure"

        cmd += f" --max-time {LLMEndpoint.DEFAULT_TIMEOUT} {endpoint_url}"
        return cmd

    @retry(wait_timeout=Timeout.TIMEOUT_30SEC, sleep=5)
    def run_inference(
        self,
        model_name: str,
        inference_input: str | None = None,
        use_default_query: bool = False,
        insecure: bool = False,
        token: str | None = None,
    ) -> str:
        """Run inference command and return raw output."""
        cmd = self.generate_command(
            model_name=model_name,
            inference_input=inference_input,
            use_default_query=use_default_query,
            insecure=insecure,
            token=token,
        )

        res, out, err = run_command(command=shlex.split(cmd), verify_stderr=False, check=False)
        if res:
            return out
        raise ValueError(f"Inference failed with error: {err}\nOutput: {out}\nCommand: {cmd}")

    def run_inference_flow(
        self,
        model_name: str,
        inference_input: str | None = None,
        use_default_query: bool = False,
        insecure: bool = False,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Run LLM inference using the same high-level flow as inference_utils."""
        out = self.run_inference(
            model_name=model_name,
            inference_input=inference_input,
            use_default_query=use_default_query,
            insecure=insecure,
            token=token,
        )
        return {"output": out}


def _validate_unauthorized_response(res: dict[str, Any], token: str | None, inference: LLMUserInference) -> None:
    """Validate response for unauthorized users."""
    auth_header = "x-ext-auth-reason"

    if auth_reason := re.search(rf"{auth_header}: (.*)", res["output"], re.MULTILINE):
        reason = auth_reason.group(1).lower()

        if token:
            assert re.search(r"not (?:authenticated|authorized)", reason)
        else:
            assert "credential not found" in reason
    else:
        forbidden_patterns = ["Forbidden", "401", "403", "Unauthorized"]
        output = res["output"]

        if any(pattern in output for pattern in forbidden_patterns):
            return

        raise ValueError(f"Auth header {auth_header} not found in response. Response: {output}")


def _validate_authorized_response(
    res: dict[str, Any],
    inference: LLMUserInference,
    inference_config: dict[str, Any],
    inference_type: str,
    expected_response_text: str | None,
    use_default_query: bool,
    model_name: str,
) -> None:
    """Validate response for authorized users."""

    use_regex = False

    if use_default_query:
        expected_response_text_config = inference_config.get("default_query_model", {})
        use_regex = expected_response_text_config.get("use_regex", False)

        if not expected_response_text_config:
            raise ValueError(f"Missing default_query_model config for inference {inference_config}")

        if inference_config.get("support_multi_default_queries"):
            query_config = expected_response_text_config.get(inference_type)
            if not query_config:
                raise ValueError(f"Missing default_query_model config for inference type {inference_type}")
            expected_response_text = query_config.get("query_output", "")
            use_regex = query_config.get("use_regex", False)
        else:
            expected_response_text = expected_response_text_config.get("query_output")

        if not expected_response_text:
            raise ValueError(f"Missing response text key for inference {inference_config}")

        if isinstance(expected_response_text, str):
            expected_response_text = Template(template=expected_response_text).safe_substitute(model_name=model_name)
        elif isinstance(expected_response_text, dict):
            response_output = expected_response_text.get("response_output")
            if response_output is not None:
                expected_response_text = Template(template=response_output).safe_substitute(model_name=model_name)
    if inference.inference_response_text_key_name:
        if inference_type == inference.STREAMING:
            if output := re.findall(
                rf"{inference.inference_response_text_key_name}\": \"(.*)\"",
                res[inference.inference_response_key_name],
                re.MULTILINE,
            ):
                assert "".join(output) == expected_response_text, (
                    f"Expected: {expected_response_text} does not match response: {output}"
                )
        elif inference_type == inference.INFER or use_regex:
            formatted_res = json.dumps(res[inference.inference_response_text_key_name]).replace(" ", "")
            if use_regex and expected_response_text is not None:
                assert re.search(expected_response_text, formatted_res), (
                    f"Expected: {expected_response_text} not found in: {formatted_res}"
                )
            else:
                formatted_res = json.dumps(res[inference.inference_response_key_name]).replace(" ", "")
                assert formatted_res == expected_response_text, (
                    f"Expected: {expected_response_text} does not match output: {formatted_res}"
                )
        else:
            response = res[inference.inference_response_key_name]
            if isinstance(response, list):
                response = response[0]

            if isinstance(response, dict):
                response_text = response[inference.inference_response_text_key_name]
                assert response_text == expected_response_text, (
                    f"Expected: {expected_response_text} does not match response: {response_text}"
                )
            else:
                raise InferenceResponseError(
                    "Inference response output does not match expected output format."
                    f"Expected: {expected_response_text}.\nResponse: {res}"
                )
    else:
        raise InferenceResponseError(f"Inference response output not found in response. Response: {res}")
