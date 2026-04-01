"""Utilities for LLM Deployment (LLMD) resources."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from timeout_sampler import TimeoutWatch

from utilities.constants import Timeout
from utilities.infra import get_services_by_isvc_label, is_disconnected_cluster
from utilities.llmd_constants import (
    ContainerImages,
    KServeGateway,
    LLMDGateway,
)

LOGGER = structlog.get_logger(name=__name__)


@contextmanager
def create_llmd_gateway(
    client: DynamicClient,
    name: str = LLMDGateway.DEFAULT_NAME,
    namespace: str = LLMDGateway.DEFAULT_NAMESPACE,
    gateway_class_name: str = LLMDGateway.DEFAULT_GATEWAY_CLASS,
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
    existing_gateway = Gateway(
        client=client,
        name=name,
        namespace=namespace,
        api_group=KServeGateway.API_GROUP,
    )
    if existing_gateway.exists:
        LOGGER.info(f"Reusing existing Gateway {name} in namespace {namespace}")
        if wait_for_condition:
            existing_gateway.wait_for_condition(
                condition="Programmed",
                status="True",
                timeout=timeout,
            )
        yield existing_gateway
    elif is_disconnected_cluster(client=client):
        raise RuntimeError(
            f"Gateway {name} in namespace {namespace} does not exist on a disconnected cluster. "
            "The gateway must be pre-created by CI using configure-disconnected-llmd-gateway.sh."
        )
    else:
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
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"Could not get service for LLMInferenceService {llm_service.name}: {e}")
    fallback_url = f"http://{llm_service.name}.{llm_service.namespace}.svc.cluster.local"
    LOGGER.debug(f"Using fallback URL for {llm_service.name}: {fallback_url}")
    return fallback_url
