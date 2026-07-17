"""Base configuration class for LLMInferenceService resources."""

import pytest
import structlog
from kubernetes.dynamic import DynamicClient

from tests.model_serving.model_server.llmd.constants import LLMD_TESTS_SUPPORTED_ACCELERATORS
from tests.model_serving.model_server.llmd.utils import (
    detect_accelerators,
    find_matching_llminferenceserviceconfig,
    log_accelerator_selection,
    log_base_refs_selection,
)
from utilities.constants import ContainerImages, Labels

LOGGER = structlog.get_logger(name=__name__)


class LLMISvcConfig:
    """Base configuration for an LLMInferenceService resource.

    Subclass and override class attributes or classmethods for each test scenario.
    Pass the class directly to create_llmisvc_from_config — no instantiation needed.
    """

    name = ""
    model_name = None
    storage_uri = ""
    replicas = 1
    container_image = None
    enable_auth = False
    wait_timeout = 300
    base_refs = None

    @classmethod
    def container_resources(cls):
        return {}

    @classmethod
    def container_env(cls):
        """Base environment variables for the vLLM container.

        Subclasses may either:
        - Call super().container_env() + [...] to extend the base env vars (used by CpuConfig)
        - Return a fresh list to fully replace (used by prefix cache configs that need
          exclusive control over VLLM_ADDITIONAL_ARGS)
        """
        return [
            {"name": "VLLM_LOGGING_LEVEL", "value": "DEBUG"},
        ]

    @classmethod
    def liveness_probe(cls):
        return {
            "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
            "initialDelaySeconds": 240,
            "periodSeconds": 60,
            "timeoutSeconds": 60,
            "failureThreshold": 10,
        }

    @classmethod
    def readiness_probe(cls):
        return None

    @classmethod
    def router_config(cls):
        return {
            "scheduler": {"configRef": "kserve-config-llm-scheduler"},
            "route": {},
            "gateway": {},
        }

    @classmethod
    def annotations(cls):
        return {
            "prometheus.io/port": "8000",
            "prometheus.io/path": "/metrics",
            "security.opendatahub.io/enable-auth": str(cls.enable_auth).lower(),
        }

    @classmethod
    def prefill_config(cls):
        return None

    @classmethod
    def labels(cls):
        return {}

    @classmethod
    def describe(cls, namespace: str = ""):
        """Return a formatted config summary for log output."""
        border = "=" * 60
        lines = [
            border,
            f"  Config: {cls.__name__}",
            border,
            f"  namespace:       {namespace}",
            f"  name:            {cls.name}",
            f"  storage_uri:     {cls.storage_uri}",
            f"  replicas:        {cls.replicas}",
            f"  container_image: {cls.container_image or '(default)'}",
            f"  auth:            {cls.annotations().get('security.opendatahub.io/enable-auth', 'false')}",
            f"  resources:       {cls.container_resources() or '(none)'}",
            f"  wait_timeout:    {cls.wait_timeout}",
        ]
        return lines

    @classmethod
    def format_describe(cls, namespace: str = ""):
        """Return a formatted config summary for log output."""
        border = "=" * 60
        lines = cls.describe(namespace=namespace)
        lines.append(border + "\n")
        return "\n".join(lines)

    @classmethod
    def build(cls, client: DynamicClient) -> type:
        """No-op for non-GPU configs. GpuConfig overrides with actual detection."""
        LOGGER.info(f"No accelerator needed for {cls.__name__}")
        return cls

    @classmethod
    def with_overrides(cls, **overrides):
        """Create a derived config class with overridden attributes."""
        return type(f"{cls.__name__}_custom", (cls,), overrides)


class CpuConfig(LLMISvcConfig):
    """CPU inference base. Sets vLLM CPU image, CPU env vars, and CPU resource limits."""

    enable_auth = False
    wait_timeout = 420
    container_image = ContainerImages.VLLM.CPU

    @classmethod
    def container_env(cls):
        # vLLM arguments to reduce engine startup time
        # --max-num-seqs 20
        # --max-model-len 128
        # --enforce-eager
        return super().container_env() + [
            {
                "name": "VLLM_ADDITIONAL_ARGS",
                "value": "--max-num-seqs 20 --max-model-len 128 --enforce-eager --ssl-ciphers ECDHE+AESGCM:DHE+AESGCM",
            },
            {"name": "VLLM_CPU_KVCACHE_SPACE", "value": "4"},
        ]

    @classmethod
    def container_resources(cls):
        return {
            "limits": {"cpu": "1", "memory": "10Gi"},
            "requests": {"cpu": "100m", "memory": "8Gi"},
        }


class GpuConfig(LLMISvcConfig):
    """GPU inference base. Call build(client) to detect accelerator, skip, and bind."""

    enable_auth = False
    wait_timeout = 600

    # default GPU requirements
    min_gpus_per_node = 1
    min_nodes = 1
    supported_accelerators = LLMD_TESTS_SUPPORTED_ACCELERATORS

    # supported-topologies values must be consistent with the samples
    # API and Dashboard usage.  See odh-model-controller docs:
    # features/samples/llm-d/llm-d-ui-flow-mechanics.md
    # features/llm-d/samples/samples-api.md
    supported_topology = "workload-single-node"

    # Default accelerator resource name used when the cluster has NVIDIA GPUs.
    # Overridden by build() after accelerator detection (e.g. "amd.com/gpu" on AMD clusters).
    accelerator = "nvidia.com/gpu"

    # Excludes fast-image CRs by default. Fast image subclasses override
    # with a positive regex (e.g. ".*fast-1$").
    accelerator_config_name_regex = "^(?!.*fast-)"

    # When True, pytest.skip instead of pytest.fail when no CR matches.
    # Fast image subclasses set this to True.
    optional_base_refs = False

    # Resolved by _select_base_refs at build time; set explicitly to skip discovery.
    base_refs = None

    @classmethod
    def build(cls, client: DynamicClient) -> type:
        """Resolve all cluster-dependent config.

        1. Detect which GPU accelerator to use.
        2. Resolve which LLMInferenceServiceConfig CR (base_refs) to use.
        3. Return a derived config class with both bound.
        """
        accelerator = cls._select_accelerator(client=client)
        base_refs = (
            cls.base_refs
            if cls.base_refs is not None
            else cls._select_base_refs(client=client, accelerator=accelerator)
        )
        return cls.with_overrides(accelerator=accelerator, base_refs=base_refs)

    @classmethod
    def _select_accelerator(cls, client: DynamicClient) -> str:
        """Select the GPU accelerator to use for this test.

        Scans worker nodes, filters to ``cls.supported_accelerators``, and
        picks the type with the most total GPUs (node count as tiebreaker).
        Skips the test if no accelerator meets ``min_gpus_per_node`` / ``min_nodes``.

        Uses ``cls.supported_accelerators``, ``cls.min_gpus_per_node``,
        and ``cls.min_nodes`` to filter and qualify.

        Args:
            client: Kubernetes dynamic client.

        Returns:
            The accelerator resource name (e.g. ``nvidia.com/gpu``).
        """
        # detected_nodes is a list of {"name": node_name, "resources": {resource: count}}
        detected_nodes = detect_accelerators(client=client)

        # Keep only accelerator types that this config supports (e.g. nvidia.com/gpu, amd.com/gpu)
        supported_nodes = []
        for node in detected_nodes:
            supported = {
                resource_name: resource_count
                for resource_name, resource_count in node["resources"].items()
                if resource_name in cls.supported_accelerators
            }
            if supported:
                supported_nodes.append(supported)

        # For each accelerator type, count nodes that meet min_gpus_per_node and sum their GPUs.
        # Nodes with fewer GPUs than required are skipped — they can't run the workload.
        candidates: dict[str, dict[str, int]] = {}
        for node in supported_nodes:
            for resource_name, resource_count in node.items():
                if resource_name not in candidates:
                    candidates[resource_name] = {"total_gpus": 0, "qualifying_nodes": 0}
                if resource_count >= cls.min_gpus_per_node:
                    candidates[resource_name]["total_gpus"] += resource_count
                    candidates[resource_name]["qualifying_nodes"] += 1

        # Keep only accelerator types with enough qualifying nodes for the current test
        qualified = {
            resource_name: node_stats
            for resource_name, node_stats in candidates.items()
            if node_stats["qualifying_nodes"] >= cls.min_nodes
        }

        # Skip the test if no accelerator type meets the requirements
        if not qualified:
            msg = (
                f"Skipping test: no supported accelerator found for {cls.__name__}."
                f" Required: {cls.min_nodes} node(s) with at least {cls.min_gpus_per_node} GPU(s) each,"
                f" supported types: {cls.supported_accelerators}."
                f" Found: {candidates or 'no GPU nodes'}"
            )
            LOGGER.warning(msg)
            pytest.skip(msg)

        # Pick the accelerator type with the most GPUs, then most nodes as tiebreaker
        selected_resource = max(
            qualified,
            key=lambda resource_name: (
                qualified[resource_name]["total_gpus"],
                qualified[resource_name]["qualifying_nodes"],
            ),
        )
        selected_stats = qualified[selected_resource]
        log_accelerator_selection(
            config_name=cls.__name__,
            detected_nodes=detected_nodes,
            selected=selected_resource,
            total_gpus=selected_stats["total_gpus"],
            qualifying_nodes=selected_stats["qualifying_nodes"],
        )
        return selected_resource

    @classmethod
    def _select_base_refs(cls, client: DynamicClient, accelerator: str) -> list[dict[str, str]]:
        """Select the LLMInferenceServiceConfig CR (base_refs) for this test.

        - **NVIDIA** (non-optional): returns ``[]`` — the controller uses
          the built-in CUDA template.
        - **Non-NVIDIA** (non-optional): discovers the matching CR.
          Fails if not found (no built-in fallback).
        - **optional_base_refs=True**: discovers the CR.
          Skips if not found.

        Uses ``cls.accelerator_config_name_regex`` and ``cls.supported_topology``
        to filter CRs.

        Args:
            client: Kubernetes dynamic client.
            accelerator: The accelerator resource name from ``_select_accelerator``.

        Returns:
            ``[{"name": cr_name}]`` if a CR is selected, or ``[]`` for default CUDA.
        """
        is_nvidia = accelerator == Labels.Nvidia.NVIDIA_COM_GPU

        if is_nvidia and not cls.optional_base_refs:
            log_base_refs_selection(
                accelerator=accelerator,
                topology=cls.supported_topology,
                name_regex=cls.accelerator_config_name_regex,
            )
            return []

        result = find_matching_llminferenceserviceconfig(
            client=client,
            accelerator=accelerator,
            topology=cls.supported_topology,
            name_regex=cls.accelerator_config_name_regex,
        )
        log_base_refs_selection(
            accelerator=accelerator,
            topology=cls.supported_topology,
            name_regex=cls.accelerator_config_name_regex,
            result=result,
        )

        if result.matched:
            return [{"name": result.matched}]

        if cls.optional_base_refs:
            msg = (
                f"No LLMInferenceServiceConfig CR matching '{cls.accelerator_config_name_regex}'"
                f" for accelerator='{accelerator}' topology='{cls.supported_topology}'."
                f" optional_base_refs=True — skipping. See logs above for details."
            )
            LOGGER.warning(msg)
            pytest.skip(msg)

        pytest.fail(
            f"No LLMInferenceServiceConfig CR matched accelerator='{accelerator}'"
            f" topology='{cls.supported_topology}'. See logs above for details.",
            pytrace=False,
        )

    @classmethod
    def describe(cls, namespace: str = ""):
        """Extend base describe with GPU-specific fields."""
        lines = super().describe(namespace=namespace)
        lines += [
            f"  gpu_resource:    {cls.gpu_resource_name()}",
            f"  base_refs:       {cls.base_refs or '(default CUDA)'}",
        ]
        return lines

    @classmethod
    def gpu_resource_name(cls):
        """Return the Kubernetes GPU resource name (e.g. nvidia.com/gpu, amd.com/gpu)."""
        return cls.accelerator

    @classmethod
    def container_resources(cls):
        """Return resource requests and limits including the detected GPU resource."""
        gpu_name = cls.gpu_resource_name()
        return {
            "limits": {"cpu": "2", "memory": "16Gi", gpu_name: "1"},
            "requests": {"cpu": "1", "memory": "8Gi", gpu_name: "1"},
        }
