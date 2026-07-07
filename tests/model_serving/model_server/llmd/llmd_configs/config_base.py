"""Base configuration class for LLMInferenceService resources."""

import pytest
import structlog
from kubernetes.dynamic import DynamicClient

from tests.model_serving.model_server.llmd.constants import LLMD_TESTS_SUPPORTED_ACCELERATORS
from tests.model_serving.model_server.llmd.utils import detect_accelerators, list_matching_accelerator_configs
from utilities.constants import ContainerImages, Labels
from utilities.infra import get_dsci_applications_namespace

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
        LOGGER.info(f"[llmd] No accelerator needed for {cls.__name__}")
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

    # Negative lookahead that excludes fast-image CRs (e.g. "…fast-1",
    # "…fast-2") so that standard GPU configs only match the regular
    # CUDA/ROCm templates.  Used as the default for
    # accelerator_config_name_regex below and as sentinel in
    # _resolve_accelerator to distinguish standard configs from fast
    # image configs that override the regex.
    _DEFAULT_ACCELERATOR_CONFIG_NAME_REGEX = "^(?!.*fast-)"

    # Regex matched against LLMInferenceServiceConfig CR names during
    # discovery.  Fast image subclasses override this with a positive
    # regex (e.g. ".*fast-1$").
    accelerator_config_name_regex = _DEFAULT_ACCELERATOR_CONFIG_NAME_REGEX

    # Resolved by _resolve_base_refs at build time; set explicitly to skip discovery.
    base_refs = None

    @classmethod
    def build(cls, client: DynamicClient) -> type:
        """Resolve all cluster-dependent config."""
        return cls._resolve_accelerator(client=client)

    @classmethod
    def _resolve_accelerator(cls, client: DynamicClient) -> type:
        """Detect cluster GPU accelerators and resolve the config for the best match.

        Scans worker nodes for GPU resources, filters to types supported by the
        current test's config class, and skips the test if the cluster doesn't meet
        ``min_gpus_per_node`` and ``min_nodes`` requirements.

        When multiple accelerator types qualify, picks the one with the most
        total GPUs available, using node count as tiebreaker.

        After selecting the accelerator, discovers the matching
        LLMInferenceServiceConfig CR via ``_resolve_base_refs`` (annotation-based
        discovery filtered by ``accelerator_config_name_regex``).  For NVIDIA GPUs,
        an empty result falls through to the default CUDA template; for non-NVIDIA
        accelerators (e.g. AMD ROCm), a missing CR is a hard failure.

        Returns a derived config class with ``accelerator`` and ``base_refs`` bound.
        """
        # node_accelerators is a list where each entry is a worker node's GPU resources,
        # e.g. [{"amd.com/gpu": 8}, {"amd.com/gpu": 8}]
        node_accelerators = detect_accelerators(client=client)

        # Keep only accelerator types that this config supports (e.g. nvidia.com/gpu, amd.com/gpu)
        supported_nodes = []
        for node in node_accelerators:
            supported = {
                resource_name: resource_count
                for resource_name, resource_count in node.items()
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
            pytest.skip(
                f"Skipping test: no supported accelerator found for {cls.__name__}."
                f" Required: {cls.min_gpus_per_node} GPU(s)/node on {cls.min_nodes} node(s),"
                f" supported types: {cls.supported_accelerators}."
                f" Found: {candidates or 'none'}"
            )

        # Pick the accelerator type with the most GPUs, then most nodes as tiebreaker
        selected_resource = max(
            qualified,
            key=lambda resource_name: (
                qualified[resource_name]["total_gpus"],
                qualified[resource_name]["qualifying_nodes"],
            ),
        )
        selected_stats = qualified[selected_resource]

        # Discover the LLMInferenceServiceConfig CR matching this config's accelerator,
        # topology, and name regex.  Uses annotation-based discovery via _resolve_base_refs.
        # If base_refs is already set explicitly on the config class, discovery is skipped.
        base_refs = cls.base_refs or cls._resolve_base_refs(client=client)

        if len(base_refs) == 0:
            _is_default_regex = cls.accelerator_config_name_regex == cls._DEFAULT_ACCELERATOR_CONFIG_NAME_REGEX
            _is_nvidia = selected_resource == Labels.Nvidia.NVIDIA_COM_GPU

            _no_cr_msg = (
                f"No LLMInferenceServiceConfig CR found"
                f" for {cls.__name__}."
                f" accelerator_config_name_regex:"
                f" '{cls.accelerator_config_name_regex}',"
                f" accelerator: '{selected_resource}',"
                f" topology: '{cls.supported_topology}'."
                f" The cluster does not have an"
                f" LLMInferenceServiceConfig CR with"
                f" opendatahub.io/recommended-accelerators"
                f" containing '{selected_resource}'"
                f" AND opendatahub.io/supported-topologies"
                f" containing '{cls.supported_topology}'"
                f" AND name matching regex"
                f" '{cls.accelerator_config_name_regex}'."
                f" Hardware: {cls.min_gpus_per_node} GPU(s)/node"
                f" on {cls.min_nodes} node(s),"
                f" supported types:"
                f" {cls.supported_accelerators}."
                f" Candidates found on cluster:"
                f" {candidates or 'none'}"
            )

            if not _is_default_regex:
                # Config targets a specific CR variant (e.g. fast-1,
                # fast-2) that is not present on this cluster.  Skip
                # rather than silently falling back to a wrong template.
                pytest.skip(_no_cr_msg)
            elif not _is_nvidia:
                # Non-NVIDIA accelerator (e.g. AMD ROCm) with standard
                # regex: a matching CR is required because there is no
                # built-in fallback template.  This is a hard failure.
                pytest.fail(_no_cr_msg)
            # else: standard NVIDIA with default regex — fall through
            # to the default CUDA template (no CR override needed).

        LOGGER.info(
            f"[llmd] Selected accelerator for {cls.__name__}: {selected_resource}"
            f" ({selected_stats['total_gpus']} GPU(s) on"
            f" {selected_stats['qualifying_nodes']} node(s)),"
            f" base_refs: {base_refs or '(default CUDA — no CR override)'}"
        )
        return cls.with_overrides(
            accelerator=selected_resource,
            base_refs=base_refs,
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
    def _resolve_base_refs(cls, client: DynamicClient) -> list[dict[str, str]]:
        """Discover the LLMInferenceServiceConfig CR matching this config's requirements.

        Searches the DSCI applications namespace for LLMInferenceServiceConfig CRs
        and filters by three criteria:

        1. **Name regex** (``cls.accelerator_config_name_regex``): the CR name must
           match this regex.  The default ``^(?!.*fast-)`` excludes fast-image CRs;
           fast image subclasses override with a positive pattern like ``.*fast-1$``.
        2. **Accelerator annotation** (``opendatahub.io/recommended-accelerators``):
           the JSON array must contain ``cls.accelerator`` (e.g. ``nvidia.com/gpu``).
        3. **Topology annotation** (``opendatahub.io/supported-topologies``):
           the JSON array must contain ``cls.supported_topology``
           (e.g. ``workload-single-node``).

        Returns:
            A single-element list ``[{"name": cr_name}]`` if a matching CR is found,
            or an empty list if no CR matches.  The caller decides whether to fail
            (non-NVIDIA) or fall back to the default CUDA template (NVIDIA).
        """
        namespace = get_dsci_applications_namespace(client=client)
        LOGGER.info(
            f"[llmd] Resolving base_refs for {cls.__name__}:"
            f" accelerator='{cls.accelerator}',"
            f" topology='{cls.supported_topology}',"
            f" name_regex='{cls.accelerator_config_name_regex}',"
            f" namespace='{namespace}'"
        )
        result = list_matching_accelerator_configs(
            client=client,
            namespace=namespace,
            accelerator=cls.accelerator,
            topology=cls.supported_topology,
            name_regex=cls.accelerator_config_name_regex,
        )
        if result.name:
            LOGGER.info(f"[llmd] Resolved base_refs for {cls.__name__}: [{{name: {result.name}}}]")
            return [{"name": result.name}]
        LOGGER.warning(
            f"[llmd] No matching CR found for {cls.__name__}."
            f" All CRs in '{namespace}': {result.all_cr_names or 'none'}."
            f" Candidates after regex/annotation filtering: {result.candidates or 'none'}"
        )
        return []

    @classmethod
    def container_resources(cls):
        """Return resource requests and limits including the detected GPU resource."""
        gpu_name = cls.gpu_resource_name()
        return {
            "limits": {"cpu": "2", "memory": "16Gi", gpu_name: "1"},
            "requests": {"cpu": "1", "memory": "8Gi", gpu_name: "1"},
        }
