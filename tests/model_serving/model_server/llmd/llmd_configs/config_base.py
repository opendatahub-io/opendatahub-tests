"""Base configuration class for LLMInferenceService resources."""

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_service_version import ClusterServiceVersion

from tests.model_serving.model_server.llmd.constants import AMD_ROCM_TEMPLATE, LLMD_TESTS_SUPPORTED_ACCELERATORS
from tests.model_serving.model_server.llmd.utils import detect_accelerators
from utilities.constants import Labels
from utilities.llmd_constants import ContainerImages

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
    wait_timeout = 240
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
    container_image = ContainerImages.VLLM_CPU

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

    # defaults, overridden by build() after accelerator detection
    accelerator = "nvidia.com/gpu"
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
        min_gpus_per_node and min_nodes requirements.

        When multiple accelerator types qualify, picks the one with the most
        total GPUs available, using node count as tiebreaker.

        For AMD GPUs, resolves the ROCm-specific base ref template from the RHOAI CSV version.
        If base_refs is already set on the config class, the existing value is preserved.

        Returns a derived config class with accelerator and base_refs bound.
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

        if selected_resource == Labels.ROCm.ROCM_GPU:
            base_refs = cls.base_refs or cls._resolve_base_refs(client=client, template_name=AMD_ROCM_TEMPLATE)
        else:
            base_refs = cls.base_refs or []

        LOGGER.info(
            f"[llmd] Selected {selected_resource}:"
            f" {selected_stats['total_gpus']} GPU(s) on {selected_stats['qualifying_nodes']} node(s),"
            f" base_refs: {base_refs or '(default CUDA)'}"
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

    @staticmethod
    def _resolve_base_refs(client: DynamicClient, template_name: str) -> list[dict[str, str]]:
        """Resolve a baseRef template name to a versioned CR name.

        LLMInferenceServiceConfig CRs use versioned names (e.g.
        "v3-4-0-ea-2-kserve-config-llm-template-amd-rocm"). We read the RHOAI CSV
        version and prepend it to the template name.
        """
        rhoai_version = None
        for csv in ClusterServiceVersion.get(client=client, namespace="redhat-ods-operator"):
            if csv.name.startswith("rhods-operator") and csv.status == csv.Status.SUCCEEDED:
                rhoai_version = csv.instance.spec.version
                LOGGER.info(f"[llmd] Found RHOAI CSV: {csv.name}, version: {rhoai_version}")
                break

        if not rhoai_version:
            raise ValueError("RHOAI CSV (rhods-operator) not found in redhat-ods-operator namespace")

        version_prefix = f"v{rhoai_version.replace('.', '-')}"
        full_name = f"{version_prefix}-{template_name}"
        LOGGER.info(f"[llmd] Resolved baseRef: {template_name} -> {full_name}")
        return [{"name": full_name}]

    @classmethod
    def container_resources(cls):
        """Return resource requests and limits including the detected GPU resource."""
        gpu_name = cls.gpu_resource_name()
        return {
            "limits": {"cpu": "2", "memory": "16Gi", gpu_name: "1"},
            "requests": {"cpu": "1", "memory": "8Gi", gpu_name: "1"},
        }
