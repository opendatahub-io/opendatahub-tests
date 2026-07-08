# Kueue Configuration
# NOTE: Lowering SINGLE_JOB_CPU_QUOTA / SINGLE_JOB_MEMORY_QUOTA is one way to try to ensure a
# second job is inadmissible while the first holds the quota. However, this is NOT a reliable
# approach — EvalHub jobs come in various sizes and a large job may exceed even the "single-job"
# quota, preventing even the first job from being admitted. Use stopPolicy: HoldAndDrain on the
# ClusterQueue instead for a deterministic inadmissible state independent of job resource requests.
SINGLE_JOB_CPU_QUOTA = "400m"
SINGLE_JOB_MEMORY_QUOTA = "700Mi"
MULTI_JOB_CPU_QUOTA = "8"
MULTI_JOB_MEMORY_QUOTA = "16Gi"

# vLLM emulator configuration
VLLM_EMULATOR = "vllm-emulator"
# Pin by digest for reproducible test results (same image as multitenancy tests)
VLLM_EMULATOR_IMAGE = (
    "quay.io/trustyai_testing/vllm_emulator@sha256:c4bdd5bb93171dee5b4c8454f36d7c42b58b2a4ceb74f29dba5760ac53b5c12d"
)
