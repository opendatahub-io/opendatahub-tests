# Kueue Configuration
SINGLE_JOB_CPU_QUOTA = "2"
SINGLE_JOB_MEMORY_QUOTA = "4Gi"
MULTI_JOB_CPU_QUOTA = "8"
MULTI_JOB_MEMORY_QUOTA = "16Gi"

# vLLM emulator configuration
VLLM_EMULATOR = "vllm-emulator"
# Pin by digest for reproducible test results (same image as multitenancy tests)
VLLM_EMULATOR_IMAGE = (
    "quay.io/trustyai_testing/vllm_emulator@sha256:c4bdd5bb93171dee5b4c8454f36d7c42b58b2a4ceb74f29dba5760ac53b5c12d"
)
