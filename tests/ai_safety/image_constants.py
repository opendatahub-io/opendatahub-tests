class AiSafetyImages:
    """Container images used by ai_safety tests."""

    VLLM_EMULATOR: str = (
        "quay.io/trustyai_testing/vllm_emulator@sha256:c4bdd5bb93171dee5b4c8454f36d7c42b58b2a4ceb74f29dba5760ac53b5c12d"
    )
    MINIO_MC: str = "quay.io/minio/mc@sha256:470f5546b596e16c7816b9c3fa7a78ce4076bb73c2c73f7faeec0c8043923123"
    MINIO_SERVER: str = "quay.io/minio/minio@sha256:46b3009bf7041eefbd90bd0d2b38c6ddc24d20a35d609551a1802c558c1c958f"
    MINIO_SERVER_OTEL: str = (
        "quay.io/minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e"
    )
    MINIO_DSPA: str = "quay.io/opendatahub/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance"
    FLAN_T5: str = (
        "quay.io/trustyai_testing/lmeval-assets-flan-t5-base"
        "@sha256:f7326d5b4069e9aa0b12ab77b1e8aa8dd25dd0bffd77b08fcc84988ea8869f7f"
    )
    ARC_EASY_DATASET: str = (
        "quay.io/trustyai_testing/lmeval-assets-arc-easy"
        "@sha256:1558997a838f2ac8ecd887b4f77485d810e5120b9f2700ecb71627e37c6d3a1b"
    )
    NEWSGROUPS_DATASET: str = (
        "quay.io/trustyai_testing/lmeval-assets-20newsgroups"
        "@sha256:106023a7ee0c93afad5d27ae50130809ccc232298b903c8b12ea452e9faafce2"
    )
    NEMO_GUARDRAILS: str = "quay.io/opendatahub/odh-trustyai-nemo-guardrails-server:odh-incubation-linux-x86-64"
    GAUSSIAN_CREDIT_MODEL: str = (
        "oci://quay.io/trustyai_testing/gaussian-credit-model-modelcar"
        "@sha256:323dbb70c980c7f57bb6a884f5d46ee1c620c0b193368d13a469b49e7c9054c4"
    )
    MLSERVER: str = (
        "quay.io/trustyai_testing/mlserver@sha256:68a4cd74fff40a3c4f29caddbdbdc9e54888aba54bf3c5f78c8ffd577c3a1c89"
    )
