class AiSafetyImages:
    """Container images used by ai_safety tests."""

    VLLM_EMULATOR: str = (
        "quay.io/trustyai_testing/vllm_emulator@sha256:32b5f26b5ec1c5c8052afa26c6d9769dbc864df27a058716c8df62d827cf1d07"
    )
    MINIO_MC: str = (
        "quay.io/trustyai_testing/minio-mc@sha256:f857d815d4dfb95ccfb6e374cf949f3daebbe05a952d44bbbde3f2552d28e6c0"
    )
    MINIO_SERVER: str = (
        "quay.io/trustyai_testing/minio@sha256:cf222021b0727b0b3efe1794dd3f1af898071b684c1c67b1ef345ccef636501a"
    )
    MINIO_SERVER_OTEL: str = (
        "quay.io/minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e"
    )
    MINIO_DSPA: str = "quay.io/opendatahub/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance"
    SIMPLE_MINIO: str = (
        "quay.io/opendatahub/minio@sha256:587abc14be9bbeed794473cf7290c40e377062f2f77f5e4e27742a77680f08e0"
    )
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
    LOAN_MODEL_ALPHA: str = (
        "oci://quay.io/trustyai_testing/loan-model-alpha-modelcar"
        "@sha256:519c05826b987615f0f12cb341715060108054fef88a462c9902084992af3054"
    )
    MLSERVER: str = (
        "quay.io/trustyai_testing/mlserver@sha256:68a4cd74fff40a3c4f29caddbdbdc9e54888aba54bf3c5f78c8ffd577c3a1c89"
    )
