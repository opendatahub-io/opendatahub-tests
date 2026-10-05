class FixturesImages:
    """Container images used by shared test fixtures."""

    LLMD_INFERENCE_SIM: str = (
        "quay.io/trustyai_testing/llm-d-inference-sim-dataset-builtin"
        "@sha256:79e525cfd57a0d72b7e71d5f1e2dd398eca9315cfbd061d9d3e535b1ae736239"
    )
    VLLM_CUDA: str = (
        "registry.redhat.io/rhaiis/vllm-cuda-rhel9"
        "@sha256:ec799bb5eeb7e25b4b25a8917ab5161da6b6f1ab830cbba61bba371cffb0c34d"
    )
    QWEN_25_3B_INSTRUCT: str = (
        "oci://quay.io/trustyai_testing/models/qwen2.5-3b-instruct"
        "@sha256:6f9d9843599a9959de23c76d6b5adb556505482a7e732b2fcbca695a9c4ce545"
    )
    VLLM_CPU: str = (
        "quay.io/rh-aiservices-bu/vllm-cpu-openai-ubi9"
        "@sha256:ada6b3ba98829eb81ae4f89364d9b431c0222671eafb9a04aa16f31628536af2"
    )
    MILVUS: str = "docker.io/milvusdb/milvus@sha256:3d772c3eae3a6107b778636cea5715b9353360b92e5dcfdcaf4ca7022f4f497c"
    ETCD: str = "quay.io/coreos/etcd@sha256:3397341272b9e0a6f44d7e3fc7c321c6efe6cbe82ce866b9b01d0c704bfc5bf3"
    PGVECTOR: str = (
        "docker.io/pgvector/pgvector@sha256:0a07c4114ba6d1d04effcce3385e9f5ce305eb02e56a3d35948a415a52f193ec"
    )
    QDRANT: str = "docker.io/qdrant/qdrant@sha256:9dfabc51ededc48158899a288a19a04de1ab54a11d8c512e1c40eebbd5e2bc92"
