# Model Serving Tests

This directory contains the most comprehensive test suite in the repository, covering all aspects of model serving functionality in OpenDataHub/RHOAI. It validates model runtimes, model server configurations, storage backends, deployment modes, and MaaS (Model as a Service) billing.

## Directory Structure

```text
model_serving/
├── conftest.py                        # Module-level fixtures (S3 secrets, protocols)
│
├── maas_billing/                      # MaaS billing and subscription tests
│   ├── conftest.py
│   ├── utils.py
│   ├── test_maas_endpoints.py         # /v1/models, /v1/chat/completions
│   ├── test_maas_token_*.py           # Token minting and revocation
│   ├── test_maas_*_rate_limits.py     # Request and token-based rate limiting
│   ├── test_maas_rbac_e2e.py          # Multi-tier user access control
│   └── maas_subscription/             # Subscription and API key management
│       ├── conftest.py
│       ├── test_api_key_*.py          # API key CRUD and authorization
│       ├── test_*_subscriptions_*.py  # Multi-subscription enforcement
│       ├── test_cascade_deletion.py   # Cascade deletion
│       └── component_health/          # MaaS controller and API health
│
├── model_runtime/                     # Runtime validation tests
│   ├── conftest.py
│   ├── utils.py
│   ├── image_validation/              # Runtime image validation
│   ├── mlserver/                      # MLServer runtime tests
│   │   └── basic_model_deployment/    # LightGBM, ONNX, Sklearn, XGBoost
│   ├── model_validation/             # Model validation tests
│   ├── openvino/                      # OpenVINO Model Server (OVMS) tests
│   │   ├── smoke/                     # Smoke test scripts
│   │   ├── test_ovms_smoke.py
│   │   └── test_ovms_model_deployment.py
│   ├── rhoai_upgrade/                 # RHOAI upgrade tests
│   ├── triton/                        # NVIDIA Triton runtime tests
│   │   └── basic_model_deployment/    # PyTorch, ONNX, TF, Keras, Python, FIL, DALI
│   └── vllm/                          # vLLM runtime tests (LLMs)
│       ├── basic_model_deployment/    # Granite, Llama, Merlinite models
│       ├── multimodal/                # Vision models (Granite 3.1 2B)
│       ├── quantization/              # AWQ quantization
│       ├── speculative_decoding/      # Draft and n-gram decoding
│       └── toolcalling/               # Function calling tests
│
└── model_server/                      # Server platform tests
    ├── conftest.py
    ├── components/                    # Component co-existence tests
    ├── kserve/                        # KServe-specific tests
    │   ├── authentication/            # Auth configuration
    │   ├── autoscaling/               # KEDA and Kueue autoscaling
    │   ├── inference_graph/           # Inference pipeline tests
    │   ├── inference_service_configuration/
    │   ├── inference_service_lifecycle/  # Replicas, env vars, stop/resume
    │   ├── ingress/                   # Route visibility, reconciliation
    │   ├── negative/                  # Missing fields, malformed JSON
    │   ├── observability/             # Metrics and monitoring
    │   ├── platform/                  # DSC deployment modes
    │   ├── private_endpoint/          # Private endpoint access
    │   └── storage/                   # S3, PVC, OCI, MinIO backends
    ├── llmd/                          # LLM Deployment (LLMD) tests
    │   ├── llmd_configs/              # LLMD configuration files
    │   └── test_llmd_*.py             # Smoke, auth, CPU/GPU, scheduler
    └── upgrade/                       # Upgrade tests
        └── test_upgrade*.py           # Metrics, auth, private endpoint, llmd
```

### Current Test Suites

- **`maas_billing/`** - MaaS billing tests including token management, rate limiting, RBAC, subscription lifecycle, API key CRUD/authorization, and cascade deletion
- **`model_runtime/`** - Runtime validation for vLLM (LLM serving with GPU), OpenVINO (CPU-optimized inference), Triton (multi-framework), and MLServer (lightweight serving). Covers basic deployment, multimodal, quantization, speculative decoding, and tool calling
- **`model_server/`** - Server platform tests for KServe deployment modes (raw, serverless), storage backends (S3, PVC, OCI, MinIO), authentication, autoscaling (KEDA, Kueue), inference graphs, lifecycle management, observability, negative testing, LLMD, and upgrade scenarios

## Test Markers

```python
@pytest.mark.smoke                 # Critical smoke tests
@pytest.mark.tier1                 # Tier 1 tests
@pytest.mark.tier2                 # Tier 2 tests
@pytest.mark.rawdeployment         # KServe raw deployment mode
@pytest.mark.gpu                   # Requires GPU
@pytest.mark.multinode             # Multi-node deployment
@pytest.mark.minio                 # MinIO storage tests
@pytest.mark.tls                   # TLS/SSL tests
@pytest.mark.metrics               # Metrics tests
@pytest.mark.kueue                 # Kueue integration
@pytest.mark.pre_upgrade           # Pre-upgrade tests
@pytest.mark.post_upgrade          # Post-upgrade tests
@pytest.mark.skip_on_disconnected  # Requires internet connectivity
```

## Model Runtimes

| Runtime         | Framework       | Use Case                                                |
| --------------- | --------------- | ------------------------------------------------------- |
| vLLM            | LLM             | GPU-accelerated LLM serving (Granite, Llama, Merlinite) |
| OpenVINO (OVMS) | General ML      | CPU-optimized inference                                 |
| Triton          | Multi-framework | PyTorch, ONNX, TensorFlow, Keras, Python backend        |
| MLServer        | Lightweight     | LightGBM, ONNX, Sklearn, XGBoost                        |

## Storage Backends

| Backend | Description                        |
| ------- | ---------------------------------- |
| S3      | AWS S3-compatible object storage   |
| MinIO   | Self-hosted S3-compatible storage  |
| PVC     | Kubernetes PersistentVolumeClaim   |
| OCI     | OCI container registry             |

## Running Tests

### Run All Model Serving Tests

```bash
uv run pytest tests/model_serving/
```

### Run Tests by Component

```bash
# Run vLLM runtime tests
uv run pytest tests/model_serving/model_runtime/vllm/

# Run OpenVINO tests
uv run pytest tests/model_serving/model_runtime/openvino/

# Run KServe platform tests
uv run pytest tests/model_serving/model_server/kserve/

# Run MaaS billing tests
uv run pytest tests/model_serving/maas_billing/

# Run LLMD tests
uv run pytest tests/model_serving/model_server/llmd/
```

### Run Tests with Markers

```bash
# Run smoke tests
uv run pytest -m smoke tests/model_serving/

# Run GPU tests only
uv run pytest -m gpu tests/model_serving/

# Run raw deployment tests
uv run pytest -m rawdeployment tests/model_serving/

# Run tests excluding GPU
uv run pytest -m "not gpu" tests/model_serving/
```

## Additional Resources

- [KServe Documentation](https://kserve.github.io/website/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
