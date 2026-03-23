# Model Explainability Tests

This directory contains tests for AI/ML model explainability, trustworthiness, evaluation, and safety components in OpenDataHub/RHOAI. It covers TrustyAI Service, Guardrails Orchestrator, LM Eval, EvalHub, and the TrustyAI Operator.

## Directory Structure

```text
model_explainability/
├── conftest.py                          # Shared fixtures (PVC, TrustyAI configmap)
├── utils.py                             # Image validation utilities
│
├── evalhub/                             # EvalHub service tests
│   ├── conftest.py
│   ├── constants.py
│   ├── test_evalhub_health.py           # Health endpoint validation
│   └── utils.py
│
├── guardrails/                          # AI Safety Guardrails tests
│   ├── conftest.py                      # Detectors, Tempo, OpenTelemetry fixtures
│   ├── constants.py
│   ├── test_guardrails.py               # Built-in, HuggingFace, autoconfig tests
│   ├── upgrade/
│   │   └── test_guardrails_upgrade.py   # Pre/post-upgrade tests
│   └── utils.py
│
├── lm_eval/                             # Language Model Evaluation tests
│   ├── conftest.py                      # LMEvalJob fixtures (HF, local, vLLM, S3, OCI)
│   ├── constants.py                     # Task definitions (UNITXT, LLMAAJ)
│   ├── data/                            # Test data files
│   ├── test_lm_eval.py                  # HuggingFace, offline, vLLM, S3 tests
│   └── utils.py
│
├── trustyai_operator/                   # TrustyAI Operator validation
│   ├── test_trustyai_operator.py        # Operator image validation
│   └── utils.py
│
└── trustyai_service/                    # TrustyAI Service core tests
    ├── conftest.py                      # MariaDB, KServe, ISVC fixtures
    ├── constants.py                     # Storage configs, model formats
    ├── trustyai_service_utils.py        # TrustyAI REST client, metrics validation
    ├── utils.py                         # Service creation, RBAC, MariaDB utilities
    │
    ├── drift/                           # Drift detection tests
    │   ├── model_data/                  # Test data batches
    │   └── test_drift.py                # Meanshift, KSTest, ApproxKSTest, FourierMMD
    │
    ├── fairness/                        # Fairness metrics tests
    │   ├── conftest.py
    │   ├── model_data/                  # Fairness test data
    │   └── test_fairness.py             # SPD, DIR fairness metrics
    │
    ├── service/                         # Core service tests
    │   ├── conftest.py
    │   ├── test_trustyai_service.py     # Image validation, DB migration, DB cert tests
    │   ├── utils.py
    │   └── multi_ns/                    # Multi-namespace tests
    │       └── test_trustyai_service_multi_ns.py
    │
    └── upgrade/                         # Upgrade compatibility tests
        └── test_trustyai_service_upgrade.py
```

### Current Test Suites

- **`evalhub/`** - EvalHub service health endpoint validation via kube-rbac-proxy
- **`guardrails/`** - Guardrails Orchestrator tests with built-in regex detectors (PII), HuggingFace detectors (prompt injection, HAP), auto-configuration, and gateway routing. Includes OpenTelemetry/Tempo trace integration
- **`lm_eval/`** - Language Model Evaluation tests covering HuggingFace models, local/offline tasks, vLLM integration, S3 storage, and OCI registry artifacts
- **`trustyai_operator/`** - TrustyAI operator container image validation (SHA256 digests, CSV relatedImages)
- **`trustyai_service/`** - TrustyAI Service tests for drift detection (4 metrics), fairness metrics (SPD, DIR), database migration, multi-namespace support, and upgrade scenarios. Tests run against both PVC and database storage backends

## Test Markers

```python
@pytest.mark.model_explainability  # Module-level marker
@pytest.mark.smoke                 # Critical smoke tests
@pytest.mark.tier1                 # Tier 1 tests
@pytest.mark.tier2                 # Tier 2 tests
@pytest.mark.pre_upgrade           # Pre-upgrade tests
@pytest.mark.post_upgrade          # Post-upgrade tests
@pytest.mark.rawdeployment         # KServe raw deployment mode
@pytest.mark.skip_on_disconnected  # Requires internet connectivity
```

## Running Tests

### Run All Model Explainability Tests

```bash
pytest tests/model_explainability/
```

### Run Tests by Component

```bash
# Run TrustyAI Service tests
pytest tests/model_explainability/trustyai_service/

# Run Guardrails tests
pytest tests/model_explainability/guardrails/

# Run LM Eval tests
pytest tests/model_explainability/lm_eval/

# Run EvalHub tests
pytest tests/model_explainability/evalhub/
```

### Run Tests with Markers

```bash
# Run only smoke tests
pytest -m "model_explainability and smoke" tests/model_explainability/

# Run drift detection tests
pytest tests/model_explainability/trustyai_service/drift/

# Run fairness tests
pytest tests/model_explainability/trustyai_service/fairness/
```

## Additional Resources

- [TrustyAI Documentation](https://github.com/trustyai-explainability)
- [OpenDataHub Documentation](https://opendatahub.io/docs)
- [OpenShift AI Documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed)
