# Model Explainability Tests

This directory contains tests for AI/ML model explainability, trustworthiness, evaluation, and safety components in OpenDataHub/RHOAI. It covers TrustyAI Service, Guardrails Orchestrator, LM Eval, EvalHub, and the TrustyAI Operator.

## Directory Structure

```text
ai_safety/
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
├── nemo_guardrails/                     # NeMo Guardrails tests
│   ├── conftest.py                      # NeMo CR, ConfigMap, Secret fixtures
│   ├── constants.py                     # Test data, entity types, policies
│   ├── test_nemo_guardrails.py          # API, chat/completions, guardrail/checks, multi-server tests
│   ├── utils.py                         # Config generation, request helpers
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
- **`nemo_guardrails/`** - NeMo Guardrails tests for LLM-as-a-judge (self-check policies), Presidio PII detection (email, SSN, credit card, person names), multi-server deployments, multi-configuration servers, authentication (kube-rbac-proxy), and secret mounting for API tokens
- **`trustyai_operator/`** - TrustyAI operator container image validation (SHA256 digests, CSV relatedImages)
- **`trustyai_service/`** - TrustyAI Service tests for drift detection (4 metrics), fairness metrics (SPD, DIR), database migration, multi-namespace support, and upgrade scenarios. Tests run against both PVC and database storage backends

## Test Markers

```python
@pytest.mark.ai_safety  # Module-level marker
@pytest.mark.smoke                 # Critical smoke tests
@pytest.mark.tier1                 # Tier 1 tests
@pytest.mark.tier2                 # Tier 2 tests
@pytest.mark.pre_upgrade           # Pre-upgrade tests
@pytest.mark.post_upgrade          # Post-upgrade tests
@pytest.mark.rawdeployment         # KServe raw deployment mode
@pytest.mark.skip_on_disconnected  # Requires internet connectivity
@pytest.mark.nemo_guardrails       # NeMo Guardrails specific tests
```

## Running Tests

### Run All Model Explainability Tests

```bash
uv run pytest tests/ai_safety/
```

### Run Tests by Component

```bash
# Run TrustyAI Service tests
uv run pytest tests/ai_safety/trustyai_service/

# Run Guardrails Orchestrator tests
uv run pytest tests/ai_safety/guardrails/

# Run NeMo Guardrails tests
uv run pytest tests/ai_safety/nemo_guardrails/

# Run LM Eval tests
uv run pytest tests/ai_safety/lm_eval/

# Run EvalHub tests
uv run pytest tests/ai_safety/evalhub/
```

### Run Tests with Markers

```bash
# Run only smoke tests
uv run pytest -m "ai_safety and smoke" tests/ai_safety/

# Run drift detection tests
uv run pytest tests/ai_safety/trustyai_service/drift/

# Run fairness tests
uv run pytest tests/ai_safety/trustyai_service/fairness/
```

## Upgrade Testing

Upgrade tests validate that AI Safety components continue to function correctly after an OpenShift AI platform upgrade. These tests use a pre-upgrade/post-upgrade pattern with pytest markers to ensure service continuity and data persistence across upgrades.

### Running Upgrade Tests

Upgrade tests run in two phases:

1. **Pre-upgrade phase** - Run before the platform upgrade to establish baseline state:

   ```bash
   uv run pytest -m pre_upgrade tests/ai_safety/
   ```

2. **Post-upgrade phase** - Run after the platform upgrade to verify persistence and functionality:

   ```bash
   uv run pytest -m post_upgrade tests/ai_safety/
   ```

### Upgrade Test Coverage

#### Guardrails Orchestrator

**Location:** `tests/ai_safety/guardrails/upgrade/test_guardrails_upgrade.py`

**Test Classes:**

- `TestGuardrailsOrchestratorWithBuiltInDetectorsPreUpgrade`
- `TestGuardrailsOrchestratorWithBuiltInDetectorsPostUpgrade`

**Covered Upgrade Paths:**

- Built-in detector persistence (regex, PII detection)
  - Pre-upgrade: Deploy orchestrator with built-in regex detectors for email and SSN detection
  - Post-upgrade: Verify detectors continue to function, health endpoints remain responsive
  - Validated: Input detection, output detection, passthrough routing, health/info endpoints

**What's Validated:**

- Orchestrator health and info endpoints remain responsive after upgrade
- Built-in regex detectors continue detecting unsuitable input/output
- Gateway routing and passthrough functionality persists
- Configuration and detector settings survive the upgrade

**Example:**

```bash
# Pre-upgrade
uv run pytest -m pre_upgrade tests/ai_safety/guardrails/upgrade/

# Perform platform upgrade

# Post-upgrade
uv run pytest -m post_upgrade tests/ai_safety/guardrails/upgrade/
```

#### TrustyAI Service

**Location:** `tests/ai_safety/trustyai_service/upgrade/test_trustyai_service_upgrade.py`

**Test Classes:**

- `TestPreUpgradeTrustyAIService` - PVC storage pre-upgrade validation
- `TestPostUpgradeTrustyAIService` - Post-upgrade validation and PVC-to-database migration
- `TestPreUpgradeDBTrustyAIService` - Database storage pre-upgrade validation
- `TestPostUpgradeDBTrustyAIService` - Database storage post-upgrade validation

**Covered Upgrade Paths:**

1. **PVC Storage Upgrade Path:**
   - Pre-upgrade: Deploy TrustyAI with PVC storage, send inferences, upload data, schedule drift metrics
   - Post-upgrade: Verify service functionality, scheduled metrics, perform PVC-to-database migration
   - Validated: Inference registration, data upload/retrieval, metric scheduling/deletion, database migration

2. **Database Storage Upgrade Path:**
   - Pre-upgrade: Deploy TrustyAI with database storage, send inferences, upload data, schedule metrics
   - Post-upgrade: Verify database credentials, service functionality, metric persistence
   - Validated: Database secret validation, inference data persistence, metric scheduling

**What's Validated:**

- TrustyAI service survives platform upgrade with both PVC and database storage
- Inference data persists across upgrade
- Scheduled metrics (drift detection) remain functional
- PVC-to-database migration works post-upgrade
- Database credentials and connections remain valid
- Metric deletion and rescheduling work after upgrade

**Test Dependencies:**

- Some database tests use `pytest.mark.dependency` to ensure proper execution order
- Dependencies: `db_pre_upgrade_inference` -> `db_pre_upgrade_data_upload` -> `db_pre_upgrade_metric_schedule`
- Post-upgrade: `db_migration` -> `db_post_upgrade_metric_delete`

**Example:**

```bash
# Pre-upgrade
uv run pytest -m pre_upgrade tests/ai_safety/trustyai_service/upgrade/

# Perform platform upgrade

# Post-upgrade
uv run pytest -m post_upgrade tests/ai_safety/trustyai_service/upgrade/
```

### Upgrade Test Patterns

All upgrade tests follow these common patterns:

1. **Namespace Reuse:** Pre- and post-upgrade test classes use the same namespace name to ensure resources persist across the upgrade
2. **Pytest Markers:** Tests are segregated by phase using `@pytest.mark.pre_upgrade` and `@pytest.mark.post_upgrade`
3. **State Validation:** Post-upgrade tests validate that pre-upgrade state (deployments, data, configurations) persists
4. **Functionality Checks:** Both phases verify core functionality to ensure no regression

### Adding New Upgrade Tests

When adding new upgrade test coverage:

1. Create test files in the appropriate `upgrade/` subdirectory
2. Define separate test classes for pre-upgrade and post-upgrade phases
3. Use the same namespace name in both classes via `pytest.param({"name": "test-<component>-upgrade"})`
4. Mark pre-upgrade tests with `@pytest.mark.pre_upgrade`
5. Mark post-upgrade tests with `@pytest.mark.post_upgrade`
6. Update this README section with the new coverage details under the appropriate component heading

## Additional Resources

- [TrustyAI Documentation](https://github.com/trustyai-explainability)
