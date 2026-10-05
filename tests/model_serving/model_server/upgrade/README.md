# Model Serving Upgrade Tests

Pre/post upgrade tests for InferenceService (ISVC) and LLMInferenceService (LLMISVC) workloads. These tests verify that deployed models survive RHOAI operator upgrades without pod restarts, redeployments, or data loss.

## How it works

Tests are split into two phases using pytest markers:

- `@pytest.mark.pre_upgrade` — deploy resources, run inference, capture baselines to ConfigMaps
- `@pytest.mark.post_upgrade` — verify resources survived, compare against baselines, run inference again

The CI pipeline runs pre-upgrade tests on the source branch, performs the operator upgrade, then runs post-upgrade tests on the target branch. Baselines are persisted as ConfigMaps in each test namespace so they survive the upgrade.

## Test scenarios

### InferenceService (ISVC)

| File                               | Scenario               | Namespace                    |
| ---------------------------------- | ---------------------- | ---------------------------- |
| `test_upgrade.py`                  | RawDeployment OVMS     | `upgrade-model-server`       |
| `test_upgrade_auth.py`             | Auth-enabled RawDeploy | `upgrade-auth-model-server`  |
| `test_upgrade_model_car.py`        | ModelCar (OCI storage) | `upgrade-model-car`          |
| `test_upgrade_metrics.py`          | Metrics (Prometheus)   | `upgrade-metrics`            |
| `test_upgrade_private_endpoint.py` | Private endpoint       | `upgrade-pvt-ep`             |
| `test_upgrade_kserve_kueue_raw.py` | RawDeployment + Kueue  | `upgrade-kserve-kueue-raw`   |

**Post-upgrade ISVC checks**: exists, not modified, runtime not modified, pods not restarted, inference works. `test_upgrade.py` also creates a new ISVC post-upgrade to verify fresh deployments. `test_upgrade_auth.py` verifies auth annotation preserved, inference with pre-upgrade token, and unauthorized rejection. `test_upgrade_metrics.py` verifies historical metric data retained and new requests captured. `test_upgrade_private_endpoint.py` verifies internal URL preserved and no external route. `test_upgrade_kserve_kueue_raw.py` verifies Kueue LocalQueue survival, running/gated pod stats, totalCopies, generation, restart counts (allowing newly admitted pods), and post-upgrade inference via the external route.

### LLMInferenceService (LLMISVC)

| File                              | Scenario                                  | Namespace                     |
| --------------------------------- | ----------------------------------------- | ----------------------------- |
| `test_upgrade_llmd.py`            | LLMISVC (no auth)                         | `upgrade-llmd`                |
| `test_upgrade_llmd_auth_kueue.py` | LLMISVC + auth + Kueue (cross-component)  | `upgrade-llmd-auth-and-kueue` |

**Post-upgrade LLMISVC checks**: Ready condition, observedGeneration, URL, replicas, model URI, container images (by digest), restart counts, LLMInferenceServiceConfig survival and ref consistency, InferencePool ownership, HTTPRoute existence, controller health, inference (single + repeated). The auth+Kueue scenario additionally verifies auth inference with pre-upgrade token, unauthorized rejection, Kueue LocalQueue existence, conditions, and gating stats.

## Supported upgrade paths

These tests run on every upgrade path defined in [`rhoai-releases.yaml`](https://gitlab.cee.redhat.com/ods/jenkins/-/blob/master/resources/configs/rhoai-releases.yaml) except for the 2.25.x → 3.x cross-major path, which uses a dedicated pipeline with its own tests.

| From   | To       | Covered             |
| ------ | -------- | ------------------- |
| 2.25.z | 2.25.z+1 | Yes (z-stream)      |
| 3.3.z  | 3.3.z+1  | Yes (z-stream)      |
| 3.3.z  | 3.4.z    | Yes (minor)         |
| 3.4.z  | 3.4.z+1  | Yes (z-stream)      |
| 3.4.z  | 3.5.z    | Yes (minor)         |
| 2.25.z | 3.x      | Separate pipeline   |

## Running locally

See [docs/UPGRADE.md](../../../../docs/UPGRADE.md) for general upgrade test instructions.

```bash
cd ~/Code/opendatahub-tests

# Pre-upgrade (run on source version cluster)
uv run pytest --pre-upgrade tests/model_serving/model_server/upgrade/

# Post-upgrade (run after operator upgrade)
uv run pytest --post-upgrade tests/model_serving/model_server/upgrade/

# Collect only (list tests without running)
uv run pytest --collect-only -q tests/model_serving/model_server/upgrade/

# Run specific scenario
uv run pytest --pre-upgrade tests/model_serving/model_server/upgrade/test_upgrade_llmd.py
uv run pytest --pre-upgrade tests/model_serving/model_server/upgrade/test_upgrade_kserve_kueue_raw.py
```

## Key implementation details

- **Baseline persistence**: Pre-upgrade state (pod names, restart counts, generation, images, URLs) is captured and stored in ConfigMaps. Most ISVC baselines are saved in the shared `upgrade-model-server` namespace; KServe+Kueue and LLMISVC baselines are saved in each workload's own namespace. The namespaces persist across the upgrade, so post-upgrade tests load the same ConfigMaps and compare against current state.
- **Auth token persistence**: For auth scenarios, pre-upgrade tokens are saved to Secrets so they can be reused post-upgrade to verify token survival.
- **Cross-branch compatibility**: Resource names, namespaces, and fixture names must be identical between branches for pre/post pairs to match. Any rename breaks the upgrade path.
- **Dependencies**: Post-upgrade tests use `@pytest.mark.dependency` to skip downstream tests if the ISVC/LLMISVC no longer exists.

## Known limitations

- **2.25.x → 3.x cross-major path**: not covered by these tests. This path uses a dedicated pipeline with its own test suite.
- **CPU-only**: all upgrade tests (including LLMISVC) are implemented to run on CPU-only clusters. There is no upgrade test coverage for GPU-accelerated model deployments.

## Maintenance

Owned by the **Serving Orchestration** team. When updating these tests:

- Keep resource names and namespaces identical across branches to preserve pre/post pairing.
- When adding new post-upgrade assertions, backport to the oldest supported branch that will be a pre-upgrade source.
- After adding scenarios, update the scenario tables in this README.
