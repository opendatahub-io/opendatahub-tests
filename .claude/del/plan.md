# Automation Plan — Model Serving Coverage Gaps
> Source: `Test Steps for Model Serving GapsCoverage.pdf`
> Codebase: `tests/model_serving/`
> Generated: 2026-05-07

---

## How to Read This Plan

Each group below maps one or more Jira IDs to a concrete automation task.
Every entry lists:
- **File** — exact path where the test/fixture goes
- **Test name(s)** — exact `def test_*` function(s) to write
- **Fixtures needed** — new conftest entries required
- **Markers** — from `pytest.ini`
- **Effort** — Low / Medium / High
- **Priority** — Critical / High / Medium (from PDF severity × gap count)

---

## Group 1 — Security (Critical, Low Effort) 🔴

**Jiras:** RHOAIENG-57815, RHOAIENG-57816, RHOAIENG-54382

These are pure assertion tests on existing running pods — no new infrastructure needed.

### 1a. TLS Private Key Not Mounted in Model Container (RHOAIENG-57815)

| Field | Value |
|-------|-------|
| **File** | `tests/model_serving/model_server/kserve/authentication/test_tls_private_key_not_mounted.py` |
| **Markers** | `@pytest.mark.tier1`, `@pytest.mark.rawdeployment`, `@pytest.mark.tls` |
| **Priority** | Critical |
| **Effort** | Low |

**Tests to write:**
```python
def test_tls_private_key_not_in_pod_env(self, predictor_pod):
    """Given a deployed raw ISVC with TLS enabled,
    When the predictor pod spec is inspected,
    Then no env var contains a TLS private key value."""

def test_tls_private_key_not_mounted_as_volume(self, predictor_pod):
    """Given a deployed raw ISVC with TLS enabled,
    When the pod volume mounts are inspected,
    Then no volume mount path contains a private key file (*.key, *.pem)."""
```

**Fixtures needed:** reuse existing `http_s3_ovms_raw_inference_service` + `get_pods_by_isvc_label`

---

### 1b. AutomountServiceAccountToken Disabled on Model Pods (RHOAIENG-57816)

| Field | Value |
|-------|-------|
| **File** | `tests/model_serving/model_server/kserve/authentication/test_tls_private_key_not_mounted.py` (same file) |
| **Markers** | `@pytest.mark.tier1`, `@pytest.mark.rawdeployment` |
| **Priority** | Critical |
| **Effort** | Low |

**Tests to write:**
```python
def test_automount_service_account_token_disabled(self, predictor_pod):
    """Given a deployed raw ISVC,
    When the predictor pod spec is inspected,
    Then automountServiceAccountToken is False on the pod spec."""
```

**Fixtures needed:** reuse existing predictor pod fixtures

---

### 1c. FIPS: No InsecureSkipVerify in DestinationRules / PodMonitor (RHOAIENG-54382)

| Field | Value |
|-------|-------|
| **File** | `tests/model_serving/model_server/kserve/authentication/test_fips_tls_destination_rules.py` |
| **Markers** | `@pytest.mark.tier1`, `@pytest.mark.tls`, `@pytest.mark.rawdeployment` |
| **Priority** | Critical |
| **Effort** | Medium |

**Tests to write:**
```python
def test_destination_rules_no_insecure_skip_verify(self, admin_client, model_namespace):
    """Given a deployed ISVC in a FIPS-enabled cluster,
    When all DestinationRules in the model namespace are inspected,
    Then none have tls.insecureSkipVerify set to True."""

def test_pod_monitor_no_insecure_skip_verify(self, admin_client, model_namespace):
    """Given a deployed ISVC,
    When all PodMonitor resources in the namespace are inspected,
    Then none have tlsConfig.insecureSkipVerify set to True."""
```

**Fixtures needed:** `admin_client`, existing model namespace fixture

---

## Group 2 — Authentication / Auth Proxy (High, Low Effort) 🟠

**Jira:** RHOAIENG-52129

### 2a. Raw Auth Toggle Does Not Cause Pod Rollout

| Field | Value |
|-------|-------|
| **File** | `tests/model_serving/model_server/kserve/authentication/test_kserve_token_authentication_raw.py` (extend existing class or add new class) |
| **Markers** | `@pytest.mark.tier1`, `@pytest.mark.rawdeployment` |
| **Priority** | High |
| **Effort** | Low |

**Tests to write:**
```python
def test_auth_toggle_disable_does_not_rollout_pod(self, unprivileged_client,
        http_s3_ovms_raw_inference_service, patched_remove_raw_authentication_isvc):
    """Given a running raw ISVC with auth enabled,
    When authentication is disabled via annotation patch,
    Then the predictor pod UID is unchanged (no new rollout occurred)."""

def test_auth_toggle_reenable_does_not_rollout_pod(self, unprivileged_client,
        http_s3_ovms_raw_inference_service, patched_remove_raw_authentication_isvc,
        patched_reenable_raw_authentication_isvc):
    """Given a running raw ISVC with auth disabled,
    When authentication is re-enabled via annotation patch,
    Then the predictor pod UID is unchanged (no new rollout occurred)."""
```

**Implementation note:** Capture pod UID via `get_pods_by_isvc_label()` before patch, assert same UID after patch completes. Pattern mirrors `verify_isvc_pods_not_restarted_against_baseline` in `upgrade/utils.py`.

**Fixtures needed:** `patched_reenable_raw_authentication_isvc` (new conftest fixture in `kserve/authentication/conftest.py`)

---

## Group 3 — ISVC Lifecycle (High, Low Effort) 🟠

**Jiras:** RHOAIENG-33695, RHOAIENG-58243

### 3a. ServingRuntime livenessProbe Update Reconciles to Deployment (RHOAIENG-33695)

| Field | Value |
|-------|-------|
| **File** | `tests/model_serving/model_server/kserve/inference_service_lifecycle/test_serving_runtime_probe_reconciliation.py` |
| **Markers** | `@pytest.mark.tier1`, `@pytest.mark.rawdeployment` |
| **Priority** | High |
| **Effort** | Low |

**Tests to write:**
```python
def test_liveness_probe_update_propagates_to_deployment(self, unprivileged_client,
        ovms_kserve_serving_runtime, ovms_raw_inference_service, patched_runtime_liveness_probe):
    """Given a running raw ISVC backed by a ServingRuntime,
    When the ServingRuntime livenessProbe is patched with new settings,
    Then the backing Deployment's livenessProbe matches the patched values."""

def test_readiness_probe_update_propagates_to_deployment(self, unprivileged_client,
        ovms_kserve_serving_runtime, ovms_raw_inference_service, patched_runtime_readiness_probe):
    """Given a running raw ISVC,
    When the ServingRuntime readinessProbe is patched,
    Then the backing Deployment reflects the updated probe."""
```

**Fixtures needed:**
- `patched_runtime_liveness_probe` — context-manager fixture that patches SR `livenessProbe.periodSeconds` and reverts
- `patched_runtime_readiness_probe` — same for `readinessProbe`

---

### 3b. LLMInferenceService Not Stuck Stopping When Config Deleted First (RHOAIENG-58243)

| Field | Value |
|-------|-------|
| **File** | `tests/model_serving/model_server/llmd/test_llmd_lifecycle.py` |
| **Markers** | `@pytest.mark.tier2`, `@pytest.mark.llmd_cpu` |
| **Priority** | High |
| **Effort** | Low |

**Tests to write:**
```python
def test_llmisvc_not_stuck_stopping_after_config_deletion(self, admin_client,
        llmisvc, llmisvc_config):
    """Given a running LLMInferenceService with an associated Config CR,
    When the Config CR is deleted before the LLMInferenceService,
    Then the LLMInferenceService transitions to Stopped state within SLA
    and does not remain stuck in Stopping indefinitely."""
```

**Fixtures needed:**
- `llmisvc_config` — fixture that creates and yields the associated Config CR
- SLA constant: `LLMISVC_STOP_TIMEOUT_SECONDS = 120`

---

## Group 4 — LLM-D (High / Medium, Medium Effort) 🟠

**Jiras:** RHOAIENG-57868, RHOAIENG-58347, RHOAIENG-56694

### 4a. LLM-D Scheduler Pod Respects ResourceQuota (RHOAIENG-57868)

| Field | Value |
|-------|-------|
| **File** | `tests/model_serving/model_server/llmd/test_llmd_scheduler_resource_quota.py` |
| **Markers** | `@pytest.mark.tier2`, `@pytest.mark.llmd_cpu` |
| **Priority** | High |
| **Effort** | Medium |

**Tests to write:**
```python
def test_scheduler_pod_has_resource_requests(self, admin_client, llmisvc):
    """Given a deployed LLMInferenceService with a scheduler,
    When the scheduler pod spec is inspected,
    Then all containers have non-empty resource requests (cpu, memory)."""

def test_scheduler_pod_respects_namespace_resource_quota(self, admin_client,
        llmisvc, namespace_resource_quota):
    """Given a namespace with a ResourceQuota set,
    When a LLMInferenceService is deployed,
    Then the scheduler pod is created successfully within quota limits."""
```

**Fixtures needed:**
- `namespace_resource_quota` — creates a `ResourceQuota` with CPU/memory limits, yields, deletes

---

### 4b. LLM-D 0.7 Controller Migration (RHOAIENG-58347)

| Field | Value |
|-------|-------|
| **File** | `tests/model_serving/model_server/llmd/test_llmd_migration.py` |
| **Markers** | `@pytest.mark.tier2`, `@pytest.mark.llmd_cpu` |
| **Priority** | High |
| **Effort** | High |

**Tests to write:**
```python
def test_llmd_07_crds_migrated_to_current_version(self, admin_client):
    """Given a cluster that has been upgraded from llm-d 0.7,
    When all LLMInferenceService CRDs are inspected,
    Then they report the current API version and no v0.7 deprecated fields remain."""

def test_llmd_07_metrics_compatibility(self, admin_client, llmisvc, prometheus):
    """Given a migrated LLMInferenceService,
    When vLLM engine metrics are scraped,
    Then both the 0.7-era metric names and current metric names are present
    or the deprecated ones have been correctly renamed."""
```

**Note:** This test requires migration environment setup. Mark `@pytest.mark.skip(reason="RHOAIENG-58347: requires pre-migration cluster state")` until infra is ready.

---

### 4c. LLM-D Router-Scheduler Tokenizer Image Valid (RHOAIENG-56694)

| Field | Value |
|-------|-------|
| **File** | `tests/model_serving/model_server/llmd/test_llmd_scheduler_image.py` |
| **Markers** | `@pytest.mark.tier2`, `@pytest.mark.llmd_cpu` |
| **Priority** | Medium |
| **Effort** | Low |

**Tests to write:**
```python
def test_router_scheduler_tokenizer_image_importable(self, admin_client, llmisvc):
    """Given a running LLMInferenceService with a router-scheduler,
    When the router-scheduler pod is inspected,
    Then the container image starts successfully (no CrashLoopBackOff)
    and vllm tokenizer can be imported (no ModuleNotFoundError in logs)."""
```

---

## Group 5 — Multi-Node Inference (High, Medium Effort) 🟠

**Jiras:** RHOAIENG-57975, RHOAIENG-58611

### 5a. Multi-Node ServingRuntime Update Recreates Head/Worker Services (RHOAIENG-57975)

| Field | Value |
|-------|-------|
| **File** | `tests/model_serving/model_server/kserve/multi_node/test_nvidia_multi_node.py` (add to existing class) |
| **Markers** | `@pytest.mark.tier2`, `@pytest.mark.multinode`, `@pytest.mark.gpu` |
| **Priority** | High |
| **Effort** | Medium |

**Tests to write:**
```python
def test_serving_runtime_update_recreates_head_service(self, admin_client,
        unprivileged_client, multi_node_inference_service, patched_multi_node_runtime):
    """Given a running multi-node ISVC,
    When the ServingRuntime is updated (e.g. image tag changed),
    Then the head Service is deleted and recreated with updated spec."""

def test_serving_runtime_update_recreates_worker_service(self, admin_client,
        unprivileged_client, multi_node_inference_service, patched_multi_node_runtime):
    """Given a running multi-node ISVC,
    When the ServingRuntime is updated,
    Then the worker Service is deleted and recreated with updated spec."""
```

**Fixtures needed:** `patched_multi_node_runtime` — patches SR image and reverts via context manager

---

### 5b. Multi-Node ISVC Deletion Does Not Hang with External Autoscaler (RHOAIENG-58611)

| Field | Value |
|-------|-------|
| **File** | `tests/model_serving/model_server/kserve/multi_node/test_nvidia_multi_node.py` (add to existing class) |
| **Markers** | `@pytest.mark.tier2`, `@pytest.mark.multinode`, `@pytest.mark.gpu` |
| **Priority** | High |
| **Effort** | Medium |

**Tests to write:**
```python
def test_isvc_deletion_completes_with_external_autoscaler(self, admin_client,
        multi_node_isvc_with_external_autoscaler):
    """Given a multi-node ISVC with autoscalerClass=external,
    When the ISVC is deleted,
    Then the ISVC is fully removed within the deletion SLA (no stuck Terminating state)."""
```

**Fixtures needed:**
- `multi_node_isvc_with_external_autoscaler` — ISVC with `autoscalerClass: external` annotation
- `ISVC_DELETION_TIMEOUT = 300` seconds

---

## Group 6 — InferenceGraph (Medium, Low Effort) 🟡

**Jira:** RHOAIENG-57805

### 6a. InferenceGraph No Dangling SAR ConfigMap Without OAuth

| Field | Value |
|-------|-------|
| **File** | `tests/model_serving/model_server/kserve/inference_graph/test_inference_graph_raw.py` (add to existing file) |
| **Markers** | `@pytest.mark.tier2`, `@pytest.mark.rawdeployment` |
| **Priority** | Medium |
| **Effort** | Low |

**Tests to write:**
```python
def test_no_dangling_sar_configmap_without_oauth(self, admin_client,
        model_namespace, dog_breed_inference_graph_no_oauth):
    """Given an InferenceGraph deployed without OAuth proxy enabled,
    When the model namespace ConfigMaps are listed,
    Then no SAR (SubjectAccessReview) ConfigMap volume remains as a dangling orphan."""
```

**Fixtures needed:** `dog_breed_inference_graph_no_oauth` — variant of existing `dog_breed_inference_graph` fixture with OAuth proxy disabled

---

## Group 7 — Storage (Medium, Low Effort) 🟡

**Jira:** RHOAIENG-57629

### 7a. ClusterStorageContainer Supports Multi-Model Download

| Field | Value |
|-------|-------|
| **File** | `tests/model_serving/model_server/kserve/storage/test_cluster_storage_container.py` |
| **Markers** | `@pytest.mark.tier2`, `@pytest.mark.rawdeployment` |
| **Priority** | Medium |
| **Effort** | Low |

**Tests to write:**
```python
def test_default_cluster_storage_container_has_multi_model_download(self, admin_client):
    """Given the default ClusterStorageContainer CR exists,
    When its spec is inspected,
    Then supportsMultiModelDownload is True."""

def test_isvc_with_multi_model_download_deploys_successfully(self, admin_client,
        multi_model_isvc):
    """Given a ClusterStorageContainer with supportsMultiModelDownload=True,
    When an ISVC using multi-model download is deployed,
    Then it reaches Ready state."""
```

---

## Group 8 — Partial Coverage Extensions (extend existing tests)

These are **additions to existing test files**, not new files.

### 8a. PVC Read-Only Annotation Lifecycle Toggle (RHOAIENG-8288)

- **File:** `tests/model_serving/model_server/kserve/storage/pvc/test_kserve_pvc_write_access.py`
- **Add test:**
```python
def test_isvc_readonly_annotation_toggle_lifecycle(self, unprivileged_client,
        patched_read_only_isvc_false, patched_read_only_isvc_true):
    """Verify write access reflects each toggle: false → write allowed, true → write denied."""
```

### 8b. Headless Service clusterIP=None for Raw Mode (RHOAIENG-5077)

- **File:** `tests/model_serving/model_server/kserve/platform/dsc_deployment_mode/test_kserve_dsc_default_deployment_mode.py`
- **Add test:**
```python
def test_raw_deployment_headless_service_cluster_ip_none(self, admin_client,
        ovms_inference_service):
    """Verify the backing Service for raw deployment has spec.clusterIP=None (headless)."""
```

### 8c. HF_TOKEN Not Exposed in Pod Spec (RHOAIENG-33721)

- **File:** `tests/model_serving/model_server/kserve/storage/oci/test_oci_image.py`
- **Add test:**
```python
def test_hf_token_not_in_pod_env(self, model_car_inference_service):
    """Verify HF_TOKEN is absent from all pod container env vars and volume mount paths."""
```

### 8d. gRPC Raw Route Tests Unskip (RHOAIENG-18360)

- **File:** `tests/model_serving/model_server/kserve/ingress/test_route_visibility.py`
- **Action:** Remove `@pytest.mark.skip(reason="skipping grpc raw for tgis-caikit")` from both
  `TestGrpcRawDeployment` classes (lines 211 and 275) once tgis-caikit gRPC is fixed.
- **Add `@pytest.mark.jira("RHOAIENG-18360")`** as tracking marker until unskipped.

### 8e. KEDA Scale-Down Assertion (RHOAIENG-32306)

- **File:** `tests/model_serving/model_server/kserve/autoscaling/keda/test_isvc_keda_scaling_gpu.py`
- **Add test:**
```python
def test_vllm_keda_scaling_verify_scale_down(self, ...):
    """After load stops, verify replica count returns to minReplicaCount within cooldown window."""
```

### 8f. LLMD 503 First Request After Ready (RHOAIENG-55154)

- **File:** `tests/model_serving/model_server/llmd/test_llmd_connection_cpu.py`
- **Add assertion** inside existing `test_llmd_connection_cpu`:
  Record timestamp when ISVC `Ready=True`, fire inference immediately, assert no 503.
  Note: warm-up workaround already tracked in `llmd/utils.py` via `is_jira_issue_open("RHOAIENG-55154")`.

---

## Implementation Order (Recommended)

| Sprint | Group | Jiras | Rationale |
|--------|-------|-------|-----------|
| 1 | Group 1 (Security) | 57815, 57816, 54382 | Critical severity, low effort, pure assertions |
| 1 | Group 2 (Auth Toggle) | 52129 | High severity, low effort, extends existing fixture |
| 1 | Group 3 (Lifecycle) | 33695, 58243 | High severity, low effort, clear test pattern |
| 2 | Group 4 (LLM-D) | 57868, 56694 | High severity, medium effort, new fixtures |
| 2 | Group 6 (InferenceGraph) | 57805 | Medium severity, low effort |
| 2 | Group 7 (Storage) | 57629 | Medium severity, low effort |
| 3 | Group 5 (Multi-Node) | 57975, 58611 | High severity, GPU required, medium effort |
| 4 | Group 4b (Migration) | 58347 | High severity, needs pre-migration cluster state |
| Ongoing | Group 8 (Partial) | 8288, 5077, 33721, 18360, 32306, 55154 | Extend existing tests incrementally |

---

## Structural Gaps (Empty Directories — Need Tests)

The following directories exist in the repo but contain **zero test files**. They represent
planned coverage areas that have never been implemented:

| Empty Directory | Intended Coverage |
|-----------------|-------------------|
| `kserve/storage/minio/` | Minio storage backend tests |
| `kserve/inference_service_configuration/` | ISVC config options tests |
| `kserve/component_health/` | KServe component health checks |
| `kserve/metrics/` | KServe metrics tests (separate from observability/) |
| `kserve/model_car/` | Model-car specific kserve tests |
| `kserve/private_endpoint/` | Private endpoint kserve tests |
| `kserve/raw_deployment/` | Raw deployment specific tests |
| `kserve/routes/` | Route management tests |
| `kserve/stop_resume/` | Stop/resume lifecycle tests |
| `model_server/authentication/` | Top-level auth tests (not kserve-specific) |
| `model_server/components/model_mesh_kserve_co_exist/` | ModelMesh + KServe co-existence |
| `model_server/components/raw_deployment_serverless_co_exist/` | Raw + Serverless co-existence |
| `model_server/metrics/` | Model server metrics |
| `model_server/model_car/` | Model-car server tests |
| `model_server/model_mesh/` | ModelMesh tests |
| `model_server/multi_node/` | Multi-node top-level tests |
| `model_server/ovms/{kserve,model_mesh}/` | OVMS integration tests |
| `model_server/private_endpoint/` | Private endpoint server tests |
| `model_server/runtime_configuration/` | Runtime configuration tests |
| `model_server/storage/pvc/` | Top-level PVC storage tests |
| `model_runtime/mlserver/basic_model_deployment/` | MLServer basic model tests |
| `llmd/kueue/` | LLMD + Kueue integration |

---

## Conventions Reminder

- All test methods **must have a Given-When-Then docstring**
- Fixtures must be **nouns** (`predictor_pod`, not `get_predictor_pod`)
- Use `openshift-python-wrapper` for all K8s API calls
- Use **context managers** for resource lifecycle in fixtures
- Jira references go in **module-level docstring or inline comments**, not test names
- Run `pre-commit run --all-files` before raising a PR
