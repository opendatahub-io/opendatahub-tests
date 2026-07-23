# Notebooks Server Controller Upgrade Tests

Controller-level upgrade survival and post-upgrade functionality tests for ODH/RHOAI workbenches.

These tests validate that the **odh-notebook-controller** (and its mutating webhook) correctly preserve running and stopped notebooks across a platform upgrade, keep auth/routing/CA resources intact, and can still create new notebooks afterward.

They are distinct from [`notebook_images/upgrade/`](../../../notebook_images/upgrade/), which focuses on N-1 **image** survival across IDE types. This suite focuses on **controller semantics** in a shared namespace with a small, fixed set of Notebook CRs.

## Architecture Overview

```text
                    --pre-upgrade                          platform upgrade                 --post-upgrade
                           |                                      |                               |
                           v                                      v                               v
  +------------------------+------------------------+    +------------------+    +------------------------+------------------------+
  | Create resources in ns upgrade-workbenches      |    | Operator /       |    | Attach to existing CRs (no recreate)            |
  |   - running notebook (upgrade-workbenches)      | -> | platform upgrade | -> | Compare against baseline ConfigMap              |
  |   - stopped notebook (upgrade-wb-stopped)       |    |                  |    | Assert survival + create upgrade-wb-new         |
  | Capture baseline CM upgrade-workbenches-baseline|    +------------------+    +------------------------+------------------------+
  +-------------------------------------------------+
```

### Two-phase execution

| Phase        | CLI flag         | What fixtures do                                                                                  | What tests assert                                                   |
| ------------ | ---------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Pre-upgrade  | `--pre-upgrade`  | Create namespace, PVCs, Notebook CRs; wait for Ready; stop one notebook; write baseline ConfigMap | Resources exist and look correct *before* upgrade                   |
| Post-upgrade | `--post-upgrade` | Wrap existing resources by name (do not recreate); load baseline; optionally tear down at end     | Resources survived; generations/specs unchanged; new notebook works |

Session-scoped fixtures in `conftest.py` own almost all lifecycle logic. Tests are thin Given-When-Then assertions over those fixtures.

### Shared resources (namespace `upgrade-workbenches`)

| Resource               | Name                           | Role                                                           |
| ---------------------- | ------------------------------ | -------------------------------------------------------------- |
| Namespace              | `upgrade-workbenches`          | Dashboard-labeled project for all controller upgrade scenarios |
| Running notebook + PVC | `upgrade-workbenches`          | Primary survival subject (pod must not restart)                |
| Stopped notebook + PVC | `upgrade-wb-stopped`           | Stopped-before-upgrade subject (`kubeflow-resource-stopped`)   |
| New notebook + PVC     | `upgrade-wb-new`               | Created only in post-upgrade to prove controller still works   |
| Baseline ConfigMap     | `upgrade-workbenches-baseline` | Bridge between pre and post phases                             |

All Notebook CRs are built via `build_notebook_dict()` with `notebooks.opendatahub.io/inject-auth=true`, so auth sidecar and Gateway API routing are always in scope.

### Baseline ConfigMap contract

`capture_notebook_baseline` (pre-upgrade, no-op post-upgrade) writes JSON under key `baseline`:

| Key                                  | Used by post-upgrade to verify                 |
| ------------------------------------ | ---------------------------------------------- |
| `ntb_creation_timestamp`             | Running pod was not recreated                  |
| `notebook_generation`                | Notebook CR was not mutated                    |
| `statefulset_generation`             | StatefulSet was not mutated                    |
| `service_ports` / `service_selector` | Service spec unchanged                         |
| `httproute_generation`               | HTTPRoute was not mutated                      |
| `stopped_annotation_value`           | Stop timestamp unchanged                       |
| `ca_bundle_resource_version`         | Workbench CA bundle change tracking            |
| `odh_ca_bundle_resource_version`     | Source `odh-trusted-ca-bundle` change tracking |

`upgrade_notebook_baseline` loads that ConfigMap in post-upgrade (empty dict in pre-upgrade).

### Fixture dependency shape

```text
upgrade_notebook_namespace
├── upgrade_notebook_pvc ──► upgrade_notebook ──► upgrade_notebook_pod
│                                              ├── upgrade_notebook_statefulset / service
│                                              ├── upgrade_notebook_httproute
│                                              └── auth_proxy_* / auth_delegator_crb
├── stopped_notebook_pvc ──► stopped_notebook ──► stopped_notebook_pre_upgrade_shutdown
│                                              ├── stopped_notebook_statefulset
│                                              └── stopped_auth_proxy_* / stopped_auth_delegator_crb
├── workbench_trusted_ca_bundle
├── capture_notebook_baseline  (depends on running + stopped shutdown + CA bundles)
└── (post only) new_notebook_pvc ──► new_notebook ──► new_notebook_pod / sts / svc / httproute / auth_*
```

Important fixture behaviours:

- **Pre vs post branch**: create-with-context-manager when not `--post-upgrade`; attach-by-name when `--post-upgrade`.
- **`stopped_notebook_pre_upgrade_shutdown`**: Ready → annotate stop → wait pod deleted → assert STS replicas=0. No-op post-upgrade.
- **`capture_notebook_baseline`**: Pulls stopped-notebook shutdown via fixture dependency so baseline always includes a stopped annotation value.
- Teardown of long-lived upgrade resources is gated by `teardown_resources` and typically happens on the post-upgrade run.

## What Is Accounted For

Coverage is grouped by concern. Each bullet maps to one or more test methods.

### 1. Running notebook survival (`test_upgrade.py`)

| Concern                                                 | Pre-upgrade                                      | Post-upgrade                                         |
| ------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------- |
| Pod Ready                                               | `test_notebook_running_before_upgrade`           | —                                                    |
| Pod not restarted (creationTimestamp)                   | baseline capture                                 | `test_notebook_not_restarted_after_upgrade`          |
| Notebook CR not modified (generation)                   | baseline                                         | `test_notebook_cr_not_modified_after_upgrade`        |
| StatefulSet not modified (generation)                   | baseline                                         | `test_statefulset_not_modified_after_upgrade`        |
| StatefulSet healthy (readyReplicas, no pending rollout) | —                                                | `test_statefulset_healthy_after_upgrade`             |
| Service ports/selector unchanged                        | baseline                                         | `test_service_not_modified_after_upgrade`            |
| Workbench CA bundle exists (`ca-bundle.crt`)            | `test_ca_bundle_configmap_exists_before_upgrade` | `test_ca_bundle_configmap_exists_after_upgrade`      |
| CA propagation consistency (odh ↔ workbench RVs)        | baseline RVs                                     | `test_ca_bundle_configmap_consistency_after_upgrade` |

CA consistency rule: if `odh-trusted-ca-bundle` changes, `workbench-trusted-ca-bundle` must change too; if the source does not change, the workbench bundle must not change unexpectedly.

### 2. Auth / kube-rbac-proxy (`test_upgrade_auth.py`)

Covered for both the **running** and **stopped** notebooks (stopped: Service/ConfigMap/CRB only — no pod sidecar check while stopped).

| Concern                                                     | Pre                     | Post                    |
| ----------------------------------------------------------- | ----------------------- | ----------------------- |
| Sidecar container `kube-rbac-proxy` on running pod          | yes                     | yes                     |
| Proxy Service exists with port `8443`                       | yes (running + stopped) | yes (running + stopped) |
| Proxy ConfigMap exists                                      | yes (running + stopped) | yes (running + stopped) |
| Auth-delegator ClusterRoleBinding → `system:auth-delegator` | yes (running + stopped) | yes (running + stopped) |

### 3. Gateway API routing (`test_upgrade_routing.py`)

| Concern                                                                       | Pre      | Post |
| ----------------------------------------------------------------------------- | -------- | ---- |
| HTTPRoute exists in applications namespace (`nb-<ns>-<name>`)                 | yes      | yes  |
| ParentRef = `openshift-ingress/data-science-gateway`                          | yes      | yes  |
| BackendRef → `<notebook>-kube-rbac-proxy:8443` + path `/notebook/<ns>/<name>` | yes      | yes  |
| HTTPRoute generation unchanged                                                | baseline | yes  |
| Exactly one HTTPRoute for the notebook (no duplicates)                        | —        | yes  |
| ReferenceGrant `notebook-httproute-access` in notebook namespace              | yes      | yes  |

### 4. Stopped notebook semantics (`test_upgrade_stopped.py`)

| Concern                                        | Pre            | Post |
| ---------------------------------------------- | -------------- | ---- |
| StatefulSet replicas = 0                       | yes            | yes  |
| Pod absent                                     | yes            | yes  |
| `kubeflow-resource-stopped` annotation present | set by fixture | yes  |
| Annotation timestamp value unchanged           | baseline       | yes  |

### 5. Post-upgrade creation + webhook (`test_upgrade_creation.py`)

Post-upgrade only. Proves the upgraded controller and webhook can still reconcile a fresh Notebook CR (`upgrade-wb-new`).

| Concern                                                                                    | Test                                                                             |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| New pod reaches Ready                                                                      | `test_new_notebook_pod_ready`                                                    |
| StatefulSet exists with 1 replica                                                          | `test_new_notebook_statefulset_exists`                                           |
| Service exists                                                                             | `test_new_notebook_service_exists`                                               |
| HTTPRoute + gateway parent + backend refs                                                  | `test_new_notebook_httproute_exists`                                             |
| Auth sidecar injected                                                                      | `test_new_notebook_has_auth_sidecar`                                             |
| Auth Service / ConfigMap / CRB reconciled                                                  | `test_new_notebook_auth_proxy_*` / `test_new_notebook_auth_delegator_crb_exists` |
| Reconciliation lock cleared (`kubeflow-resource-stopped` ≠ `odh-notebook-controller-lock`) | `test_new_notebook_reconciliation_lock_cleared`                                  |
| MutatingWebhookConfiguration exists                                                        | `test_mutating_webhook_exists`                                                   |
| Webhook `failurePolicy=Fail`                                                               | `test_mutating_webhook_failure_policy`                                           |
| Webhook targets `kubeflow.org/notebooks` CREATE + UPDATE                                   | `test_mutating_webhook_targets_notebooks`                                        |

## File Map

```text
upgrade/
|-- README.md                 # This document
|-- conftest.py               # Session fixtures, baseline capture/load, stop lifecycle
|-- test_upgrade.py           # Running notebook survival + CA bundles
|-- test_upgrade_auth.py      # kube-rbac-proxy for running and stopped notebooks
|-- test_upgrade_routing.py   # HTTPRoute + ReferenceGrant
|-- test_upgrade_stopped.py   # Stopped notebook stays stopped
+-- test_upgrade_creation.py  # New notebook + mutating webhook (post only)
```

## Running

```bash
# Pre-upgrade (create resources + baseline)
uv run pytest --pre-upgrade tests/workbenches/notebooks_server/controller/upgrade/

# Platform upgrade happens outside this suite

# Post-upgrade (survival + new creation)
uv run pytest --post-upgrade tests/workbenches/notebooks_server/controller/upgrade/

# Narrow to one concern
uv run pytest --post-upgrade tests/workbenches/notebooks_server/controller/upgrade/test_upgrade_routing.py
```

## Out of Scope (covered elsewhere)

| Concern                                                                            | Where                                                                                                                  |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| N-1 workbench image survival across IDEs (JupyterLab, Code Server, RStudio, Elyra) | [`../../../notebook_images/upgrade/`](../../../notebook_images/upgrade/)                                               |
| Dashboard-driven image bump N-1 → N                                                | [`../../../notebook_images/upgrade/test_bump_jupyterlab.py`](../../../notebook_images/upgrade/test_bump_jupyterlab.py) |
| ImageStream import health / digest tags                                            | [`../../operator/test_imagestream_health.py`](../../operator/test_imagestream_health.py)                               |
| Non-upgrade spawn and auth resource customization                                  | [`../test_spawning.py`](../test_spawning.py)                                                                           |
| Custom image package verification                                                  | [`../test_custom_images.py`](../test_custom_images.py)                                                                 |

## Notes

- Namespace `upgrade-workbenches` must not collide with `upgrade-notebook-images` used by the image upgrade suite.
- Pre-upgrade tests that need a complete baseline should depend on (or use) `capture_notebook_baseline` via `@pytest.mark.usefixtures("capture_notebook_baseline")`.
- Most of the behavioural complexity lives in `conftest.py`; review fixtures before changing assertions.
