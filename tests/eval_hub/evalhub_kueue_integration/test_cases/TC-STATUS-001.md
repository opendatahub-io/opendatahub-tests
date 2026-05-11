---
test_case_id: TC-STATUS-001
source_key: RHOAIENG-59092
priority: P1
status: Draft
automation_status: Automated
last_updated: '2026-05-04'
---
# TC-STATUS-001: Workload resource reflects detailed queue status conditions

**Objective**: Verify that the Kubernetes Workload resource created for a Kueue-managed job exposes detailed status conditions (QuotaReserved, Admitted) that reflect the job's queue position.

**Preconditions**:

- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=1, memory=4Gi
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`

**Test Steps**:

1. Create ClusterQueue and LocalQueue as per preconditions
2. Submit an evaluation job with queue specification
3. **TC-STATUS-001 — Pre-admission baseline (required):** In a polling/query loop on the Workload during pre-admission, capture and record **at least one** sample where condition **`Admitted`** is `False` or **missing** *and* condition **`QuotaReserved`** is `False` or **missing** (the same sample must satisfy both). If bounded polling completes without such a sample, **fail TC-STATUS-001** (no pre-admission baseline was observed).
4. **TC-STATUS-001 — Post-admission (bounded; CWE-400):** Only after step 3 succeeds, run a **bounded** polling loop (not an unbounded “wait for admission”): repeat **query Workload → evaluate `status.conditions`** at interval **`TC_STATUS_001_ADMISSION_POLL_INTERVAL_SECONDS`** until **`Admitted=True`** and **`QuotaReserved=True`**, or until elapsed time exceeds **`TC_STATUS_001_ADMISSION_TIMEOUT_SECONDS`**. If the timeout is reached, **fail explicitly**: abort further polling, log **the last observed Workload resource** (full YAML or at minimum `status.conditions` / **`Admitted`** / **`QuotaReserved`**) for diagnostics, and **do not** pass the test. If admission succeeds within the timeout, **query the Workload again** and assert conditions include **`QuotaReserved=True`** and **`Admitted=True`** (same observation).
5. Verify the Workload has an `ownerReference` pointing to the Kubernetes Job
6. Teardown: Delete the job, LocalQueue, and ClusterQueue

**Harness parameters (override via environment for automation):**

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `TC_STATUS_001_ADMISSION_TIMEOUT_SECONDS` | `600` | Max wall-clock time for post-admission polling (step 4) |
| `TC_STATUS_001_ADMISSION_POLL_INTERVAL_SECONDS` | `5` | Sleep between Workload queries in step 4 |

**Expected Results**:

- **TC-STATUS-001** records at least one pre-admission observation where **`Admitted`** is false/missing and **`QuotaReserved`** is false/missing; the test fails if this never occurs
- After that baseline, Workload conditions transition to **`Admitted=True`** and **`QuotaReserved=True`** within **`TC_STATUS_001_ADMISSION_TIMEOUT_SECONDS`** (see harness table); otherwise the run fails and logs the last Workload snapshot
- Workload `metadata.ownerReferences` contains the Kubernetes Job name

**Validation**:

- `oc get workload "${WORKLOAD_NAME}" -n "${NAMESPACE}" -o yaml` shows **`Admitted`** and **`QuotaReserved`** in `status.conditions` for TC-STATUS-001; automation must persist at least one snapshot with both not `True` before asserting both `True`
- `oc get workload "${WORKLOAD_NAME}" -n "${NAMESPACE}" -o jsonpath='{.metadata.ownerReferences[?(@.kind=="Job")].name}'` returns the Kubernetes Job ownerReference name

**Notes**: To be filled later in the process.
