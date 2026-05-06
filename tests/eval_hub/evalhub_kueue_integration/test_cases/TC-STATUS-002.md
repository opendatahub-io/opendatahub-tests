---
test_case_id: TC-STATUS-002
source_key: RHOAIENG-59092
priority: P1
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-STATUS-002: LocalQueue status reflects pending and admitted job counts

**Objective**: Verify that the LocalQueue resource status accurately reflects the count of pending and admitted workloads.

**Preconditions**:

- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=500m, memory=1Gi
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`

**Test Steps**:

1. Create ClusterQueue with small quota (cpu=500m, memory=1Gi) and LocalQueue
2. Query LocalQueue status — verify pending=0, admitted=0
3. Submit job-a (cpu=500m, memory=1Gi) — wait for admission
4. Query LocalQueue status — verify pending=0, admitted=1
5. Submit job-b (cpu=500m, memory=1Gi) — should remain pending
6. Query LocalQueue status — verify pending=1, admitted=1
7. Teardown: Delete both jobs, LocalQueue, and ClusterQueue

**Expected Results**:

- After job-a admitted: `oc get localqueue eval-queue -n ${NAMESPACE}` shows ADMITTED=1, PENDING=0
- After job-b submitted: `oc get localqueue eval-queue -n ${NAMESPACE}` shows ADMITTED=1, PENDING=1

**Validation**:

- `oc get localqueue eval-queue -n ${NAMESPACE} -o jsonpath='{.status.pendingWorkloads}'` returns expected count
- `oc get localqueue eval-queue -n ${NAMESPACE} -o jsonpath='{.status.admittedWorkloads}'` returns expected count

**Notes**: To be filled later in the process.
