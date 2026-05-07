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
3. Immediately query the Workload resource before admission
4. Verify `QuotaReserved` and `Admitted` conditions are present
5. Wait for the job to be admitted
6. Query the Workload resource again
7. Verify conditions transition: `QuotaReserved=True`, `Admitted=True`
8. Verify the Workload has an `ownerReference` pointing to the Kubernetes Job
9. Teardown: Delete the job, LocalQueue, and ClusterQueue

**Expected Results**:

- Before admission: Workload has `QuotaReserved=False`, `Admitted=False`
- After admission: Workload has `QuotaReserved=True`, `Admitted=True`
- Workload `metadata.ownerReferences` contains the Kubernetes Job name

**Validation**:

- `oc get workload -n ${NAMESPACE} -o yaml` shows complete status.conditions array
- `oc get workload -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.ownerReferences[0].name}'` returns the Kubernetes Job name

**Notes**: To be filled later in the process.
