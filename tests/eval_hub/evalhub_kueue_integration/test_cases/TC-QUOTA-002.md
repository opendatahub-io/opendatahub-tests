---
test_case_id: TC-QUOTA-002
source_key: RHOAIENG-59092
priority: P0
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-QUOTA-002: Job exceeding ClusterQueue quota is queued

**Objective**: Verify that a single job requesting more resources than the ClusterQueue quota remains queued and is not admitted.

**Preconditions**:
- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=500m, memory=1Gi
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`
- No other jobs consuming quota

**Test Steps**:
1. Create ClusterQueue `eval-cq` with cpu=500m, memory=1Gi
2. Create LocalQueue `eval-queue` in test namespace
3. Submit an evaluation job requesting cpu=1, memory=2Gi (exceeds quota)
4. Wait 30 seconds
5. Verify the Workload resource shows `QuotaReserved=False` with a message indicating insufficient quota
6. Verify the Kubernetes Job remains suspended
7. Verify EvalHub API reports the job as `pending`
8. Teardown: Delete the job, LocalQueue, and ClusterQueue

**Expected Results**:
- Workload status shows `QuotaReserved=False` with message containing "insufficient unused quota"
- Kubernetes Job has `spec.suspend: true`
- EvalHub API reports `status.state: "pending"`
- The job is not scheduled on any node

**Validation**:
- `oc get workload -n ${NAMESPACE} -o jsonpath='{.items[0].status.conditions[?(@.type=="QuotaReserved")].message}'` contains "insufficient"

**Notes**: To be filled later in the process.
