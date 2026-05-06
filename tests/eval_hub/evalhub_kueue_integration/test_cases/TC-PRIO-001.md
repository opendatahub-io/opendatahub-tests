---
test_case_id: TC-PRIO-001
source_key: RHOAIENG-59092
priority: P1
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-PRIO-001: Default priority 0 assigned when not specified

**Objective**: Verify that a job submitted without an explicit priority is assigned the default priority value of 0 in the Kueue Workload resource.

**Preconditions**:
- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=1, memory=4Gi
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`

**Test Steps**:
1. Create ClusterQueue and LocalQueue as per preconditions
2. Submit an evaluation job without specifying a priority field
3. Find the Kueue Workload resource created for the job
4. Verify the Workload's `spec.priority` field is 0
5. Teardown: Delete the job, LocalQueue, and ClusterQueue

**Expected Results**:
- Workload resource has `spec.priority: 0`
- The job is admitted following standard FIFO ordering

**Validation**:
- `oc get workload -n ${NAMESPACE} -o jsonpath='{.items[0].spec.priority}'` returns `0`

**Notes**: To be filled later in the process.
