---
test_case_id: TC-QUEUE-001
source_key: RHOAIENG-59092
priority: P0
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-QUEUE-001: Job admitted when LocalQueue has available quota

**Objective**: Verify that a job submitted to a LocalQueue with available ClusterQueue quota is admitted and starts running.

**Preconditions**:

- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=1, memory=4Gi
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`
- No other jobs consuming quota in `eval-cq`

**Test Steps**:

1. Create ClusterQueue `eval-cq` with cpu=1, memory=4Gi quota
2. Create LocalQueue `eval-queue` in the test namespace pointing to `eval-cq`
3. Submit an evaluation job with `queue.kind: "kueue"` and `queue.name: "eval-queue"`
4. Wait for the job to be admitted (check Workload resource `status.conditions` for `Admitted=True`)
5. Verify the Kubernetes Job is unsuspended (`spec.suspend: false`)
6. Verify the EvalHub API reports the job as `running` (poll GET /api/v1/evaluations/jobs/{id})
7. Teardown: Delete the evaluation job, LocalQueue, and ClusterQueue

**Expected Results**:

- Workload resource has `Admitted=True` and `QuotaReserved=True` conditions
- Kubernetes Job has `spec.suspend: false`
- EvalHub API reports `status.state: "running"` within the admission timeout

**Validation**:

- `oc get workload -n ${NAMESPACE} -o jsonpath='{.items[0].status.conditions[?(@.type=="Admitted")].status}'` returns `True`
- `oc get job -n ${NAMESPACE} -o jsonpath='{.items[0].spec.suspend}'` returns `false`

**Notes**: To be filled later in the process.
