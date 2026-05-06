---
test_case_id: TC-QUEUE-003
source_key: RHOAIENG-59092
priority: P1
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-QUEUE-003: Pending job auto-admitted after resources freed

**Objective**: Verify that a pending job is automatically admitted once quota becomes available after a running job completes or is cancelled.

**Preconditions**:

- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=500m, memory=1Gi
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`
- Two jobs submitted: job-a admitted and running, job-b pending (as per TC-QUEUE-002)

**Test Steps**:

1. Set up the state from TC-QUEUE-002: job-a running, job-b pending
2. Cancel or wait for job-a to complete (DELETE /api/v1/evaluations/jobs/{job-a-id})
3. Verify that quota is released from job-a
4. Wait for job-b to be auto-admitted (poll Workload status for up to 60 seconds)
5. Verify job-b Workload shows `Admitted=True` and `QuotaReserved=True`
6. Verify job-b Kubernetes Job is unsuspended (`spec.suspend: false`)
7. Verify EvalHub API reports job-b as `running`
8. Teardown: Delete job-b, LocalQueue, and ClusterQueue

**Expected Results**:

- After job-a is cancelled/completed, job-b transitions from pending to admitted
- Job-b Workload shows `Admitted=True`
- Job-b Kubernetes Job has `spec.suspend: false`
- EvalHub API reports job-b `status.state: "running"`

**Notes**: To be filled later in the process.
