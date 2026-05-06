---
test_case_id: TC-PREEMPT-002
source_key: RHOAIENG-59092
priority: P2
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-PREEMPT-002: Preempted job restarts from beginning when resumed

**Objective**: Verify that a preempted job restarts from the beginning (not from checkpoint) when it is re-admitted after the preempting job completes.

**Preconditions**:
- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq-preempt` with preemption enabled (`withinClusterQueue: LowerPriority`)
- LocalQueue `eval-queue` created in test namespace
- A preemption scenario set up as per TC-PREEMPT-001

**Test Steps**:
1. Set up the preemption scenario from TC-PREEMPT-001: job-low preempted, job-high running
2. Record the pod name for job-low before preemption
3. Wait for job-high to complete or cancel job-high
4. Wait for job-low to be re-admitted (Workload `Admitted=True`)
5. Verify a new pod is created for job-low (different pod name than the pre-preemption pod)
6. Verify the Workload resource for job-low shows `Requeued=True` (history preserved)
7. Verify the new pod starts execution from the beginning (no checkpoint state)
8. Teardown: Delete all jobs, LocalQueue, and ClusterQueue

**Expected Results**:
- Job-low is re-admitted after job-high completes
- A new pod is created for job-low (pod name differs from the original)
- Workload shows `Requeued=True` even after re-admission (history preserved)
- Workload shows `Evicted=False` and `Preempted=False` with message "Previously: Preempted..."
- The job restarts from the beginning, not from any checkpoint

**Validation**:
- Compare pod names before and after preemption: they should differ
- `oc get workload <job-low-workload> -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Requeued")].status}'` returns `True`

**Notes**: To be filled later in the process.
