---
test_case_id: TC-PREEMPT-001
source_key: RHOAIENG-59092
priority: P2
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-PREEMPT-001: Higher priority job preempts lower priority when preemption enabled

**Objective**: Verify that when a ClusterQueue has `withinClusterQueue: LowerPriority`, a higher-priority job preempts a running lower-priority job.

**Preconditions**:

- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq-preempt` created with:
  - nominalQuota: cpu=500m, memory=1Gi
  - preemption.withinClusterQueue: LowerPriority
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq-preempt`

**Test Steps**:

1. Create ClusterQueue `eval-cq-preempt` with preemption enabled (`withinClusterQueue: LowerPriority`) and quota cpu=500m, memory=1Gi
2. Create LocalQueue `eval-queue` in test namespace
3. Submit job-low (priority 100, cpu=500m, memory=1Gi) — it is admitted and consumes full quota
4. Wait for job-low to be admitted and running
5. Submit job-high (priority 1000, cpu=500m, memory=1Gi)
6. Wait for Kueue to preempt job-low
7. Verify job-low Workload shows `Evicted=True` and `Preempted=True`
8. Verify job-low Kubernetes Job is suspended (`spec.suspend: true`)
9. Verify job-high Workload shows `Admitted=True`
10. Verify job-high Kubernetes Job is unsuspended
11. Teardown: Delete both jobs, LocalQueue, and ClusterQueue

**Expected Results**:

- Job-low is preempted: Workload conditions show `Evicted=True`, `Preempted=True`, `Requeued=True`
- Job-low Kubernetes Job is re-suspended (`spec.suspend: true`)
- Job-high is admitted and running
- Kubernetes Events show preemption sequence (EvictedDueToPreempted, Preempted, Suspended)

**Validation**:

- `oc get workload <job-low-workload> -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Preempted")].status}'` returns `True`
- `oc get events -n ${NAMESPACE} --sort-by='.lastTimestamp' | grep -i preempt` shows preemption events

**Notes**: To be filled later in the process.
