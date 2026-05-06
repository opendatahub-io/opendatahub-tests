---
test_case_id: TC-PREEMPT-003
source_key: RHOAIENG-59092
priority: P2
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-PREEMPT-003: No preemption when withinClusterQueue is set to Never

**Objective**: Verify that when a ClusterQueue has `withinClusterQueue: Never` (default), higher-priority jobs do not preempt lower-priority running jobs.

**Preconditions**:
- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq-nopreempt` created with:
  - nominalQuota: cpu=500m, memory=1Gi
  - preemption.withinClusterQueue: Never (or no preemption section, which defaults to Never)
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq-nopreempt`

**Test Steps**:
1. Create ClusterQueue `eval-cq-nopreempt` without specifying preemption (defaults to `withinClusterQueue: Never`)
2. Create LocalQueue `eval-queue` in test namespace
3. Submit job-low (priority 100, cpu=500m, memory=1Gi) — it is admitted and running
4. Wait for job-low to be admitted
5. Submit job-high (priority 1000, cpu=500m, memory=1Gi)
6. Wait 60 seconds
7. Verify job-low is still running (NOT preempted)
8. Verify job-high remains pending (quota exhausted, no preemption allowed)
9. Verify no Preempted or Evicted conditions exist on job-low's Workload
10. Teardown: Delete both jobs, LocalQueue, and ClusterQueue

**Expected Results**:
- Job-low remains running with `Admitted=True`, no `Preempted` or `Evicted` conditions
- Job-high remains pending with `Admitted=False`
- No preemption events in the namespace

**Validation**:
- `oc get workload <job-low-workload> -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Preempted")].status}'` returns empty or does not exist
- `oc get events -n ${NAMESPACE} | grep -i preempt | wc -l` returns `0`

**Notes**: To be filled later in the process.
