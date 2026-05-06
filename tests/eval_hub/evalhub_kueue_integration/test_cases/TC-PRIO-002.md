---
test_case_id: TC-PRIO-002
source_key: RHOAIENG-59092
priority: P1
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-PRIO-002: Higher priority job admitted before lower priority when quota available

**Objective**: Verify that when multiple jobs are pending and quota becomes available, the higher-priority job is admitted first.

**Preconditions**:

- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=500m, memory=1Gi
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`
- A job currently consuming the full quota

**Test Steps**:

1. Create ClusterQueue `eval-cq` with cpu=500m, memory=1Gi
2. Create LocalQueue `eval-queue`
3. Submit job-a (cpu=500m, memory=1Gi) — it is admitted and consumes full quota
4. Submit job-low (priority 0, cpu=500m, memory=1Gi) — remains pending
5. Submit job-high (priority 1000, cpu=500m, memory=1Gi) — remains pending
6. Cancel job-a to free quota
7. Wait for one pending job to be admitted
8. Verify job-high is admitted first (despite job-low being submitted first)
9. Verify job-low remains pending
10. Teardown: Delete all jobs, LocalQueue, and ClusterQueue

**Expected Results**:

- After quota is freed, job-high (priority 1000) is admitted before job-low (priority 0)
- Job-high Workload shows `Admitted=True`
- Job-low Workload shows `Admitted=False`

**Validation**:

- `oc get workload <job-high-workload> -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Admitted")].status}'` returns `True`
- `oc get workload <job-low-workload> -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Admitted")].status}'` returns `False`

**Notes**: To be filled later in the process.
