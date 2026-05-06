---
test_case_id: TC-MULTI-001
source_key: RHOAIENG-59092
priority: P2
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-MULTI-001: Jobs in different namespaces use separate ClusterQueues

**Objective**: Verify that jobs submitted from different namespaces map to their respective ClusterQueues via LocalQueues, providing quota isolation between tenants.

**Preconditions**:

- Kueue Operator installed on the cluster
- Namespace `team-a-ns` created with labels: team=team-a, kueue.openshift.io/managed=true
- Namespace `team-b-ns` created with labels: team=team-b, kueue.openshift.io/managed=true
- ClusterQueue `team-a-cq` created with nominalQuota: cpu=500m, memory=2Gi, namespaceSelector matching team=team-a
- ClusterQueue `team-b-cq` created with nominalQuota: cpu=500m, memory=2Gi, namespaceSelector matching team=team-b
- LocalQueue `eval-queue` created in each namespace mapped to respective ClusterQueue
- EvalHub configured for both namespaces

**Test Steps**:

1. Create two namespaces with appropriate labels
2. Create two ClusterQueues with namespace selectors and equal quotas
3. Create LocalQueue `eval-queue` in each namespace pointing to the corresponding ClusterQueue
4. Submit job-a in team-a-ns (cpu=500m, memory=1Gi) — fills team-a-cq quota
5. Submit job-b in team-b-ns (cpu=500m, memory=1Gi) — should be admitted independently
6. Verify both jobs are admitted concurrently (different ClusterQueues)
7. Submit job-c in team-a-ns (cpu=500m, memory=1Gi) — should be pending (team-a quota full)
8. Verify job-c is pending while job-b is still running (quota isolation)
9. Teardown: Delete all jobs, LocalQueues, ClusterQueues, and test namespaces

**Expected Results**:

- Job-a and job-b are both admitted concurrently (separate ClusterQueue quotas)
- Job-c (team-a-ns) is pending because team-a-cq quota is exhausted
- Job-b (team-b-ns) is unaffected by team-a's quota exhaustion
- Each ClusterQueue shows independent `flavorsUsage`

**Notes**: To be filled later in the process.
