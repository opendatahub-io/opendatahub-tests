---
test_case_id: TC-QUOTA-001
source_key: RHOAIENG-59092
priority: P0
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-QUOTA-001: Multiple jobs admitted within ClusterQueue quota limits

**Objective**: Verify that multiple concurrent jobs are admitted when their combined resource requests fit within the ClusterQueue quota.

**Preconditions**:

- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=1, memory=4Gi
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`
- Cluster capacity: 2 CPU, 8 GB (sufficient for all jobs plus system overhead)

**Test Steps**:

1. Create ClusterQueue `eval-cq` with cpu=1, memory=4Gi quota
2. Create LocalQueue `eval-queue` in test namespace
3. Submit job-a requesting cpu=300m, memory=1Gi with `queue.name: "eval-queue"`
4. Submit job-b requesting cpu=300m, memory=1Gi with `queue.name: "eval-queue"`
5. Wait for both jobs to be admitted
6. Verify both Workload resources show `Admitted=True`
7. Verify both Kubernetes Jobs are unsuspended
8. Submit job-c requesting cpu=500m, memory=2Gi (exceeds remaining quota)
9. Verify job-c remains pending (quota exhausted: 600m of 1000m used, 500m more needed)
10. Teardown: Delete all jobs, LocalQueue, and ClusterQueue

**Expected Results**:

- Job-a and job-b are both admitted concurrently
- Both have `Admitted=True` in their Workload resources
- Job-c remains pending with `Admitted=False` due to insufficient remaining quota
- Total admitted quota usage is 600m CPU, 2Gi memory (within the 1 CPU, 4Gi limit)

**Notes**: To be filled later in the process.
