---
test_case_id: TC-QUEUE-002
source_key: RHOAIENG-59092
priority: P0
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-QUEUE-002: Job remains pending when ClusterQueue quota is exhausted

**Objective**: Verify that a job submitted when the ClusterQueue quota is fully consumed stays in pending state and is not admitted.

**Preconditions**:

- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=500m, memory=1Gi (deliberately small)
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`
- An existing job already admitted and consuming the full quota (cpu=500m, memory=1Gi)

**Test Steps**:

1. Create ClusterQueue `eval-cq` with small quota: cpu=500m, memory=1Gi
2. Create LocalQueue `eval-queue` in test namespace
3. Submit a first evaluation job (job-a) requesting cpu=500m, memory=1Gi — it should be admitted and consume full quota
4. Wait for job-a to be admitted
5. Submit a second evaluation job (job-b) requesting cpu=500m, memory=1Gi
6. Wait 30 seconds and verify job-b remains in pending state
7. Verify the Workload resource for job-b shows `Admitted=False` and `QuotaReserved=False`
8. Verify the Kubernetes Job for job-b has `spec.suspend: true`
9. Verify the EvalHub API reports job-b as `pending`
10. Teardown: Delete both jobs, LocalQueue, and ClusterQueue

**Expected Results**:

- Job-b Workload resource shows `Admitted=False`
- Job-b Kubernetes Job remains suspended (`spec.suspend: true`)
- EvalHub API reports job-b `status.state: "pending"`
- Job-b QuotaReserved condition shows insufficient quota message

**Validation**:

- `oc get workload <job-b-workload> -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Admitted")].status}'` returns `False`
- `oc get workload <job-b-workload> -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="QuotaReserved")].message}'` contains "insufficient"

**Notes**: To be filled later in the process.
