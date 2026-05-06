---
test_case_id: TC-API-004
source_key: RHOAIENG-59092
priority: P0
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-API-004: Cancel evaluation job via soft delete

**Objective**: Verify that DELETE /api/v1/evaluations/jobs/{id} cancels a Kueue-managed job with soft delete (default behavior).

**Preconditions**:

- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=2, memory=8Gi
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`
- An evaluation job submitted and in pending or running state

**Test Steps**:

1. Set up Kueue resources and submit an evaluation job
2. Verify the job is in pending or running state
3. Send DELETE request to `/api/v1/evaluations/jobs/{resource_id}`
4. Verify the response status code is 204
5. Send GET request to `/api/v1/evaluations/jobs/{resource_id}` to confirm job is cancelled
6. Verify the Kubernetes Job is terminated
7. Verify the Kueue Workload resource reflects the cancellation
8. Teardown: Clean up remaining Kueue resources (LocalQueue, ClusterQueue)

**Expected Results**:

- HTTP response status code is 204 (No Content)
- Subsequent GET returns the job with a terminal status
- The Kubernetes Job associated with the evaluation is terminated or deleted
- The Kueue Workload resource is cleaned up

**Test Data**:

```bash
curl -s -X DELETE \
  "https://${EVALHUB_ROUTE}/api/v1/evaluations/jobs/${RESOURCE_ID}" \
  -H "Authorization: Bearer ${TOKEN}"
```

**Notes**: To be filled later in the process.
