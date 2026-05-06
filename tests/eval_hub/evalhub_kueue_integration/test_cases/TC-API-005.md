---
test_case_id: TC-API-005
source_key: RHOAIENG-59092
priority: P0
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-API-005: Cancel evaluation job via hard delete

**Objective**: Verify that DELETE /api/v1/evaluations/jobs/{id}?hard_delete=true permanently removes a Kueue-managed job.

**Preconditions**:
- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=2, memory=8Gi
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`
- An evaluation job submitted and in pending or running state

**Test Steps**:
1. Set up Kueue resources and submit an evaluation job
2. Capture the `resource.id` from the submission response
3. Send DELETE request to `/api/v1/evaluations/jobs/{resource_id}?hard_delete=true`
4. Verify the response status code is 204
5. Send GET request to `/api/v1/evaluations/jobs/{resource_id}`
6. Verify the GET returns 404 (job permanently deleted)
7. Verify no Kubernetes Job or Workload resource remains for this evaluation
8. Teardown: Clean up remaining Kueue resources (LocalQueue, ClusterQueue)

**Expected Results**:
- HTTP response status code is 204 (No Content)
- Subsequent GET returns 404 (Not Found)
- No Kubernetes Job associated with the evaluation exists in the namespace
- No Kueue Workload resource exists for the evaluation

**Test Data**:
```bash
curl -s -X DELETE \
  "https://${EVALHUB_ROUTE}/api/v1/evaluations/jobs/${RESOURCE_ID}?hard_delete=true" \
  -H "Authorization: Bearer ${TOKEN}"
```

**Notes**: To be filled later in the process.
