---
test_case_id: TC-API-003
source_key: RHOAIENG-59092
priority: P1
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-API-003: List evaluation jobs with status filter

**Objective**: Verify that GET /api/v1/evaluations/jobs returns a filtered list of jobs when status query parameter is provided.

**Preconditions**:
- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=2, memory=8Gi
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`
- At least one evaluation job previously submitted

**Test Steps**:
1. Set up Kueue resources and submit two evaluation jobs
2. Send GET request to `/api/v1/evaluations/jobs?status=pending&limit=10`
3. Verify the response status code is 200
4. Verify the response contains a list of jobs matching the status filter
5. Verify pagination fields are present (limit, offset)
6. Teardown: Delete evaluation jobs, LocalQueue, and ClusterQueue

**Expected Results**:
- HTTP response status code is 200
- Response body contains a list of evaluation jobs
- All returned jobs have `status.state` matching `pending`
- Response respects the `limit` parameter

**Test Data**:
```bash
curl -s -X GET \
  "https://${EVALHUB_ROUTE}/api/v1/evaluations/jobs?status=pending&limit=10" \
  -H "Authorization: Bearer ${TOKEN}"
```

**Notes**: To be filled later in the process.
