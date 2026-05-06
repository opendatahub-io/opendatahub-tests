---
test_case_id: TC-NEG-005
source_key: RHOAIENG-59092
priority: P1
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-NEG-005: Get non-existent job returns 404

**Objective**: Verify that GET /api/v1/evaluations/jobs/{id} returns HTTP 404 when the job ID does not exist.

**Preconditions**:
- EvalHub deployed and accessible via route
- A valid bearer token for the API

**Test Steps**:
1. Send a GET request to `/api/v1/evaluations/jobs/00000000-0000-0000-0000-000000000000`
2. Verify the response status code is 404
3. Verify the response body contains a not-found error message

**Expected Results**:
- HTTP response status code is 404
- Response body indicates the resource was not found

**Test Data**:
```bash
curl -s -o /dev/null -w "%{http_code}" -X GET \
  "https://${EVALHUB_ROUTE}/api/v1/evaluations/jobs/00000000-0000-0000-0000-000000000000" \
  -H "Authorization: Bearer ${TOKEN}"
```

**Notes**: To be filled later in the process.
