---
test_case_id: TC-NEG-003
source_key: RHOAIENG-59092
priority: P0
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-NEG-003: Unauthorized request returns 401

**Objective**: Verify that an API request without a valid authorization token returns HTTP 401 Unauthorized.

**Preconditions**:
- EvalHub deployed and accessible via route

**Test Steps**:
1. Send a POST request to `/api/v1/evaluations/jobs` without an Authorization header
2. Verify the response status code is 401
3. Send a GET request to `/api/v1/evaluations/jobs` without an Authorization header
4. Verify the response status code is 401
5. Send a request with an expired or invalid token
6. Verify the response status code is 401

**Expected Results**:
- All requests without valid authorization return HTTP 401
- Response body contains an error message indicating authentication failure

**Test Data**:
```bash
# No Authorization header
curl -s -o /dev/null -w "%{http_code}" -X POST \
  "https://${EVALHUB_ROUTE}/api/v1/evaluations/jobs" \
  -H "Content-Type: application/json" \
  -d '{"name": "tc-neg-003-unauth"}'

# Invalid token
curl -s -o /dev/null -w "%{http_code}" -X GET \
  "https://${EVALHUB_ROUTE}/api/v1/evaluations/jobs" \
  -H "Authorization: Bearer invalid-token-value"
```

**Notes**: To be filled later in the process.
