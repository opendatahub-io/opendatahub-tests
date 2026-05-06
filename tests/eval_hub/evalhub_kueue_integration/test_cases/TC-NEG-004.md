---
test_case_id: TC-NEG-004
source_key: RHOAIENG-59092
priority: P1
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-NEG-004: Forbidden request returns 403

**Objective**: Verify that an API request with a valid token but insufficient permissions returns HTTP 403 Forbidden.

**Preconditions**:
- EvalHub deployed and accessible via route
- A service account `limited-user` with a valid token but no permissions to create evaluation jobs

**Test Steps**:
1. Create a service account `limited-user` in the test namespace without evaluation job permissions
2. Obtain a token for `limited-user`
3. Send a POST request to `/api/v1/evaluations/jobs` with the limited-user token
4. Verify the response status code is 403
5. Teardown: Delete the `limited-user` service account

**Expected Results**:
- HTTP response status code is 403
- Response body contains an error message indicating insufficient permissions

**Test Data**:
```bash
LIMITED_TOKEN=$(oc create token limited-user -n ${NAMESPACE})

curl -s -o /dev/null -w "%{http_code}" -X POST \
  "https://${EVALHUB_ROUTE}/api/v1/evaluations/jobs" \
  -H "Authorization: Bearer ${LIMITED_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "tc-neg-004-forbidden",
    "model": {"url": "http://llm.svc:8080/v1", "name": "test"},
    "benchmarks": [{"id": "arc_easy", "provider_id": "lm_evaluation_harness"}]
  }'
```

**Notes**: To be filled later in the process.
