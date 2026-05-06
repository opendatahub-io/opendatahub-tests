---
test_case_id: TC-NEG-001
source_key: RHOAIENG-59092
priority: P0
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-NEG-001: Submit job with non-existent queue name returns error

**Objective**: Verify that submitting a job with a queue name that does not match any existing LocalQueue results in an appropriate error response.

**Preconditions**:

- Kueue Operator installed on the cluster
- No LocalQueue named `nonexistent-queue` exists in the test namespace
- EvalHub deployed and accessible

**Test Steps**:

1. Ensure Kueue is installed but no LocalQueue named `nonexistent-queue` exists
2. Submit a POST request to `/api/v1/evaluations/jobs` with `queue.name: "nonexistent-queue"`
3. Verify the response indicates an error (400 Bad Request or the job enters a permanently pending state)
4. If the job is accepted (202), verify the Workload resource shows admission failure with a descriptive message
5. Teardown: Delete any created resources

**Expected Results**:

- Either: HTTP 400 with an error message indicating the queue does not exist
- Or: HTTP 202 followed by the Workload showing `Admitted=False` with a message referencing the missing queue

**Test Data**:

```bash
curl -s -X POST \
  "https://${EVALHUB_ROUTE}/api/v1/evaluations/jobs" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "tc-neg-001-invalid-queue",
    "model": {
      "url": "http://llm-service.${NAMESPACE}.svc.cluster.local:8080/v1",
      "name": "granite-3.1-8b"
    },
    "queue": {
      "kind": "kueue",
      "name": "nonexistent-queue"
    },
    "benchmarks": [
      {
        "id": "arc_easy",
        "provider_id": "lm_evaluation_harness"
      }
    ]
  }'
```

**Notes**: To be filled later in the process.
