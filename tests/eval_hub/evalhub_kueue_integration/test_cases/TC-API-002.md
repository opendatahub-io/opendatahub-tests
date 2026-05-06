---
test_case_id: TC-API-002
source_key: RHOAIENG-59092
priority: P0
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-API-002: Retrieve job status for Kueue-managed job

**Objective**: Verify that GET /api/v1/evaluations/jobs/{id} returns accurate status for a Kueue-managed evaluation job.

**Preconditions**:
- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=2, memory=8Gi
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`
- An evaluation job previously submitted via TC-API-001 or equivalent

**Test Steps**:
1. Set up Kueue resources and submit an evaluation job with queue specification
2. Capture the `resource.id` from the submission response
3. Send GET request to `/api/v1/evaluations/jobs/{resource_id}`
4. Verify the response status code is 200
5. Verify the response contains the job configuration and current status
6. Verify the status reflects the Kueue-managed state (pending, running, or completed)
7. Teardown: Delete the evaluation job, LocalQueue, and ClusterQueue

**Expected Results**:
- HTTP response status code is 200
- Response body contains `resource.id` matching the submitted job ID
- Response body contains `status.state` with a valid value (pending, running, or completed)
- Response body contains the original job configuration (name, model, benchmarks, queue)

**Test Data**:
```bash
RESOURCE_ID="<id-from-submission>"

curl -s -X GET \
  "https://${EVALHUB_ROUTE}/api/v1/evaluations/jobs/${RESOURCE_ID}" \
  -H "Authorization: Bearer ${TOKEN}"
```

**Notes**: To be filled later in the process.
