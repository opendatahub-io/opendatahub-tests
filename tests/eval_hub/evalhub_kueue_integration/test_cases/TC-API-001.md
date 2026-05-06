---
test_case_id: TC-API-001
source_key: RHOAIENG-59092
priority: P0
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
---
# TC-API-001: Submit evaluation job with Kueue queue specification

**Objective**: Verify that an evaluation job submitted via POST /api/v1/evaluations/jobs with a valid Kueue queue specification is accepted with HTTP 202.

**Preconditions**:
- Kueue Operator installed on the cluster
- ClusterQueue `eval-cq` created with nominalQuota: cpu=2, memory=8Gi
- LocalQueue `eval-queue` created in test namespace mapped to `eval-cq`
- Namespace labeled with `kueue.openshift.io/managed=true`
- EvalHub deployed and accessible via route

**Test Steps**:
1. Set up Kueue resources (ClusterQueue, LocalQueue) as per preconditions
2. Obtain a valid bearer token for the EvalHub API
3. Submit a POST request to `/api/v1/evaluations/jobs` with queue specification
4. Verify the response status code is 202 (Accepted)
5. Verify the response body contains a `resource.id` field
6. Verify the response body contains `status.state` as `pending`
7. Teardown: Delete the created evaluation job, LocalQueue, and ClusterQueue

**Expected Results**:
- HTTP response status code is 202
- Response body contains `resource.id` with a non-empty UUID
- Response body contains `status.state` set to `pending`

**Test Data**:
```bash
TOKEN=$(oc create token evalhub-client -n ${NAMESPACE})

curl -s -X POST \
  "https://${EVALHUB_ROUTE}/api/v1/evaluations/jobs" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "tc-api-001-kueue-submit",
    "model": {
      "url": "http://llm-service.${NAMESPACE}.svc.cluster.local:8080/v1",
      "name": "granite-3.1-8b"
    },
    "queue": {
      "kind": "kueue",
      "name": "eval-queue"
    },
    "benchmarks": [
      {
        "id": "arc_easy",
        "provider_id": "lm_evaluation_harness"
      }
    ]
  }'
```

**Expected Response**:
```json
{
  "resource": {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "created_at": "2026-05-04T10:00:00Z"
  },
  "status": {
    "state": "pending",
    "message": {
      "message": "Evaluation job created",
      "message_code": "evaluation_job_created"
    }
  }
}
```

**Validation**:
- Verify a Kubernetes Job is created in the namespace with the label `kueue.x-k8s.io/queue-name: eval-queue`
- Verify a Kueue Workload resource is created for the job

**Notes**: To be filled later in the process.
