---
test_case_id: TC-NEG-002
source_key: RHOAIENG-59092
priority: P1
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
upgrade_phase: both
---
# TC-NEG-002: Submit job without queue specification (backwards compatibility)

**Objective**: Verify that a job submitted without a queue specification is processed by EvalHub without Kueue involvement, maintaining backwards compatibility.

**Preconditions**:

- Kueue Operator may or may not be installed
- EvalHub deployed and accessible
- No queue specification in the request body

**Test Steps**:

1. Submit a POST request to `/api/v1/evaluations/jobs` without the `queue` field
2. Verify the response status code is 202 (Accepted)
3. Verify the Kubernetes Job is created without the `kueue.x-k8s.io/queue-name` label
4. Verify no Kueue Workload resource is created for the job
5. Verify the job runs directly (not managed by Kueue)
6. Teardown: Delete the evaluation job

**Expected Results**:

- HTTP response status code is 202
- Kubernetes Job does not have the `kueue.x-k8s.io/queue-name` label
- No Workload resource exists for the job
- The job runs without Kueue queue management

**Test Data**:

```bash
curl -s -X POST \
  "https://${EVALHUB_ROUTE}/api/v1/evaluations/jobs" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "tc-neg-002-no-queue",
    "model": {
      "url": "http://llm-service.${NAMESPACE}.svc.cluster.local:8080/v1",
      "name": "granite-3.1-8b"
    },
    "benchmarks": [
      {
        "id": "arc_easy",
        "provider_id": "lm_evaluation_harness"
      }
    ]
  }'
```

**Validation**:

- `oc get workloads -n ${NAMESPACE} -o json | jq '.items | length'` shows no new Workload resource
- `oc get job -n ${NAMESPACE} -l kueue.x-k8s.io/queue-name -o json | jq '.items | length'` returns 0 for the new job

**Notes**: To be filled later in the process.
