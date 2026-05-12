---
feature: evalhub_kueue_integration
source_key: RHOAIENG-59092
status: Open
gap_count: 14
last_updated: '2026-05-04'
---
# Gaps — EvalHub Kueue Integration

## Scope & Endpoints

1. **Missing details on how EvalHub translates submitted jobs to Kueue Workload resources** — would be resolved by: ADR / design doc
2. **Namespace labeling requirements for LocalQueue mapping not specified** — would be resolved by: ADR / feature refinement
3. **Authentication mechanism (OAuth, API key, bearer token) not specified** — would be resolved by: API spec / design doc
4. **No details on job restart mechanism after preemption** — would be resolved by: ADR / design doc
5. **Missing specification of how resource quotas are enforced at the EvalHub level vs Kueue level** — would be resolved by: ADR
6. **No details on observability/monitoring for queue metrics and admission rates** — would be resolved by: ADR / design doc
7. **Preemption behavior validation requires clarification on expected API responses when a job is preempted** — would be resolved by: API spec / feature refinement
8. **Hard delete vs. soft delete behavior differences not fully specified (e.g., whether hard delete bypasses graceful shutdown)** — would be resolved by: ADR / design doc
9. **Per-benchmark status granularity structure not detailed in API spec (nested status schema)** — would be resolved by: API spec
10. **MLflow integration in results response schema lacks detail on artifact structure** — would be resolved by: API spec
11. **Component health details in health check response not enumerated** — would be resolved by: API spec

## Test Strategy & Risks

1. **Preemption Policy Details** — The strategy mentions preemption but does not specify which ClusterQueues should have preemption enabled vs. disabled — would be resolved by: ADR defining preemption policy per cluster queue and rationale
2. **Default Queue Configuration** — The strategy does not specify what happens if a job is submitted without queue.kind or queue.name — would be resolved by: Feature refinement clarifying default behavior (fail, use default queue, or run without Kueue)
3. **Best practice guidance on disabling preemption for evaluation jobs is mentioned but not reflected in API validation rules** — would be resolved by: API spec / feature refinement
4. **Service Account Permissions** — The strategy mentions service account permissions but does not list required RBAC permissions for EvalHub to interact with Kueue — would be resolved by: Design doc specifying exact Kubernetes RBAC roles and bindings required
5. **Cohort Borrowing Configuration** — The strategy mentions cohort-based resource borrowing but does not explain when or how it should be configured — would be resolved by: ADR defining cohort strategy and when teams should share quota

## Environment & Infrastructure

1. **OpenShift and RHOAI version requirements not specified** — would be resolved by: ADR / design doc

## Test Case Coverage Gaps

No coverage gaps found. All validation checks passed:

- **Endpoint coverage**: All 11 endpoints/resources from Section 4 have at least one test case
- **E2E coverage**: All 4 P0 endpoints have E2E scenario coverage (TC-E2E-001, TC-E2E-002, TC-E2E-003)
- **Test objective coverage**: All 7 test objectives from Section 1.3 are addressed by at least one test case
- **Priority distribution**: All P0 endpoints have P0-priority test cases
- **Gap cross-reference**: No test cases were created for areas flagged as pending/missing in this gaps document
