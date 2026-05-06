---
feature: evalhub_kueue_integration
source_key: RHOAIENG-59092
source_type: issue
version: 1.0.0
status: Draft
author: RHOAI QE
components: []
additional_docs: []
last_updated: '2026-05-04'
reviewers: []
---
# EvalHub Kueue Integration Test Plan

## RHOAI QE – Evaluation Job Queue Management

**Strategy**: [RHOAIENG-59092](https://issues.redhat.com/browse/RHOAIENG-59092)

---

## 1. Executive Summary

### 1.1 Purpose

This test plan validates the integration of Kueue with EvalHub to enable production-ready LLM evaluation job management with fair resource sharing, priority-based scheduling, and automatic queueing. The integration allows multiple teams to run evaluation benchmarks simultaneously on limited GPU/CPU resources while preventing cluster instability and job failures due to resource contention. Testing will verify that jobs can be submitted with queue specifications, admitted based on resource quotas, queued when resources are unavailable, and managed through their complete lifecycle including preemption scenarios.

### 1.2 Scope

#### In Scope (RHOAI QE Responsibilities)

- Job submission via POST /api/v1/evaluations/jobs with Kueue queue specification (queue.kind: "kueue", queue.name)
- Job status retrieval via GET /api/v1/evaluations/jobs/{id} with Kueue-managed state tracking
- Job listing via GET /api/v1/evaluations/jobs with filtering (status, name, tags)
- Job cancellation via DELETE /api/v1/evaluations/jobs/{id} (soft and hard delete modes)
- Health check endpoint GET /api/v1/health with active job counts
- Job lifecycle states (pending, running, completed) transitions under Kueue management
- Default priority assignment (priority 0) for submitted jobs
- Authentication and authorization (401, 403) for all endpoints
- API validation and error responses (400, 404)
- Detailed job queue status via Kubernetes Workload resources
- Preemption scenarios and job restart behavior
- Resource quota enforcement across multi-tenant environments
- ClusterQueue and LocalQueue configuration for evaluation workloads

#### Out of Scope (Other Teams)

- Internal Kueue implementation details (only the integration interface is tested)
- Cluster administration tasks (Kueue installation, ClusterQueue/ResourceFlavor creation)
- MLflow integration internal logic (tested only as part of job results validation)
- Benchmark execution logic (tested only that Kueue-queued jobs trigger execution correctly)
- Model authentication mechanisms (tested only that auth parameters are accepted and passed through)
- Performance testing and load testing (functional validation only)
- Job checkpointing functionality (evaluation workloads cannot checkpoint progress)

### 1.3 Test Objectives

1. Verify that jobs submitted with valid Kueue queue configuration via POST /api/v1/evaluations/jobs are successfully accepted (202) and queued
2. Validate that job lifecycle states transition correctly (pending → running → completed) and are accurately reflected in GET /api/v1/evaluations/jobs/{id} responses
3. Confirm that default priority (0) is applied when not specified, and custom priorities are honored
4. Verify that job preemption behavior works as configured, with preempted jobs restarting from the beginning
5. Ensure that job status retrieval via GET /api/v1/evaluations/jobs/{id} accurately reflects Kueue-managed state and per-benchmark granularity
6. Validate that job cancellation via DELETE /api/v1/evaluations/jobs/{id} correctly terminates Kueue-managed jobs (both soft and hard delete)
7. Confirm that authentication, authorization, and validation errors (400, 401, 403, 404) are properly handled across all endpoints

---

## 2. Test Strategy

### 2.1 Test Levels

- **API Integration Testing** — Validate EvalHub API accepts queue specifications (queue.kind, queue.name) and correctly translates job submissions into Kubernetes resources with Kueue workloads
- **Functional Testing** — Verify job queueing, admission, priority handling, quota enforcement, and integration between EvalHub service and Kueue controllers
- **Kubernetes Resource Testing** — Validate Workload resources are created, LocalQueue/ClusterQueue mappings work, and resource quota enforcement operates correctly
- **Status Reporting Testing** — Confirm EvalHub API reports job states (pending, running, completed) and Kubernetes Workload resources expose detailed queue position and admission status

### 2.2 Test Types

- **Positive Testing** — Valid job submissions with proper queue specifications, successful admission when quota available, normal job lifecycle completion
- **Negative Testing** — Invalid queue names, missing queue specifications, jobs submitted to non-existent LocalQueues, quota exceeded scenarios
- **Boundary Testing** — Maximum concurrent jobs in queue, quota limits, priority ranges, cohort borrowing limits
- **Regression Testing** — Ensure existing EvalHub job submission without Kueue still works, backwards compatibility with non-Kueue workflows

### 2.3 Test Priorities

- **P0 (Critical)** - Core Kueue integration functionality: job submission with queue specification, job admission when quota available, job execution and completion, resource quota enforcement
- **P1 (High)** - Priority-based scheduling, queue status visibility via API and Kubernetes, preemption behavior (if enabled), cohort-based resource borrowing
- **P2 (Medium)** - Edge cases like queue reconfiguration, LocalQueue to ClusterQueue remapping, detailed Workload status conditions, multi-tenant quota isolation

---

## 3. Test Environment

### 3.1 Test Cluster Configuration

- OpenShift cluster with Kubernetes 1.21+ (Kueue requirement)
- RHOAI/ODH platform installed
- **Cluster Capacity**: 1 node with 2 CPU cores, 8 GB memory, 0 GPUs
- Kueue Operator installation and configuration performed per scenario (may or may not be pre-installed)
- Kueue Operator version read from config or environment variable (hard-coded version for initial implementation)
- Multi-namespace setup to simulate multi-tenancy (within resource constraints)

### 3.2 Test Data Requirements

- Sample LocalQueue YAML manifests with various queue configurations
- Sample ClusterQueue YAML manifests with different resource quotas (CPU, memory) and preemption policies
- Sample ResourceFlavor definitions for CPU and memory resources (no GPU resources)
- ClusterQueue configurations constrained to cluster capacity (max 2 CPU, 8 GB memory across all queues)
- EvalHub job submission payloads with queue specifications (queue.kind: "kueue", queue.name)
- Namespace label configurations (team labels, kueue.openshift.io/managed, evalhub.trustyai.opendatahub.io/tenant)
- Sample Kueue Workload resources for status validation
- Job priority configurations (default priority 0, high/low priority variants)
- Job resource requests fitting within cluster capacity (e.g., 500m CPU, 1Gi memory per job for multi-job scenarios)

### 3.3 Test Users

- Cluster Administrator with permissions to install operators and create cluster-scoped resources (ClusterQueues, ResourceFlavors)
- Namespace Owner with permissions to create LocalQueues, label namespaces, and configure namespace-scoped resources
- EvalHub User with permissions to submit jobs via API and monitor job status
- Service accounts for EvalHub API access
- Multiple tenant users for multi-tenancy testing (team-a, team-b)

---

## 4. API Endpoints / Kubernetes Resources Under Test

| Endpoint/Method | Type | Purpose | Priority |
| --- | --- | --- | --- |
| POST /api/v1/evaluations/jobs | REST | Submit evaluation job with Kueue queue configuration (queue.kind, queue.name) | P0 |
| GET /api/v1/evaluations/jobs/{id} | REST | Retrieve job status and results for Kueue-managed job | P0 |
| GET /api/v1/evaluations/jobs | REST | List jobs with filtering (status, name, tags, limit, offset) | P1 |
| DELETE /api/v1/evaluations/jobs/{id} | REST | Cancel Kueue-managed job (soft or hard delete) | P0 |
| GET /api/v1/health | REST | Health check with active job counts | P2 |
| Kubernetes Workload Resource | K8s API | Query detailed queue status including queue position, preemption, and admission conditions | P1 |
| LocalQueue Resource | K8s Config | Namespace-level queue configuration mapping to ClusterQueues | P0 |
| ClusterQueue Resource | K8s Config | Cluster-level queue configuration with resource quotas and preemption policies | P1 |
| Job Priority Field | Config | Default priority assignment (0) for submitted jobs | P1 |
| Preemption Policy (withinClusterQueue) | Config | Configure preemption behavior (Never, LowerPriority) | P2 |
| Workload Status Conditions | K8s API | Status reporting for Evicted, Preempted, Requeued states | P1 |

---

## 5. Test Cases

**28 test cases generated** across 10 categories. See [test_cases/INDEX.md](test_cases/INDEX.md) for the complete index.

**Test Cases Directory**: [test_cases/](test_cases/)
**Complete Test Case Index**: [test_cases/INDEX.md](test_cases/INDEX.md)

### 5.1 Test Case Organization

| Category | Test Cases | Priority Distribution |
| --- | --- | --- |
| API Integration (TC-API) | 5 | P0: 4, P1: 1 |
| Queue Management (TC-QUEUE) | 3 | P0: 2, P1: 1 |
| Resource Quota (TC-QUOTA) | 2 | P0: 2 |
| Priority Scheduling (TC-PRIO) | 2 | P1: 2 |
| Preemption (TC-PREEMPT) | 3 | P2: 3 |
| Status Reporting (TC-STATUS) | 2 | P1: 2 |
| Multi-Tenancy (TC-MULTI) | 1 | P2: 1 |
| Negative Testing (TC-NEG) | 5 | P0: 2, P1: 3 |
| End-to-End (TC-E2E) | 3 | P0: 3 |
| Upgrade Testing (TC-UPGRADE) | 2 | P1: 2 |
| **Total** | **28** | **P0: 13, P1: 11, P2: 4** |

### 5.2 Test Case Naming Convention

Test cases follow the naming pattern: `TC-<CATEGORY>-<NUMBER>`

- **TC-API-** - API Integration testing for job submission and status retrieval
- **TC-QUEUE-** - Queue management and admission testing
- **TC-QUOTA-** - Resource quota enforcement testing
- **TC-PRIO-** - Priority-based scheduling testing
- **TC-PREEMPT-** - Preemption scenario testing
- **TC-STATUS-** - Status reporting and visibility testing
- **TC-MULTI-** - Multi-tenancy and namespace isolation testing
- **TC-E2E-** - End-to-end scenario testing
- **TC-NEG-** - Negative testing for error handling
- **TC-UPGRADE-** - Upgrade and rollback scenario testing

---

## 6. E2E Test Scenarios

End-to-end scenarios that validate the user journeys defined in the strategy. Each scenario maps to one or more TC-E2E-*.md test cases generated by `/test-plan.create-cases`.

> **Requirement**: At least one E2E scenario MUST be generated for each P0 endpoint in Section 4.
> E2E scenarios will be filled by `/test-plan.create-cases`.

### 6.1 Scenario Summary

| ID | Scenario | Endpoints Covered | Priority |
| --- | --- | --- | --- |
| TC-E2E-001 | Complete job lifecycle — submit, queue, admit, run, complete | POST jobs, GET jobs/{id}, LocalQueue, ClusterQueue, Workload | P0 |
| TC-E2E-002 | Job queuing under resource pressure — submit, wait, admit, complete | POST jobs, GET jobs/{id}, DELETE jobs/{id}, LocalQueue, ClusterQueue | P0 |
| TC-E2E-003 | Job cancellation during execution — submit, admit, run, cancel | POST jobs, GET jobs/{id}, DELETE jobs/{id}, Workload | P0 |

### 6.2 E2E Coverage Matrix

| Endpoint (from Section 4) | E2E Scenarios |
| --- | --- |
| POST /api/v1/evaluations/jobs | TC-E2E-001, TC-E2E-002, TC-E2E-003 |
| GET /api/v1/evaluations/jobs/{id} | TC-E2E-001, TC-E2E-002, TC-E2E-003 |
| GET /api/v1/evaluations/jobs | — |
| DELETE /api/v1/evaluations/jobs/{id} | TC-E2E-002, TC-E2E-003 |
| GET /api/v1/health | — |
| Kubernetes Workload Resource | TC-E2E-001, TC-E2E-003 |
| LocalQueue Resource | TC-E2E-001, TC-E2E-002 |
| ClusterQueue Resource | TC-E2E-001, TC-E2E-002 |
| Job Priority Field | — |
| Preemption Policy (withinClusterQueue) | — |
| Workload Status Conditions | TC-E2E-001 |

---

## 7. Non-Functional Requirements

Each category below must be explicitly addressed. If a category does not apply to this feature, state **Not Applicable** with a brief justification.

### 7.1 Disconnected/Air-Gapped

**Not Applicable** — The strategy describes integration between EvalHub (already deployed) and Kueue (Kubernetes-native controller). There is no mention of external registries, runtime image pulls, or network-dependent catalog sources specific to this integration. Kueue itself is installed by cluster administrators as a prerequisite.

### 7.2 Upgrade/Migration

- **CRD Schema Changes** — Kueue introduces new CRDs (ClusterQueue, LocalQueue, ResourceFlavor, Workload). Test that upgrading Kueue versions does not break existing EvalHub job submissions.
- **API Version Changes** — Validate that EvalHub can handle Kueue API version changes (e.g., v1beta1 → v1) without breaking job submissions.
- **Backwards Compatibility** — Ensure EvalHub API remains backwards compatible for users who do NOT specify queue parameters (jobs should work without Kueue).
- **Data Migration** — If EvalHub persists queue configuration or job metadata, test that existing jobs in the database can be re-queued or re-submitted after an upgrade.
- **Rollback Scenarios** — Test that disabling Kueue integration (reverting to non-Kueue mode) does not corrupt job state or prevent job submissions.

### 7.3 Performance/Scalability

- **API Response Time** — Measure latency for job submission when Kueue is enabled vs. disabled. Ensure queue specification does not add significant overhead.
- **Queue Admission Latency** — Test how long jobs remain in pending state before admission when cluster is at quota vs. when quota is available.
- **Large Queue Depth** — Validate behavior when hundreds of jobs are queued: does EvalHub API degrade? Does Kueue controller handle the workload?
- **Concurrent Job Submission** — Test multiple users submitting jobs simultaneously to the same LocalQueue and different LocalQueues. Ensure no race conditions or quota violations.
- **Resource Consumption** — Monitor EvalHub service and Kueue controller resource usage (CPU, memory) as queue depth increases.

### 7.4 RBAC/Authorization

- **LocalQueue Access Control** — Test that users in one namespace cannot submit jobs to LocalQueues in another namespace (multi-tenant isolation).
- **Service Account Permissions** — Validate that EvalHub service account has correct permissions to create/update Workload resources and query queue status.
- **User Role Boundaries** — Ensure EvalHub API enforces that users can only submit jobs to queues they have access to, based on namespace RBAC.
- **Privilege Escalation Prevention** — Test that users cannot bypass quota limits by specifying arbitrary queue names or manipulating queue.kind parameter.

---

## 8. Risks and Mitigation

| Risk | Impact | Probability | Mitigation |
| --- | --- | --- | --- |
| Preemption causes jobs to restart from beginning with no checkpointing, wasting resources | High | Medium | Create dedicated ClusterQueue for evaluation jobs with preemption disabled (withinClusterQueue: Never) as per best practice. Document this requirement clearly. |
| EvalHub API does not expose preemption status, users cannot tell if job was preempted | Medium | High | Enhance EvalHub API to surface Workload status.conditions (Evicted, Preempted, Requeued). Until implemented, document that users must check Kubernetes Workload resource directly. |
| Dependency on cluster administrator to install Kueue and configure ClusterQueues before EvalHub can use it | High | High | Provide installation documentation, automated testing, and validation that Kueue is available before EvalHub enables integration. Consider helm chart or operator to bundle prerequisites. |
| Jobs submitted with invalid queue names fail silently or get stuck in pending state | Medium | Medium | Add validation in EvalHub API to check that specified LocalQueue exists before accepting job submission. Return clear error messages. |
| Quota exhaustion causes jobs to queue indefinitely with no user notification | Medium | High | Implement queue position and estimated wait time in EvalHub API responses. Add configurable timeout for maximum queue wait time. |
| Kueue API version changes (v1beta1 → v1) may break EvalHub integration | High | Medium | Pin to stable Kueue API version. Add integration tests that run against multiple Kueue versions. Monitor Kueue release notes for breaking changes. |
| Multi-tenant quota isolation failure allows one team to consume another team's quota | High | Low | Test multi-namespace scenarios rigorously. Validate that LocalQueue to ClusterQueue mappings are correctly enforced by Kueue. |

---

## 9. Test Environment Requirements

### 9.1 Infrastructure

- Single-node Kubernetes cluster (2 CPU cores, 8 GB memory, 0 GPUs)
- Multi-namespace setup (at least 2 namespaces for multi-tenancy scenarios)
- Kueue Operator deployment with cluster instance (version from config/env var, hard-coded for initial implementation)
- EvalHub service deployment with API endpoint
- ResourceFlavors configured for CPU and memory workloads (no GPU flavors)
- ClusterQueues configured with resource quotas constrained to cluster capacity (e.g., team-a-cq: 1 CPU, 4 GB; team-b-cq: 1 CPU, 4 GB)
- Cohort configurations for resource borrowing tests (within 2 CPU, 8 GB total limit)
- Preemption policy configurations (enabled/disabled variants)
- **Setup/Teardown**: Each test scenario performs Kueue operator installation, ClusterQueue/LocalQueue creation, and cleanup to revert cluster to original state

### 9.2 Configuration

- Kueue Operator version configuration via environment variable or config file (hard-coded version for initial implementation)
- Namespace labels: team=team-a, kueue.openshift.io/managed=true, evalhub.trustyai.opendatahub.io/tenant=true
- LocalQueue to ClusterQueue mappings
- Resource quota limits (CPU, memory) assigned to ClusterQueues (constrained to 2 CPU, 8 GB total cluster capacity)
- Feature gates for Kueue preemption (disabled by default per best practice)
- EvalHub API configuration for queue integration
- Job priority settings (default: 0)
- Job resource requests configured to fit cluster capacity (e.g., 500m CPU, 1Gi memory per job)

### 9.3 Test Tools

- kubectl/oc CLI for Kueue resource inspection and namespace management
- curl/httpie for EvalHub API job submission and status queries
- kubectl get workload commands for detailed queue status
- Kubernetes event monitoring tools for tracking job admission and preemption events
- Resource monitoring tools for quota enforcement validation
- Log aggregation for debugging job failures and queue transitions

---

## 10. Appendix

### 10.1 Test Case Summary

| Category | Total | P0 | P1 | P2 |
| --- | --- | --- | --- | --- |
| API Integration (TC-API) | 5 | 4 | 1 | 0 |
| Queue Management (TC-QUEUE) | 3 | 2 | 1 | 0 |
| Resource Quota (TC-QUOTA) | 2 | 2 | 0 | 0 |
| Priority Scheduling (TC-PRIO) | 2 | 0 | 2 | 0 |
| Preemption (TC-PREEMPT) | 3 | 0 | 0 | 3 |
| Status Reporting (TC-STATUS) | 2 | 0 | 2 | 0 |
| Multi-Tenancy (TC-MULTI) | 1 | 0 | 0 | 1 |
| Negative Testing (TC-NEG) | 5 | 2 | 3 | 0 |
| End-to-End (TC-E2E) | 3 | 3 | 0 | 0 |
| Upgrade Testing (TC-UPGRADE) | 2 | 0 | 2 | 0 |
| **Total** | **28** | **13** | **11** | **4** |

### 10.2 Endpoint/Method Coverage

| Endpoint | Test Cases | Coverage |
| --- | --- | --- |
| POST /api/v1/evaluations/jobs | TC-API-001, TC-QUEUE-001, TC-QUEUE-002, TC-QUOTA-001, TC-NEG-001, TC-NEG-002, TC-E2E-001, TC-E2E-002, TC-E2E-003 | |
| GET /api/v1/evaluations/jobs/{id} | TC-API-002, TC-E2E-001, TC-E2E-002, TC-E2E-003, TC-NEG-005 | |
| GET /api/v1/evaluations/jobs | TC-API-003 | |
| DELETE /api/v1/evaluations/jobs/{id} | TC-API-004, TC-API-005, TC-E2E-002, TC-E2E-003 | |
| GET /api/v1/health | TC-UPGRADE-002 | |
| Kubernetes Workload Resource | TC-STATUS-001, TC-PREEMPT-001, TC-PREEMPT-002, TC-E2E-001 | |
| LocalQueue Resource | TC-QUEUE-001, TC-QUEUE-002, TC-STATUS-002, TC-E2E-001, TC-E2E-002 | |
| ClusterQueue Resource | TC-QUOTA-001, TC-QUOTA-002, TC-MULTI-001, TC-E2E-001 | |
| Job Priority Field | TC-PRIO-001, TC-PRIO-002 | |
| Preemption Policy (withinClusterQueue) | TC-PREEMPT-001, TC-PREEMPT-003 | |
| Workload Status Conditions | TC-STATUS-001, TC-PREEMPT-001, TC-PREEMPT-002, TC-E2E-001 | |

### 10.3 Document Change Log

| Version | Date | Changes |
| --- | --- | --- |
| 1.0.0 | 2026-05-04 | Initial test plan |
| 1.1.0 | 2026-05-04 | Generated 28 test cases across 10 categories |

---

## End of Test Plan
