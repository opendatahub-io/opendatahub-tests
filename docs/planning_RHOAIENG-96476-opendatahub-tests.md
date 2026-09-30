# AI Coding Session Prompt: RHOAIENG-96476 in `opendatahub-tests`

## Session Role

You are an AI coding agent working in:

`/Users/chambrid/Code/odh/opendatahub-tests`

Implement the integrated release-contract tests for Jira issue RHOAIENG-96476. This repository is the primary home for deterministic fixtures, raw PromQL/Thanos/API assertions, independent personas, namespace isolation, MaaS/GPUaaS checks, and machine-readable evidence.

Use a test-first workflow. Inspect existing pytest fixtures, clients, user/session helpers, model-serving fixtures, and observability tests before editing. State the proposed test package and changed files before implementation. Do not create a second fixture framework or duplicate existing MaaS and model-serving utilities without a concrete gap.

## Issue Objective

Prove that seeded observability telemetry is visible to the correct persona and namespace through the routes used by the release dashboards. The tests must distinguish:

- Missing hardware from missing telemetry.
- Missing telemetry from a bad PromQL query.
- A successful empty result from an HTTP or Prometheus error.
- Frontend dashboard filtering from actual namespace authorization.
- Token usage, token breakdown, and showback capabilities.
- A product failure from an environment-blocked release gate.

The repository owns the integrated contract. Operator-generated resource checks remain in `odh-observability`, and live browser rendering remains in `odh-dashboard`.

## Repository Evidence

The checked-out repository already provides useful building blocks:

- `tests/conftest.py` provides shared clients and test context.
- `tests/ai_gateway/models_as_a_service/conftest.py` and the MaaS fixture tree provide tenant, subscription, model, API key, user, and cleanup patterns.
- `tests/ai_gateway/models_as_a_service/observability/` validates MaaS ServiceMonitors, scrape results, and usage-logging reconciliation.
- `tests/model_serving/model_server/kserve/observability/` validates non-admin metric access and source model metrics.
- `tests/fixtures/inference.py` contains model-serving fixture patterns.
- `utilities/monitoring.py` provides basic Prometheus polling and metric-value validation, but it does not retain the full HTTP/Prometheus response contract required here.
- `utilities/user_utils.py` provides user/session patterns that should be reused for independent persona authentication.

Create a new focused package under the repository's established observability test area after inspection. Prefer a package such as `tests/observability/` or another existing component-approved location rather than forcing all cases into `tests/ai_gateway/models_as_a_service/observability/`. Keep MaaS-specific helpers reusable and do not move unrelated tests.

## Scope

Implement the following in this repository:

1. Release and environment preflight with explicit `ready`, `blocked`, and `failed` outcomes.
2. Two-namespace deterministic fixtures for model, GPU, LLM, GPUaaS, and MaaS cases as supported by the release.
3. Independent cluster-admin, namespace-admin, namespace-contributor, and regular-user personas.
4. A raw Prometheus/Thanos/API query helper that retains status, errors, labels, and result type.
5. Admin data, GPU, GPUaaS regression, MaaS usage/token, and namespace-isolation tests.
6. Release capability checks for token usage, token breakdown, and showback.
7. Sanitized machine-readable evidence and human-readable failure output.
8. Safe cleanup that works after partial failures.

Do not implement the following here:

- Operator template or generated-resource tests owned by `odh-observability`.
- Perses dashboard rendering or browser selectors owned by `odh-dashboard`.
- Product feature, RBAC, proxy, datasource, or metric changes to make an assertion pass.
- Admin-token tests that merely change a namespace query parameter.
- Hardcoded credentials, tokens, API keys, tenant secrets, or cluster-specific names.

## Shared Contract Required Before Coding

Create or consume one reviewed, versioned contract input for the release run. Do not silently duplicate stale dashboard queries. Every capability/panel record should identify:

- Release stage: EA1, EA2, or GA.
- Product/repository/image versions under test.
- Dashboard and panel identifier.
- Datasource and exact route.
- Exact PromQL expression, including namespace and model variables.
- Expected HTTP status and Prometheus `status`.
- Expected result type and minimum series/labels.
- Whether an empty result is valid and what UI state it maps to.
- Whether the capability is `shipped`, `not-shipped`, or `environment-blocked`.

The initial matrix must include:

- Cluster dashboard system health, deployed models, GPU utilization, GPU utilization by project, CPU, memory, and network.
- Models dashboard model-deployment variable, model table, request queue, replicas, latency, TTFT, token generation, throughput, and response distribution.
- Accelerator source and recording series including `DCGM_FI_DEV_GPU_UTIL`, `accelerator_gpu_utilization`, `accelerator_memory_used_bytes`, temperature, and power where shipped.
- Inference series including `kserve_vllm:*` and release-supported request/token metrics.
- MaaS series including `authorized_hits`, `authorized_calls`, `limited_calls`, and only the labels shipped by the release.
- Namespace proxy, data-science Thanos, cluster-wide Thanos, and tenancy endpoint paths.
- Token breakdown and showback capability status.

Before writing negative authorization assertions, resolve the exact unauthorized response contract: 403, 404, successful empty result, or filtered result. Record that decision in the contract rather than accepting every non-success response.

## Required Implementation Sequence

### 1. Establish the test package and baseline

Before editing:

1. Read the relevant `conftest.py` files, observability tests, user utilities, model-serving fixtures, and repository contribution guidance.
2. Run collection for the existing observability packages and record baseline failures.
3. Identify which fixtures already create users, sessions, model deployments, GPU workloads, MaaS tenants, subscriptions, and API keys.
4. Choose the smallest new package boundary that can cover the integrated contract without duplicating those fixtures.
5. Add unit tests for new pure helpers before adding cluster tests.

Use the repository's markers, fixture scopes, retry/timeouts, cleanup registration, and logging conventions. Add a new marker only after confirming the marker configuration.

### 2. Implement release and environment preflight

Run preflight before creating or mutating test resources. Record sanitized results in the run evidence directory.

The preflight must resolve or verify:

- Release stage and product/component versions.
- Observability feature flag and expected dashboard resources.
- Perses CRDs, datasource resources, Prometheus/Thanos routes, and expected namespaces.
- GPU operator/DCGM exporter and at least one schedulable GPU for GPU cases.
- Ability to schedule the configured GPU workload and discover its metrics target.
- MaaS tenant, subscription, model, API-key/token path, gateway policy, usage logging, and Limitador target for MaaS cases.
- Independent authentication for every configured persona.
- Baseline SubjectAccessReview permissions for each persona.

Use these dispositions consistently:

- `ready`: all prerequisites for the selected contract are present.
- `blocked`: an external prerequisite prevents a meaningful test, such as no GPU capacity or an unavailable release component.
- `failed`: the environment claims to provide a prerequisite but does not satisfy it.

Do not turn a blocked release capability into a pass. Do not use a generic `pytest.skip` that hides whether the product or environment is responsible; use the repository's explicit blocked reporting mechanism or add one.

### 3. Build deterministic two-namespace fixtures

Use unique test-owned namespace names generated from the run. Create namespace A and namespace B so leakage cannot be hidden by a single fixture.

The fixture lifecycle must:

1. Register cleanup before resource creation.
2. Create both namespaces with the labels required by dashboard discovery and monitoring.
3. Create a deterministic model deployment in each namespace with distinct names such as `model-a` and `model-b`.
4. Create a GPU-backed workload in namespace A when the selected contract requires GPU data.
5. Wait for scheduling, readiness, target discovery, and source metric availability.
6. Send a fixed inference request and generate enough traffic to activate lazy vLLM metrics.
7. Use existing MaaS tenant/subscription/model/API-key fixtures where possible.
8. Generate fixed successful and rate-limited requests for MaaS, retaining only non-secret request metadata and expected totals.
9. Use the production-style GPUaaS route and resource configuration required by the release. Do not replace it with a local Prometheus shortcut.
10. Record resource names, UIDs, namespace, model name, request count, expected labels, and readiness timestamps.
11. Verify source metrics directly before any dashboard/UI test consumes the fixture.
12. Clean up only test-owned resources, including after partial failures, without deleting pre-existing tenant or operator configuration.

A missing source metric must fail the telemetry stage. It must not be reported only as a dashboard empty state.

### 4. Create real independent personas

Required personas:

- Cluster administrator: allowed to view the cluster-wide and in-scope namespace data.
- Namespace administrator: allowed in namespace A but not namespace B.
- Namespace contributor: allowed in namespace A but not namespace B.
- Regular user: allowed for the intended user workflow and namespace A without cluster-scoped permissions.

For every persona:

1. Authenticate with separate credentials.
2. Capture the principal from the API/session response or `oc whoami` equivalent.
3. Record groups, role bindings, and allowed namespaces without logging secrets.
4. Run explicit SAR checks for metrics resources and dashboard/API routes.
5. Run the query set with the persona's own bearer token.
6. Restore the original session even after a failure.

Fail if a restricted persona accidentally authenticates as the cluster administrator. A failed login followed by an inherited admin session is not a valid restricted-persona result.

### 5. Add a raw Prometheus/Thanos evidence helper

Extend `utilities/monitoring.py` only if the existing utility conventions support the new API; otherwise add a focused package helper and unit-test it. The helper must preserve the full response instead of returning only the first value.

For each query, retain:

- Persona and authenticated principal.
- Requested namespace and fixture namespace.
- Datasource/route name and sanitized URL path.
- HTTP method, status, and response time.
- URL/query parameters, including the tenancy namespace parameter.
- PromQL expression and time range.
- Prometheus `status`, `errorType`, `error`, `warnings`, and `data.resultType`.
- Normalized series labels and values with timestamps rounded for stable comparison.
- Expected disposition: pass, fail, blocked, or unavailable.

Test the routes used by Perses, not only a convenient direct service:

- Cluster-wide Thanos: authorized success and no known production-style 400.
- Namespace proxy: authorization and query label injection.
- Tenancy endpoint: required namespace parameter and no foreign namespace series.
- Data-science Prometheus path: expected in-scope workload series.

Assertions must inspect both HTTP/Prometheus response fields and returned series. HTTP 200 alone is not evidence of a valid query.

### 6. Implement acceptance test groups

Use Given-When-Then docstrings and the repository's component/release markers.

#### Admin data contract

With the complete deterministic fixture and cluster-admin persona:

- Every shipped contract query returns HTTP 2xx and Prometheus `status=success`.
- Required panels return non-empty series or the explicitly contracted zero state.
- Returned labels contain the expected model, namespace, GPU, MaaS, or token dimensions.
- Unexpected 400 responses, query errors, or warnings that alter displayed results fail the test.

#### GPU population and active utilization

With a scheduled, recently used GPU workload:

- DCGM source metrics identify the expected node/pod.
- `accelerator_gpu_utilization` contains expected namespace/pod labels.
- Utilization is active and non-zero unless the release contract says otherwise.
- Memory conversion and accelerator units match the contract.
- Cluster and Models queries return a series instead of `No data`.

If source telemetry is absent, fail at the telemetry stage with the source failure category.

#### GPUaaS regression

With production-style GPUaaS data and the authorized persona:

- The exact dashboard route/query returns success.
- The response is not the known 400 failure.
- Returned series contain expected GPUaaS labels.
- Sanitized route and response evidence are retained so a 400 is distinguishable from a successful empty result.

#### MaaS usage and token contract

With deterministic successful and rate-limited requests, assert separately:

- Request/capacity metrics such as `authorized_calls`.
- Token usage metrics such as `authorized_hits`.
- `limited_calls` where shipped.
- Model, subscription, user, organization, and cost-center labels only when shipped.
- Token usage is not mislabeled as monetary cost.
- Prompt/input and completion/output token breakdown only when explicitly shipped.
- An unshipped breakdown produces an explicit unavailable disposition and does not pass as a total-token result.
- Showback is tested independently from token usage and is unavailable unless explicitly shipped.

#### Namespace authorization and isolation

For each namespace-scoped persona, run identical queries for namespace A, namespace B, and an omitted or tampered namespace selector:

- Authorized namespace returns the expected success and only authorized series.
- Unauthorized namespace returns exactly the reviewed contract response.
- Tampered queries never return a foreign namespace series.
- Cluster-wide data is available only to the authorized persona and follows the documented denial or filtering behavior for restricted personas.

Compare returned label sets against the persona's authorization scope. A successful response containing a foreign namespace series is a security failure.

### 7. Implement capability declarations and evidence

For every EA1, EA2, and GA run, declare each capability as:

- `shipped`: positive data and UI assertions are mandatory.
- `not-shipped`: tests verify explicit absence/unavailability and documentation must not claim support.
- `environment-blocked`: the release supports it but the environment lacks a dependency; this is not a product pass.

Separate request counts, token usage, token breakdown, and showback. A total-token result cannot satisfy a token-breakdown contract, and usage cannot satisfy a cost/showback contract.

Emit a machine-readable evidence record and human-readable logs containing:

- Jira key and test identifier.
- Release stage and component versions.
- Cluster/run identifier.
- Persona, principal, groups, and namespace scope.
- Fixture resources and UIDs.
- Dashboard, panel, datasource, endpoint, and query identifier.
- PromQL, HTTP status, Prometheus error fields, result type, and normalized series.
- Disposition and failure category.
- Sanitized UI handoff metadata when the dashboard suite consumes the fixture.

Redact bearer tokens, passwords, cookies, API keys, secret data, and secret-bearing URLs. Store artifacts in the CI-provided location with the retention required for release review.

### 8. Define the release handoff

Document the interface consumed by the `odh-dashboard` live suite:

- Fixture namespace and model names.
- Persona identity variables without credential values.
- Dashboard and panel contract version.
- Evidence directory and artifact naming.
- Readiness signal that means source metrics are available.
- Cleanup owner and execution order.

The release runner should execute this repository after operator/resource checks and before live Cypress:

1. Preflight and capability declaration.
2. Fixture creation and source telemetry validation.
3. Persona PromQL/API and namespace-isolation tests.
4. Handoff of fixture/evidence metadata to the dashboard suite.
5. Cleanup after UI execution or explicit failure handling.

## Test Design Requirements

- Write tests for pure helpers before implementing them.
- Reuse existing `ocp_resources`, model-serving, MaaS, and user-session fixtures.
- Do not create a new admin shortcut for restricted-user tests.
- Use explicit retry/timeouts for eventual metric appearance; do not use arbitrary sleeps.
- Generate unique resource names and register cleanup before creation.
- Treat no GPU, no route, or unshipped capability as a declared disposition, not as a hidden skip.
- Keep raw query evidence stable by normalizing timestamps and sorting labels.
- Never place secrets in source, fixtures, logs, screenshots, or artifact JSON.

## Acceptance Criteria

The work in this repository is complete only when:

- A deterministic two-namespace fixture can produce source telemetry before dashboard assertions run.
- All four required personas use independent authentication and have recorded permission baselines.
- Raw queries validate status, errors, result type, labels, values, and namespace scope.
- GPU, GPUaaS, MaaS usage, token, token-breakdown, and showback capabilities are tested independently.
- Unauthorized and tampered queries cannot return foreign namespace series.
- Missing hardware and missing telemetry are distinguishable from dashboard/API failures.
- Unsupported capabilities produce explicit unavailable results.
- Evidence is sanitized, machine-readable, and consumable by the release job and Cypress handoff.
- Existing MaaS and model-serving observability tests remain green.
- Cleanup is verified after successful and failed runs.

## Validation Commands

Run focused collection and tests first, then repository-wide checks. Confirm marker names and package scripts before using them:

```bash
uv run pytest --collect-only
uv run pytest tests/observability/
uv run pytest tests/ai_gateway/models_as_a_service/observability/
uv run pytest tests/model_serving/model_server/kserve/observability/
pre-commit run --all-files
tox
```

If the new package is placed elsewhere, replace the focused path with the actual path and report it. Cluster tests require explicit test variables and an isolated environment. Do not run them with credentials embedded in the command line or source.

## Final Coding-Session Report

At the end of the session, report:

1. Files changed and why each belongs in this repository.
2. Fixture and persona design, including cleanup behavior.
3. Contract tests added and the release capability each proves.
4. Commands run and results, including blocked prerequisites.
5. Evidence/artifact locations with confirmation that secrets were redacted.
6. The exact fixture and evidence handoff required by `odh-dashboard`.
7. Any remaining decisions that must be resolved by observability, dashboard, MaaS, or QE owners.
