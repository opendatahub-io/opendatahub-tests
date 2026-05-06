---
test_case_id: TC-UPGRADE-002
source_key: RHOAIENG-59092
priority: P1
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
upgrade_phase: both
---
# TC-UPGRADE-002: EvalHub functions when Kueue is removed (rollback scenario)

**Objective**: Verify that EvalHub continues to function when Kueue is uninstalled or disabled, ensuring rollback safety.

**Preconditions**:
- EvalHub deployed and accessible
- Kueue Operator installed and configured
- At least one job previously submitted with queue specification

**Test Steps**:
1. **With Kueue**: Submit a job with queue specification, verify it is admitted and runs
2. **Remove Kueue**: Uninstall the Kueue Operator from the cluster
3. **Post-removal**: Submit a job without queue specification, verify HTTP 202 and job runs normally
4. **Post-removal**: Submit a job with queue specification, verify EvalHub handles the missing Kueue gracefully (expected: error response or job runs without queue management)
5. **Post-removal**: Verify EvalHub health check (GET /api/v1/health) returns healthy status
6. Teardown: Re-install Kueue if needed for other tests

**Expected Results**:
- Jobs without queue specification continue to work after Kueue removal
- Jobs with queue specification either return a clear error or run without queue management
- EvalHub does not crash or enter an error state when Kueue is not present
- Health endpoint returns healthy status

**Notes**: To be filled later in the process.
