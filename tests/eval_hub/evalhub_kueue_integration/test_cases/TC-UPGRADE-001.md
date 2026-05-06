---
test_case_id: TC-UPGRADE-001
source_key: RHOAIENG-59092
priority: P1
status: Draft
automation_status: Not Started
last_updated: '2026-05-04'
upgrade_phase: both
---
# TC-UPGRADE-001: Job submission works with and without Kueue after EvalHub upgrade

**Objective**: Verify that EvalHub remains backwards compatible after upgrade — jobs submitted without queue specification work on both old and new versions, and Kueue-managed jobs work after upgrade.

**Preconditions**:
- EvalHub deployed and accessible
- Kueue Operator installed on the cluster
- ClusterQueue and LocalQueue configured

**Test Steps**:
1. **Pre-upgrade**: Submit a job without queue specification, verify HTTP 202 and job runs normally
2. **Pre-upgrade**: Submit a job with queue specification, verify it is queued and admitted
3. **Perform EvalHub upgrade** (version under test)
4. **Post-upgrade**: Submit a job without queue specification, verify HTTP 202 and job runs normally
5. **Post-upgrade**: Submit a job with queue specification, verify it is queued and admitted
6. **Post-upgrade**: Retrieve status of pre-upgrade jobs, verify they are still accessible
7. Teardown: Delete all jobs, LocalQueue, and ClusterQueue

**Expected Results**:
- Jobs without queue specification work identically before and after upgrade
- Jobs with queue specification work after upgrade
- Pre-upgrade job metadata is accessible after upgrade
- No data corruption or state loss during upgrade

**Notes**: To be filled later in the process.
