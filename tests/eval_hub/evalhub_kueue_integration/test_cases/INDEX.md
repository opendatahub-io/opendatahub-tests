# Test Case Index — EvalHub Kueue Integration

**Test Plan**: [TestPlan.md](../TestPlan.md)
**Source**: [RHOAIENG-59092](https://issues.redhat.com/browse/RHOAIENG-59092)

## Quick Stats

| Total | P0 | P1 | P2 |
|-------|----|----|-----|
| 28 | 13 | 11 | 4 |

---

## API Integration Testing

| Test Case ID | Title | Priority |
|--------------|-------|----------|
| [TC-API-001](TC-API-001.md) | Submit evaluation job with Kueue queue specification | P0 |
| [TC-API-002](TC-API-002.md) | Retrieve job status for Kueue-managed job | P0 |
| [TC-API-003](TC-API-003.md) | List evaluation jobs with status filter | P1 |
| [TC-API-004](TC-API-004.md) | Cancel evaluation job via soft delete | P0 |
| [TC-API-005](TC-API-005.md) | Cancel evaluation job via hard delete | P0 |

## Queue Management Testing

| Test Case ID | Title | Priority |
|--------------|-------|----------|
| [TC-QUEUE-001](TC-QUEUE-001.md) | Job admitted when LocalQueue has available quota | P0 |
| [TC-QUEUE-002](TC-QUEUE-002.md) | Job remains pending when ClusterQueue quota is exhausted | P0 |
| [TC-QUEUE-003](TC-QUEUE-003.md) | Pending job auto-admitted after resources freed | P1 |

## Resource Quota Enforcement Testing

| Test Case ID | Title | Priority |
|--------------|-------|----------|
| [TC-QUOTA-001](TC-QUOTA-001.md) | Multiple jobs admitted within ClusterQueue quota limits | P0 |
| [TC-QUOTA-002](TC-QUOTA-002.md) | Job exceeding ClusterQueue quota is queued | P0 |

## Priority-Based Scheduling Testing

| Test Case ID | Title | Priority |
|--------------|-------|----------|
| [TC-PRIO-001](TC-PRIO-001.md) | Default priority 0 assigned when not specified | P1 |
| [TC-PRIO-002](TC-PRIO-002.md) | Higher priority job admitted before lower priority when quota available | P1 |

## Preemption Scenario Testing

| Test Case ID | Title | Priority |
|--------------|-------|----------|
| [TC-PREEMPT-001](TC-PREEMPT-001.md) | Higher priority job preempts lower priority when preemption enabled | P2 |
| [TC-PREEMPT-002](TC-PREEMPT-002.md) | Preempted job restarts from beginning when resumed | P2 |
| [TC-PREEMPT-003](TC-PREEMPT-003.md) | No preemption when withinClusterQueue is set to Never | P2 |

## Status Reporting Testing

| Test Case ID | Title | Priority |
|--------------|-------|----------|
| [TC-STATUS-001](TC-STATUS-001.md) | Workload resource reflects detailed queue status conditions | P1 |
| [TC-STATUS-002](TC-STATUS-002.md) | LocalQueue status reflects pending and admitted job counts | P1 |

## Multi-Tenancy Testing

| Test Case ID | Title | Priority |
|--------------|-------|----------|
| [TC-MULTI-001](TC-MULTI-001.md) | Jobs in different namespaces use separate ClusterQueues | P2 |

## Negative Testing

| Test Case ID | Title | Priority |
|--------------|-------|----------|
| [TC-NEG-001](TC-NEG-001.md) | Submit job with non-existent queue name returns error | P0 |
| [TC-NEG-002](TC-NEG-002.md) | Submit job without queue specification (backwards compatibility) | P1 |
| [TC-NEG-003](TC-NEG-003.md) | Unauthorized request returns 401 | P0 |
| [TC-NEG-004](TC-NEG-004.md) | Forbidden request returns 403 | P1 |
| [TC-NEG-005](TC-NEG-005.md) | Get non-existent job returns 404 | P1 |

## End-to-End Scenario Testing

| Test Case ID | Title | Priority |
|--------------|-------|----------|
| [TC-E2E-001](TC-E2E-001.md) | Complete job lifecycle — submit, queue, admit, run, complete | P0 |
| [TC-E2E-002](TC-E2E-002.md) | Job queuing under resource pressure — submit, wait, admit, complete | P0 |
| [TC-E2E-003](TC-E2E-003.md) | Job cancellation during execution — submit, admit, run, cancel | P0 |

## Upgrade Testing

| Test Case ID | Title | Priority |
|--------------|-------|----------|
| [TC-UPGRADE-001](TC-UPGRADE-001.md) | Job submission works with and without Kueue after EvalHub upgrade | P1 |
| [TC-UPGRADE-002](TC-UPGRADE-002.md) | EvalHub functions when Kueue is removed (rollback scenario) | P1 |
