# Spark upgrade tests

The Pi tests cover resubmission of a completed pre-upgrade workload and new job
execution. The continuity tests additionally keep an executor task in progress
across the externally performed upgrade.

Run the continuity scenario alone:

```bash
uv run pytest tests/spark/upgrade/test_upgrade.py --pre-upgrade -k test_spark_job_in_progress -v
# Perform the OpenShift AI / Spark Operator upgrade, keeping Spark Managed.
uv run pytest tests/spark/upgrade/test_upgrade.py --post-upgrade -k test_spark_job_in_progress -v
```

Omit `-k` to include the existing Pi scenarios. Do not use
`--delete-pre-upgrade-resources` for an actual upgrade: the namespace, application,
control ConfigMap, and workload permissions must remain available between phases.

The pre-upgrade test waits for an executor to log that its task started, then saves
the running application's UID and generation plus both pods' UIDs and restart
counts. The task waits on a mounted ConfigMap release file, with a 15-minute limit.
This uses PySpark from the existing pinned data-processing image. Driver and
executor pod templates mount the control ConfigMap directly, without relying on
webhook volume injection. Terminal startup failures report the driver logs immediately.

The post-upgrade test first requires that the original application and both pods
are still running with unchanged identities and restart counts. It then releases
the task without changing the SparkApplication spec, waits up to five minutes for
completion, and verifies the result in the original driver's log. Task retries and
speculative execution are disabled. A replaced or restarted execution cannot pass
as the original execution. The test harness does not perform or verify the product
upgrade itself.

Post-upgrade setup references the preserved RBAC and network policy instead of
deleting and recreating them. Post-upgrade session teardown deletes the test
namespace; the full suite also removes Spark through the existing new-job class
teardown. Finish the post-upgrade phase within 15 minutes of task startup.
