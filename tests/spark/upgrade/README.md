# Spark upgrade tests

The `test_spark_pi_execution` smoke test runs on fresh installs and in both upgrade
phases. Before upgrade it preserves the completed Pi application for the reuse
test; after upgrade it creates a fresh application. On install runs it cleans up
the workload namespace and restores the original Spark Operator configuration.
The continuity tests additionally keep an executor task in progress across the
externally performed upgrade.

Run the install smoke test (no upgrade flags):

```bash
uv run pytest tests/spark/test_spark.py -m smoke -v
```

Run the complete upgrade phases against `tests/spark/` so they include the shared
smoke test in `tests/spark/test_spark.py`:

```bash
uv run pytest tests/spark/ --pre-upgrade -v
# Perform the operator upgrade.
uv run pytest tests/spark/ --post-upgrade -v
```

Run the continuity scenario alone:

```bash
uv run pytest tests/spark/upgrade/test_upgrade.py --pre-upgrade -k test_spark_job_in_progress -v
# Perform the OpenShift AI / Spark Operator upgrade, keeping Spark Managed.
uv run pytest tests/spark/upgrade/test_upgrade.py --post-upgrade -k test_spark_job_in_progress -v
```

Use the directory-wide commands above to include the Pi scenarios. Do not use
`--delete-pre-upgrade-resources` for an actual upgrade: the namespace, application,
control ConfigMap, and workload permissions must remain available between phases.

The pre-upgrade test waits for an executor to log that its task started, then saves
the running application's UID and generation plus both pods' UIDs and restart
counts. The task waits on a mounted ConfigMap release file, with a 4-hour limit.
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
namespace and waits for deletion before setting Spark Operator to `Removed` and
waiting for its removal condition. This applies to individual post-upgrade tests
as well as the full suite, including test failures. If namespace deletion fails,
operator removal is not attempted, so the operator remains available to finish
resource cleanup. Finish the post-upgrade phase within 4 hours of task startup.
