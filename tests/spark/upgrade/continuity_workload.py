"""Executor task held across an upgrade, released through a mounted ConfigMap."""

import time
from pathlib import Path


def upgrade_task(value: int) -> int:
    """Keep one task active for up to 4 hours while waiting for post-upgrade release.

    Args:
        value: Input whose square is returned after release.

    Returns:
        The squared input.

    Raises:
        TimeoutError: If the post-upgrade test does not release the task within 4 hours.
    """
    print("UPGRADE_TASK_STARTED", flush=True)
    for _ in range(2880):
        if Path("/opt/spark/upgrade/release").read_text().strip() == "true":
            return value * value
        time.sleep(5)
    raise TimeoutError("Post-upgrade release was not received within 4 hours")


if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("upgrade-continuity").getOrCreate()
    try:
        result = spark.sparkContext.parallelize([7], 1).map(upgrade_task).collect()
        assert result == [49], f"Unexpected workload result: {result}"
        print("UPGRADE_TASK_COMPLETED: 49", flush=True)
    finally:
        spark.stop()
