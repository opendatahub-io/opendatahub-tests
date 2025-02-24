from typing import Any

from ocp_resources.prometheus import Prometheus
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

LOGGER = get_logger(name=__name__)


def validate_metrics_value(
    prometheus: Prometheus, metrics_query: str, expected_value: Any, timeout: int = 60 * 4
) -> None:
    """
    Validate metrics value against expected value

    Args:
        prometheus (Prometheus): Prometheus object
        metrics_query (str): Metrics query string
        expected_value (Any): Expected value
        timeout (int): Timeout in seconds

    Raises:
        TimeoutExpiredError: raised when expected conditions are not met within the timeframe

    """
    sample = None
    try:
        for sample in TimeoutSampler(
            wait_timeout=timeout,
            sleep=15,
            func=get_metrics_value,
            prometheus=prometheus,
            metrics_query=metrics_query,
        ):
            if sample:
                LOGGER.info(f"metric: {metrics_query} value is: {sample}, the expected value is {expected_value}")
                if sample == expected_value:
                    LOGGER.info("Metrics value matches the expected value!")
                    return
    except TimeoutExpiredError:
        LOGGER.info(f"Metrics value: {sample}, expected: {expected_value}")
        raise


def get_metrics_value(prometheus: Prometheus, metrics_query: str) -> Any:
    """
    Get metrics value from prometheus

    Args:
        prometheus (Prometheus): Prometheus object
        metrics_query (str): Metrics query string

    Returns:
        Any: Metrics value

    """
    metric_results = prometheus.query_sampler(query=metrics_query)
    if metric_values_list := [value for metric_val in metric_results for value in metric_val.get("value")]:
        return metric_values_list[1]
