from collections.abc import Callable
from typing import Any

from ocp_resources.prometheus import Prometheus
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

LOGGER = get_logger(name=__name__)


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


def get_metric_label(
    prometheus: Prometheus,
    metrics_query: str,
    label_name: str,
) -> Any:
    """
    Get the value of a specific label from the first matching metric.

    Args:
        prometheus (Prometheus): Prometheus object
        metrics_query (str): Metrics query string
        label_name (str): Label to retrieve

    Returns:
        Any: Value of the requested label, or None if not found
    """
    metric_results = prometheus.query_sampler(query=metrics_query)
    LOGGER.info(f"Fields: {metric_results}")

    if metric_results:
        # Assume we care about the first result
        return metric_results[0]["metric"].get(label_name)

    return None


def validate_metrics_field(
    prometheus: Prometheus,
    metrics_query: str,
    expected_value: Any,
    field_getter: Callable[..., Any] = get_metrics_value,
    timeout: int = 60 * 4,
    greater_than: bool = False,
) -> None:
    """
    Validate any metric field or label using a custom getter function.
    Defaults to checking the metric's value if no getter is provided.

    Args:
        prometheus (Prometheus): Prometheus object
        metrics_query (str): Metrics query string
        expected_value (Any): Expected value
        field_getter (Callable): Function to extract the desired field/label/value
        timeout (int): Timeout in seconds
        greater_than (bool): Whether to check if the metric is greater than the expected value

    Raises:
        TimeoutExpiredError: If expected value isn't met within the timeout
    """
    try:
        for sample in TimeoutSampler(
            wait_timeout=timeout,
            sleep=15,
            func=field_getter,
            prometheus=prometheus,
            metrics_query=metrics_query,
        ):
            if greater_than:
                if sample is not None and float(sample) >= float(expected_value):
                    LOGGER.info(f"Metric field {sample} is greater than or equal to expected value {expected_value}!")
                    return
            else:
                if sample == expected_value:
                    LOGGER.info("Metric field matches the expected value!")
                    return
            LOGGER.info(f"Current value: {sample}, waiting for: {expected_value}")
    except TimeoutExpiredError:
        LOGGER.error(f"Timed out. Last value: {sample}, expected: {expected_value}")
        raise
