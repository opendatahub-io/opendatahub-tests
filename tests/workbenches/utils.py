#
# Copyright Skodjob authors.
# License: Apache License 2.0 (see the file LICENSE or http://apache.org/licenses/LICENSE-2.0.html).
#
from __future__ import annotations

import logging
import time
import traceback
from contextlib import contextmanager
from typing import Any

from typing import Callable, Generator

from kubernetes.dynamic import DynamicClient, ResourceField, Resource

from ocp_resources.pod import Pod


class TestFrameConstants:
    GLOBAL_POLL_INTERVAL_MEDIUM = 10


class PodUtils:
    READINESS_TIMEOUT = 10 * 60

    @staticmethod
    def wait_for_pods_ready(
        client: DynamicClient, namespace_name: str, label_selector: str, expect_pods_count: int
    ) -> None:
        """Wait for all pods in namespace to be ready
        :param client:
        :param namespace_name: name of the namespace
        :param label_selector:
        :param expect_pods_count:
        """

        # it's a dynamic client with the `resource` parameter already filled in
        class ResourceType(Resource, DynamicClient):
            pass

        resource: ResourceType = client.resources.get(
            kind=Pod.kind,
            api_version=Pod.api_version,
        )

        def ready() -> bool:
            pods = resource.get(namespace=namespace_name, label_selector=label_selector).items
            if not pods and expect_pods_count == 0:
                logging.debug("All expected Pods %s in Namespace %s are ready", label_selector, namespace_name)
                return True
            if not pods:
                logging.debug("Pods matching %s/%s are not ready", namespace_name, label_selector)
                return False
            if len(pods) != expect_pods_count:
                logging.debug("Expected Pods %s/%s are not ready", namespace_name, label_selector)
                return False
            pod: ResourceField
            for pod in pods:
                if not Readiness.is_pod_ready(pod) and not Readiness.is_pod_succeeded(pod):
                    logging.debug("Pod is not ready: %s/%s", namespace_name, pod.metadata.name)
                    return False
                else:
                    # check all containers in pods are ready
                    for cs in pod.status.containerStatuses:
                        if not (cs.ready or cs.state.get("terminated", {}).get("reason", "") == "Completed"):
                            logging.debug(
                                f"Container {cs.getName()} of Pod {namespace_name}/{pod.metadata.name} not ready"
                            )
                            return False
            logging.info("Pods matching %s/%s are ready", namespace_name, label_selector)
            return True

        Wait.until(
            description=f"readiness of all Pods matching {label_selector} in Namespace {namespace_name}",
            poll_interval=TestFrameConstants.GLOBAL_POLL_INTERVAL_MEDIUM,
            timeout=PodUtils.READINESS_TIMEOUT,
            ready=ready,
        )


class Wait:
    @staticmethod
    def until(
        description: str,
        poll_interval: float,
        timeout: float,
        ready: Callable[[], bool],
        on_timeout: Callable[[], None] | None = None,
    ) -> None:
        """For every poll (happening once each {@code pollIntervalMs}) checks if supplier {@code ready} is true.

        If yes, the wait is closed. Otherwise, waits another {@code pollIntervalMs} and tries again.
        Once the wait timeout (specified by {@code timeoutMs} is reached and supplier wasn't true until that time,
        runs the {@code onTimeout} (f.e. print of logs, showing the actual value that was checked inside {@code ready}),
        and finally throws {@link WaitException}.
        @param description    information about on what we are waiting
        @param pollIntervalMs poll interval in milliseconds
        @param timeoutMs      timeout specified in milliseconds
        @param ready          {@link BooleanSupplier} containing code, which should be executed each poll,
                               verifying readiness of the particular thing
        @param onTimeout      {@link Runnable} executed once timeout is reached and
                               before the {@link WaitException} is thrown."""
        logging.info("Waiting for: %s", description)
        deadline = time.monotonic() + timeout

        exception_message: str | None = None
        previous_exception_message: str | None = None

        # in case we are polling every 1s, we want to print exception after x tries, not on the first try
        # for minutes poll interval will 2 be enough
        exception_appearance_count: int = 2 if (poll_interval // 60) > 0 else max(int(timeout // poll_interval // 4), 2)
        exception_count: int = 0
        new_exception_appearance: int = 0

        stack_trace_error: str | None = None

        while True:
            try:
                result: bool = ready()
            except KeyboardInterrupt:
                raise  # quick exit if the user gets tired of waiting
            except Exception as e:
                exception_message = str(e)

                exception_count += 1
                new_exception_appearance += 1
                if (
                    exception_count == exception_appearance_count
                    and exception_message is not None
                    and exception_message == previous_exception_message
                ):
                    logging.info(f"While waiting for: {description} exception occurred: {exception_message}")
                    # log the stacktrace
                    stack_trace_error = traceback.format_exc()
                elif (
                    exception_message is not None
                    and exception_message != previous_exception_message
                    and new_exception_appearance == 2
                ):
                    previous_exception_message = exception_message

                result = False

            time_left: float = deadline - time.monotonic()
            if result:
                return
            if time_left <= 0:
                if exception_count > 1:
                    logging.error("Exception waiting for: %s, %s", description, exception_message)

                    if stack_trace_error is not None:
                        # printing handled stacktrace
                        logging.error(stack_trace_error)
                if on_timeout is not None:
                    on_timeout()
                wait_exception: WaitException = WaitException(f"Timeout after {timeout} s waiting for {description}")
                logging.error(wait_exception)
                raise wait_exception

            sleep_time: float = min(poll_interval, time_left)
            time.sleep(sleep_time)  # noqa: FCN001


class WaitException(Exception):
    pass


class Readiness:
    @staticmethod
    def is_pod_ready(pod: ResourceField) -> bool:
        Utils.check_not_none(value=pod, message="Pod can't be null.")

        condition = Pod.Condition.READY
        status = Pod.Condition.Status.TRUE
        for cond in pod.get("status", {}).get("conditions", []):
            if cond["type"] == condition and cond["status"].casefold() == status.casefold():
                return True
        return False

    @staticmethod
    def is_pod_succeeded(pod: ResourceField) -> bool:
        Utils.check_not_none(value=pod, message="Pod can't be null.")
        return pod.status is not None and "Succeeded" == pod.status.phase


class Utils:
    @staticmethod
    def check_not_none(value: Any, message: str) -> None:
        if value is None:
            raise ValueError(message)


@contextmanager
def step(description: str) -> Generator[None, None, None]:
    yield
