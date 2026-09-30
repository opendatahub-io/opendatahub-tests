"""Reusable deterministic fixture helpers for integrated observability tests."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tests.observability.query import RawQueryResult

if TYPE_CHECKING:
    from kubernetes.dynamic import DynamicClient
    from ocp_resources.inference_service import InferenceService
    from ocp_resources.namespace import Namespace


@dataclass(frozen=True)
class FixtureResource:
    """Sanitized resource metadata for release evidence."""

    kind: str
    name: str
    namespace: str
    uid: str | None

    def to_dict(self) -> dict[str, str | None]:
        """Return resource metadata without the resource body."""
        return {
            "kind": self.kind,
            "name": self.name,
            "namespace": self.namespace,
            "uid": self.uid,
        }


@dataclass(frozen=True)
class NamespacePair:
    """The two isolated namespaces used by one release-contract run."""

    namespace_a: Namespace
    namespace_b: Namespace

    @property
    def names(self) -> tuple[str, str]:
        """Return namespace names in fixture order."""
        namespace_a = self.namespace_a.name
        namespace_b = self.namespace_b.name
        if not namespace_a or not namespace_b:
            raise ValueError("created observability namespaces must have names")
        return namespace_a, namespace_b


@dataclass(frozen=True)
class TrafficSummary:
    """Sanitized counts from deterministic successful and rate-limited requests."""

    total_requests: int
    successful_requests: int
    rate_limited_requests: int

    def to_dict(self) -> dict[str, int]:
        """Return request counts for evidence."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "rate_limited_requests": self.rate_limited_requests,
        }


@contextmanager
def create_namespace_pair(
    admin_client: DynamicClient,
    teardown: bool = True,
) -> Generator[NamespacePair]:
    """Create two run-unique dashboard-discoverable namespaces with safe unwind cleanup."""
    from ocp_resources.namespace import Namespace as OcpNamespace

    from utilities.constants import Labels
    from utilities.general import generate_random_name
    from utilities.infra import create_ns

    with ExitStack() as stack:
        labels = {Labels.OpenDataHub.DASHBOARD: "true"}
        namespace_a = stack.enter_context(
            cm=create_ns(
                admin_client=admin_client,
                name=generate_random_name(prefix="odh-observability-a"),
                labels=labels.copy(),
                teardown=teardown,
            )
        )
        namespace_b = stack.enter_context(
            cm=create_ns(
                admin_client=admin_client,
                name=generate_random_name(prefix="odh-observability-b"),
                labels=labels.copy(),
                teardown=teardown,
            )
        )
        if not isinstance(namespace_a, OcpNamespace) or not isinstance(namespace_b, OcpNamespace):
            raise TypeError("observability namespace creation must return Namespace resources")
        yield NamespacePair(namespace_a=namespace_a, namespace_b=namespace_b)


@contextmanager
def create_model_pair(
    admin_client: DynamicClient,
    namespaces: NamespacePair,
    runtime_template: str,
    model_format: str,
    storage_uri: str,
    deployment_mode: str | None = None,
    resources: dict[str, Any] | None = None,
    enable_auth: bool = False,
    external_route: bool = False,
    wait_for_predictor_pods: bool = False,
    teardown: bool = True,
) -> Generator[list[InferenceService]]:
    """Create model-a and model-b using the repository's serving runtime and ISVC helpers."""
    from ocp_resources.serving_runtime import ServingRuntime

    from utilities.constants import KServeDeploymentType
    from utilities.inference_utils import create_isvc
    from utilities.serving_runtime import ServingRuntimeFromTemplate

    selected_deployment_mode = deployment_mode or KServeDeploymentType.RAW_DEPLOYMENT
    with ExitStack() as stack:
        models: list[InferenceService] = []
        for namespace, model_name in zip((namespaces.namespace_a, namespaces.namespace_b), ("model-a", "model-b")):
            namespace_name = namespace.name
            if not namespace_name:
                raise ValueError("created observability namespace must have a name")
            runtime = stack.enter_context(
                cm=ServingRuntimeFromTemplate(
                    client=admin_client,
                    name=f"{model_name}-runtime",
                    namespace=namespace_name,
                    template_name=runtime_template,
                    multi_model=False,
                    deployment_type=selected_deployment_mode,
                    teardown=teardown,
                )
            )
            if not isinstance(runtime, ServingRuntime):
                raise TypeError("serving runtime helper did not return ServingRuntime")
            runtime_name = runtime.name
            if not runtime_name:
                raise ValueError("created serving runtime must have a name")
            model = stack.enter_context(
                cm=create_isvc(
                    client=admin_client,
                    name=model_name,
                    namespace=namespace_name,
                    model_format=model_format,
                    runtime=runtime_name,
                    storage_uri=storage_uri,
                    deployment_mode=selected_deployment_mode,
                    enable_auth=enable_auth,
                    external_route=external_route,
                    resources=resources,
                    wait_for_predictor_pods=wait_for_predictor_pods,
                    teardown=teardown,
                )
            )
            models.append(model)
        yield models


def resource_evidence(resource: object) -> FixtureResource:
    """Extract only stable identity metadata from an OCP resource wrapper."""
    instance = getattr(resource, "instance", None)
    metadata = getattr(instance, "metadata", None)
    return FixtureResource(
        kind=str(getattr(resource, "kind", type(resource).__name__)),
        name=str(getattr(resource, "name", "")),
        namespace=str(getattr(resource, "namespace", "")),
        uid=getattr(metadata, "uid", None),
    )


def source_metric_available(result: RawQueryResult) -> bool:
    """Return true only for a successful response containing source series."""
    return (
        result.http_status is not None
        and 200 <= result.http_status < 300
        and result.prometheus_status == "success"
        and bool(result.series)
    )


def wait_for_source_metric(
    query: Callable[[], RawQueryResult],
    timeout: int = 240,
    sleep: int = 5,
) -> RawQueryResult:
    """Poll source telemetry with explicit bounds before dashboard assertions consume it."""
    from timeout_sampler import TimeoutExpiredError, TimeoutSampler

    last_result: RawQueryResult | None = None
    try:
        for result in TimeoutSampler(wait_timeout=timeout, sleep=sleep, func=query):
            last_result = result
            if source_metric_available(result=result):
                return result
    except TimeoutExpiredError as error:
        raise AssertionError(f"source telemetry did not become available; last response: {last_result}") from error
    raise AssertionError(f"source telemetry did not become available; last response: {last_result}")


def summarize_traffic(statuses: Iterable[int]) -> TrafficSummary:
    """Count bounded successful and HTTP 429 MaaS requests without retaining credentials."""
    status_list = list(statuses)
    unexpected = [status for status in status_list if status not in {200, 429}]
    if unexpected:
        raise ValueError(f"unexpected MaaS request statuses: {unexpected}")
    return TrafficSummary(
        total_requests=len(status_list),
        successful_requests=status_list.count(200),
        rate_limited_requests=status_list.count(429),
    )
