"""Kueue v1beta1 resource utilities for API migration testing.

This module provides Kueue resource classes using the v1beta1 API version
for testing CRD migration from v1beta1 to v1beta2 during RHOAI upgrades.
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import MissingRequiredArgumentError, NamespacedResource, Resource

LOGGER = structlog.get_logger(name=__name__)


class ResourceFlavorV1Beta1(Resource):
    """Kueue ResourceFlavor resource using v1beta1 API."""

    api_group: str = "kueue.x-k8s.io"
    api_version: str = "kueue.x-k8s.io/v1beta1"
    kind: str = "ResourceFlavor"

    def __init__(
        self,
        node_labels: dict[str, str] | None = None,
        tolerations: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            node_labels: Node labels for ResourceFlavor
            tolerations: Tolerations for ResourceFlavor
            kwargs: Keyword arguments to pass to the ResourceFlavor constructor
        """
        super().__init__(
            **kwargs,
        )
        self.node_labels = node_labels or {}
        self.tolerations = tolerations or []

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.node_labels:
                _spec["nodeLabels"] = self.node_labels
            if self.tolerations:
                _spec["tolerations"] = self.tolerations


class LocalQueueV1Beta1(NamespacedResource):
    """Kueue LocalQueue resource using v1beta1 API."""

    api_group: str = "kueue.x-k8s.io"
    api_version: str = "kueue.x-k8s.io/v1beta1"
    kind: str = "LocalQueue"

    def __init__(
        self,
        cluster_queue: str,
        **kwargs: Any,
    ):
        """
        Args:
            cluster_queue: Name of the cluster queue to use
            kwargs: Keyword arguments to pass to the LocalQueue constructor
        """
        super().__init__(
            **kwargs,
        )
        self.cluster_queue = cluster_queue

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            if not self.cluster_queue:
                raise MissingRequiredArgumentError(argument="cluster_queue")
            self.res["spec"] = {}
            _spec = self.res["spec"]
            _spec["clusterQueue"] = self.cluster_queue


class ClusterQueueV1Beta1(Resource):
    """Kueue ClusterQueue resource using v1beta1 API."""

    api_group: str = "kueue.x-k8s.io"
    api_version: str = "kueue.x-k8s.io/v1beta1"
    kind: str = "ClusterQueue"

    def __init__(
        self,
        namespace_selector: dict[str, Any] | None = None,
        resource_groups: list[dict[str, Any]] | None = None,
        stop_policy: str | None = None,
        cohort: str | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            namespace_selector: Namespace selector to use
            resource_groups: Resource groups to use
            stop_policy: Stop policy (Hold, None)
            cohort: Cohort name for resource sharing
            kwargs: Keyword arguments to pass to the ClusterQueue constructor
        """
        super().__init__(
            **kwargs,
        )
        self.namespace_selector = namespace_selector
        self.resource_groups = resource_groups
        self.stop_policy = stop_policy
        self.cohort = cohort

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            if not self.resource_groups:
                raise MissingRequiredArgumentError(argument="resource_groups")
            self.res["spec"] = {}
            _spec = self.res["spec"]

            if self.namespace_selector is not None:
                _spec["namespaceSelector"] = self.namespace_selector
            else:
                _spec["namespaceSelector"] = {}

            if self.resource_groups:
                _spec["resourceGroups"] = self.resource_groups

            if self.stop_policy:
                _spec["stopPolicy"] = self.stop_policy

            if self.cohort:
                _spec["cohort"] = self.cohort


class WorkloadV1Beta1(NamespacedResource):
    """Kueue Workload resource using v1beta1 API."""

    api_group: str = "kueue.x-k8s.io"
    api_version: str = "kueue.x-k8s.io/v1beta1"
    kind: str = "Workload"


@contextmanager
def create_resource_flavor_v1beta1(
    client: DynamicClient,
    name: str,
    node_labels: dict[str, str] | None = None,
    tolerations: list[dict[str, Any]] | None = None,
    teardown: bool = True,
) -> Generator[ResourceFlavorV1Beta1, Any, Any]:
    """
    Context manager to create and optionally delete a ResourceFlavor using v1beta1 API.
    """
    with ResourceFlavorV1Beta1(
        client=client,
        name=name,
        node_labels=node_labels,
        tolerations=tolerations,
        teardown=teardown,
    ) as resource_flavor:
        yield resource_flavor


@contextmanager
def create_local_queue_v1beta1(
    client: DynamicClient,
    name: str,
    cluster_queue: str,
    namespace: str,
    teardown: bool = True,
) -> Generator[LocalQueueV1Beta1, Any, Any]:
    """
    Context manager to create and optionally delete a LocalQueue using v1beta1 API.
    """
    with LocalQueueV1Beta1(
        client=client,
        name=name,
        cluster_queue=cluster_queue,
        namespace=namespace,
        teardown=teardown,
    ) as local_queue:
        yield local_queue


@contextmanager
def create_cluster_queue_v1beta1(
    client: DynamicClient,
    name: str,
    resource_groups: list[dict[str, Any]],
    namespace_selector: dict[str, Any] | None = None,
    stop_policy: str | None = None,
    cohort: str | None = None,
    teardown: bool = True,
) -> Generator[ClusterQueueV1Beta1, Any, Any]:
    """
    Context manager to create and optionally delete a ClusterQueue using v1beta1 API.
    """
    with ClusterQueueV1Beta1(
        client=client,
        name=name,
        resource_groups=resource_groups,
        namespace_selector=namespace_selector,
        stop_policy=stop_policy,
        cohort=cohort,
        teardown=teardown,
    ) as cluster_queue:
        yield cluster_queue


def create_test_resource_groups() -> list[dict[str, Any]]:
    """Create comprehensive resource groups for testing migration scenarios.

    Note: The flavor names must match the ResourceFlavor names created in test fixtures:
    - migration-test-cpu-flavor
    - migration-test-gpu-flavor
    """
    return [
        {
            "coveredResources": ["cpu", "memory"],
            "flavors": [
                {
                    "name": "migration-test-cpu-flavor",
                    "resources": [
                        {"name": "cpu", "nominalQuota": "4"},
                        {"name": "memory", "nominalQuota": "8Gi"},
                    ],
                }
            ],
        },
        {
            "coveredResources": ["nvidia.com/gpu"],
            "flavors": [
                {
                    "name": "migration-test-gpu-flavor",
                    "resources": [
                        {"name": "nvidia.com/gpu", "nominalQuota": "2"},
                    ],
                }
            ],
        },
    ]


def create_test_namespace_selector() -> dict[str, Any]:
    """Create test namespace selector for ClusterQueue."""
    return {"matchLabels": {"kueue-migration-test": "enabled"}}


def create_test_tolerations() -> list[dict[str, Any]]:
    """Create test tolerations for ResourceFlavor."""
    return [
        {
            "key": "nvidia.com/gpu",
            "operator": "Equal",
            "value": "true",
            "effect": "NoSchedule",
        },
        {
            "key": "test-workload",
            "operator": "Exists",
            "effect": "NoExecute",
        },
    ]


def create_test_node_labels() -> dict[str, str]:
    """Create test node labels for ResourceFlavor."""
    return {
        "node.kubernetes.io/instance-type": "test-gpu-node",
        "topology.zone": "test-zone-a",
        "kueue-test": "migration-test",
    }
