"""Deployment resource."""

from __future__ import annotations

import asyncio
from typing import Any

from utilities.openshift_resources._sync import _run_sync
from utilities.openshift_resources.namespace_scoped_resource import NamespaceScopedResource
from utilities.openshift_resources.oc import ResourceNotFoundError


class Deployment(NamespaceScopedResource):
    """Deployment enables declarative updates for Pods and ReplicaSets."""

    api_group: str = "apps"
    api_version: str = "v1"

    def __init__(
        self,
        min_ready_seconds: int | None = None,
        paused: bool | None = None,
        progress_deadline_seconds: int | None = None,
        replicas: int | None = None,
        revision_history_limit: int | None = None,
        selector: dict[str, Any] | None = None,
        strategy: dict[str, Any] | None = None,
        template: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            min_ready_seconds (int): Minimum number of seconds for which a newly created pod
                should be ready without any of its container crashing, for it to be considered available.
            paused (bool): Indicates that the deployment is paused.
            progress_deadline_seconds (int): The maximum time in seconds for a deployment to
                make progress before it is considered to be failed.
            replicas (int): Number of desired pods. This is a pointer to distinguish between
                explicit zero and not specified. Defaults to 1.
            revision_history_limit (int): The number of old ReplicaSets to retain to allow
                rollback. This is a pointer to distinguish between explicit zero and not specified.
            selector (dict[str, Any]) (required): Label selector for pods. Existing ReplicaSets
                whose pods are selected by this will be the ones affected by this deployment.
            strategy (dict[str, Any]): The deployment strategy to use to replace existing
                pods with new ones.
            template (dict[str, Any]) (required): Template describes the pods that will be
                created. The only allowed template.spec.restartPolicy value is "Always".
        """
        super().__init__(**kwargs)
        self.min_ready_seconds = min_ready_seconds
        self.paused = paused
        self.progress_deadline_seconds = progress_deadline_seconds
        self.replicas = replicas
        self.revision_history_limit = revision_history_limit
        self.selector = selector
        self.strategy = strategy
        self.template = template

    def _build_dict(self) -> dict[str, Any]:
        resource = super()._build_dict()

        spec: dict[str, Any] = {}
        if self.min_ready_seconds is not None:
            spec["minReadySeconds"] = self.min_ready_seconds
        if self.paused is not None:
            spec["paused"] = self.paused
        if self.progress_deadline_seconds is not None:
            spec["progressDeadlineSeconds"] = self.progress_deadline_seconds
        if self.replicas is not None:
            spec["replicas"] = self.replicas
        if self.revision_history_limit is not None:
            spec["revisionHistoryLimit"] = self.revision_history_limit
        spec["selector"] = self.selector
        if self.strategy is not None:
            spec["strategy"] = self.strategy
        spec["template"] = self.template
        if spec:
            resource["spec"] = spec

        return resource

    # -- Async --

    async def _async_scale_replicas(self, replicas: int) -> None:
        await self._async_edit(patch={"spec": {"replicas": replicas}})

    async def _async_wait_for_replicas(
        self,
        timeout: int = 300,
        poll_interval: int = 5,
        replicas: int | None = None,
    ) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                data = await self._get_json()
                desired = replicas if replicas is not None else data.get("spec", {}).get("replicas", 1)
                ready = data.get("status", {}).get("readyReplicas", 0)
                if ready >= desired:
                    return
            except ResourceNotFoundError:
                pass
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"{self.kind}/{self.name}: replicas not ready within {timeout}s")

    # -- Sync --

    def scale_replicas(self, replicas: int) -> None:
        _run_sync(coro=self._async_scale_replicas(replicas=replicas))

    def wait_for_replicas(self, timeout: int = 300, poll_interval: int = 5, replicas: int | None = None) -> None:
        _run_sync(coro=self._async_wait_for_replicas(timeout=timeout, poll_interval=poll_interval, replicas=replicas))
