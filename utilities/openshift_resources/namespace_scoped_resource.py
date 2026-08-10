"""Base class for namespace-scoped Kubernetes/OpenShift resources."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Generator
from typing import Any, Self

from utilities.openshift_resources._sync import _run_sync
from utilities.openshift_resources.client import ApiClient
from utilities.openshift_resources.cluster_scoped_resource import ClusterScopedResource, _collect
from utilities.openshift_resources.oc import OCError, oc_get_json, run_oc

__all__ = ["NamespaceScopedResource"]


class NamespaceScopedResource(ClusterScopedResource):
    """Base for namespace-scoped resources (Pod, Deployment, Secret, etc.)."""

    def __init__(
        self,
        name: str | None = None,
        namespace: str | None = None,
        label: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
        teardown: bool = True,
        kind_dict: dict[str, Any] | None = None,
        ensure_exists: bool = False,
        wait_for_resource: bool = False,
        client: ApiClient | None = None,
        **kwargs: Any,
    ) -> None:
        if kind_dict and not namespace:
            namespace = kind_dict.get("metadata", {}).get("namespace")
        super().__init__(
            name=name,
            label=label,
            annotations=annotations,
            teardown=teardown,
            kind_dict=kind_dict,
            ensure_exists=ensure_exists,
            wait_for_resource=wait_for_resource,
            client=client,
            **kwargs,
        )
        self.namespace: str = namespace or ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, namespace={self.namespace!r})"

    def _base_args(self, verb: str) -> list[str]:
        return [verb, f"{self._qualified_kind()}/{self.name}", "-n", self.namespace]

    async def _get_json(self) -> dict[str, Any]:
        if self._client:
            return await self._client.get(path=self._api_path())
        result = await oc_get_json(kind=self._qualified_kind(), name=self.name, namespace=self.namespace)
        if isinstance(result, list):
            if not result:
                raise OCError(command=["get"], returncode=1, stderr=f"{self.kind}/{self.name} not found")
            return result[0]
        return result

    def _build_dict(self) -> dict[str, Any]:
        resource = super()._build_dict()
        resource["metadata"]["namespace"] = self.namespace
        return resource

    async def _write(self, resource_dict: dict[str, Any], verb: str) -> Self:
        if self._client:
            if verb == "create":
                await self._client.post(path=self._api_path(collection=True), body=resource_dict)
            elif verb == "apply":
                await self._client.server_side_apply(self._api_path(), body=resource_dict)
            return self
        await run_oc(args=[verb, "-f", "-", "-n", self.namespace, "-o", "json"], input=json.dumps(resource_dict))
        return self

    # -- Async --

    @classmethod
    async def _async_list_resources(  # type: ignore[override]
        cls,
        namespace: str | None = None,
        label_selector: str | None = None,
        client: ApiClient | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[NamespaceScopedResource]:
        if client:
            params = {}
            if label_selector:
                params["labelSelector"] = label_selector
            path = cls._collection_path(namespace=namespace)
            result = await client.get(path=path, **params)
            items = result.get("items", [result])
            for item in items:
                item_ns = item.get("metadata", {}).get("namespace", namespace or "")
                yield cls(name=item["metadata"]["name"], namespace=item_ns, client=client)
            return

        result = await oc_get_json(
            kind=cls._qualified_kind(),
            namespace=namespace,
            all_namespaces=namespace is None,
            label_selector=label_selector,
        )
        items = result if isinstance(result, list) else [result]
        for item in items:
            item_ns = item.get("metadata", {}).get("namespace", namespace or "")
            yield cls(name=item["metadata"]["name"], namespace=item_ns)

    @classmethod
    async def _async_watch_resources(
        cls,
        client: ApiClient,
        namespace: str | None = None,
        timeout: int = 300,
        label_selector: str | None = None,
        resource_version: str | None = None,
    ) -> AsyncGenerator[dict[str, Any]]:
        async for event in client.watch(
            cls._collection_path(namespace=namespace),
            resource_version=resource_version,
            timeout=timeout,
            label_selector=label_selector,
        ):
            yield event

    # -- Sync --

    @classmethod
    def list_resources(  # type: ignore[override]
        cls,
        namespace: str | None = None,
        label_selector: str | None = None,
        client: ApiClient | None = None,
        **kwargs: Any,
    ) -> Generator[NamespaceScopedResource]:
        items = _run_sync(
            coro=_collect(
                async_gen=cls._async_list_resources(
                    namespace=namespace,
                    label_selector=label_selector,
                    client=client,
                    **kwargs,
                )
            )
        )
        yield from items
