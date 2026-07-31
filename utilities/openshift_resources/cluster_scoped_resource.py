"""Base class for cluster-scoped Kubernetes/OpenShift resources.

Provides both sync and async APIs. Sync methods are the public interface.
Async implementations are prefixed with _async_.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import copy
import json
import subprocess
import types
from collections.abc import AsyncGenerator, Generator
from typing import Any, Self

import structlog
import yaml

from utilities.openshift_resources._sync import _run_sync
from utilities.openshift_resources.client import ApiClient, _k8s_plural, api_path
from utilities.openshift_resources.oc import OCError, ResourceNotFoundError, oc_get_json, run_oc
from utilities.openshift_resources.resource_dict import ResourceDict

logger = structlog.get_logger()

# -- SIGINT / cleanup safety ----------------------------------------------

_ACTIVE_RESOURCES: set[ClusterScopedResource] = set()


def _atexit_cleanup() -> None:
    if not _ACTIVE_RESOURCES:
        return
    for resource in list(_ACTIVE_RESOURCES):
        if resource._teardown:
            try:
                args = resource._base_args("delete") + ["--ignore-not-found"]
                subprocess.run(["oc", *args], capture_output=True, timeout=30, check=False)
            except Exception:
                logger.debug("atexit cleanup failed", resource=resource.name, exc_info=True)
    _ACTIVE_RESOURCES.clear()


atexit.register(_atexit_cleanup)  # noqa: FCN001


async def _collect(async_gen: AsyncGenerator) -> list:
    return [item async for item in async_gen]


class StopConditionError(Exception):
    """Raised when a wait method hits a stop condition (e.g., status=Failed)."""

    def __init__(self, resource: str, field: str, value: str) -> None:
        self.resource = resource
        self.field = field
        self.value = value
        super().__init__(f"{resource}: hit stop condition {field}={value}")


class ClusterScopedResource:
    """Base for cluster-scoped resources (Namespace, Node, ClusterRole, etc.)."""

    api_version: str = "v1"
    api_group: str = ""

    class Status:
        ACTIVE: str = "Active"
        COMPLETE: str = "Complete"
        COMPLETED: str = "Completed"
        CRASH_LOOPBACK_OFF: str = "CrashLoopBackOff"
        FAILED: str = "Failed"
        PENDING: str = "Pending"
        READY: str = "Ready"
        RUNNING: str = "Running"
        SUCCEEDED: str = "Succeeded"
        TERMINATING: str = "Terminating"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.kind = cls.__name__

    @classmethod
    def full_api_version(cls) -> str:
        if cls.api_group:
            return f"{cls.api_group}/{cls.api_version}"
        return cls.api_version

    @classmethod
    def _resource_plural(cls) -> str:
        if hasattr(cls, "plural"):
            return cls.plural
        return _k8s_plural(kind=cls.kind)

    def __init__(
        self,
        name: str | None = None,
        label: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
        teardown: bool = True,
        kind_dict: dict[str, Any] | None = None,
        ensure_exists: bool = False,
        wait_for_resource: bool = False,
        client: ApiClient | None = None,
        **kwargs: Any,
    ) -> None:
        if kind_dict:
            name = kind_dict.get("metadata", {}).get("name", name)
            label = kind_dict.get("metadata", {}).get("labels", label)
            annotations = kind_dict.get("metadata", {}).get("annotations", annotations)
            self._kind_dict = kind_dict
        else:
            self._kind_dict = None
        self.name = name or ""
        self.namespace: str = ""
        self._label = label or {}
        self._annotations = annotations or {}
        self._teardown = teardown
        self._ensure_exists = ensure_exists
        self._wait_for_resource = wait_for_resource
        self._client = client

    # -- API path helpers --------------------------------------------------

    def _api_path(self, collection: bool = False) -> str:
        return api_path(
            api_group=self.api_group,
            api_version=self.api_version,
            plural=self._resource_plural(),
            name=None if collection else self.name,
            namespace=self.namespace or None,
        )

    @classmethod
    def _collection_path(cls, namespace: str | None = None) -> str:
        return api_path(
            api_group=cls.api_group,
            api_version=cls.api_version,
            plural=cls._resource_plural(),
            namespace=namespace,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    # ======================================================================
    # ASYNC implementation
    # ======================================================================

    async def _async_ensure_exists(self) -> None:
        if not await self._async_exists():
            raise ResourceNotFoundError(
                command=["get"], returncode=1, stderr=f"{self.kind}/{self.name} not found on cluster"
            )

    async def _async_wait_until_exists(self, timeout: int = 300, poll_interval: int = 5) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if await self._async_exists():
                return
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"{self.kind}/{self.name}: not found within {timeout}s")

    async def __aenter__(self) -> Self:
        if self._ensure_exists:
            await self._async_ensure_exists()
        else:
            await self._async_create()
        if self._wait_for_resource:
            await self._async_wait_until_exists()
        if self._teardown:
            _ACTIVE_RESOURCES.add(self)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        _ACTIVE_RESOURCES.discard(self)
        if self._teardown:
            await self._async_delete(ignore_not_found=True)

    async def _async_exists(self) -> bool:
        try:
            await self._get_json()
            return True
        except OCError as err:
            if "NotFound" in err.stderr or "not found" in err.stderr:
                return False
            raise

    async def _async_instance(self) -> ResourceDict:
        return ResourceDict(await self._get_json())  # noqa: FCN001

    def to_dict(self) -> dict[str, Any]:
        return self._build_dict()

    async def _async_output(self, output_format: str = "yaml") -> str:
        if self._client:
            data = await self._get_json()
            if output_format == "json":
                return json.dumps(data, indent=2)
            if output_format == "yaml":
                return yaml.dump(data, default_flow_style=False)
            if output_format == "name":
                return f"{self.kind.lower()}/{self.name}"
        args = [*self._base_args("get"), "-o", output_format]
        result = await run_oc(args=args)
        return result.stdout.strip()

    async def _async_describe(self) -> str:
        result = await run_oc(args=self._base_args("describe"))
        return result.stdout

    async def _async_create(self, from_dict: dict[str, Any] | None = None) -> Self:
        return await self._write(resource_dict=from_dict or self._kind_dict or self._build_dict(), verb="create")

    async def _async_apply(self, resource_dict: dict[str, Any] | None = None) -> Self:
        return await self._write(resource_dict=resource_dict or self._build_dict(), verb="apply")

    async def _async_delete(
        self,
        wait: bool = False,
        timeout: int = 120,
        ignore_not_found: bool = False,
    ) -> None:
        if self._client:
            try:
                await self._client.delete(path=self._api_path())
            except ResourceNotFoundError:
                if not ignore_not_found:
                    raise
            if wait:
                await self._async_wait_deleted(timeout=timeout)
            return
        args = self._base_args(verb="delete")
        if ignore_not_found:
            args.append("--ignore-not-found")
        if wait:
            args.extend(["--wait=true", "--timeout", f"{timeout}s"])
        await run_oc(args=args)

    async def _async_label(self, labels: dict[str, str | None], overwrite: bool = True) -> None:
        if self._client:
            await self._client.patch(path=self._api_path(), body={"metadata": {"labels": labels}})
            return
        args = self._base_args(verb="label")
        if overwrite:
            args.append("--overwrite")
        for key, value in labels.items():
            args.append(f"{key}-" if value is None else f"{key}={value}")
        await run_oc(args=args)

    async def _async_annotate(self, annotations: dict[str, str | None], overwrite: bool = True) -> None:
        if self._client:
            await self._client.patch(path=self._api_path(), body={"metadata": {"annotations": annotations}})
            return
        args = self._base_args(verb="annotate")
        if overwrite:
            args.append("--overwrite")
        for key, value in annotations.items():
            args.append(f"{key}-" if value is None else f"{key}={value}")
        await run_oc(args=args)

    async def _async_edit(self, patch: dict[str, Any], subresource: str | None = None) -> None:
        await self._apply_patch(patch=patch, subresource=subresource)

    @contextlib.asynccontextmanager
    async def _async_patch_and_restore(
        self, patch: dict[str, Any], subresource: str | None = None
    ) -> AsyncGenerator[Self]:
        snapshot = copy.deepcopy(await self._get_json())
        await self._apply_patch(patch=patch, subresource=subresource)
        try:
            yield self
        finally:
            restore = _build_restore_patch(snapshot, patch)
            if restore:
                await self._apply_patch(restore, subresource=subresource)

    async def _async_watch(
        self, timeout: int = 300, resource_version: str | None = None
    ) -> AsyncGenerator[dict[str, Any]]:
        if not self._client:
            raise NotImplementedError("watch() requires an HTTP client.")
        async for event in self._client.watch(
            self._api_path(collection=True),
            resource_version=resource_version,
            timeout=timeout,
            field_selector=f"metadata.name={self.name}",
        ):
            yield event

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
            cls._collection_path(),
            resource_version=resource_version,
            timeout=timeout,
            label_selector=label_selector,
        ):
            yield event

    async def _async_wait_for_condition(
        self,
        condition: str,
        status: str = "True",
        timeout: int = 300,
        poll_interval: int = 5,
        stop_condition: str | None = None,
    ) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                conditions = (await self._get_json()).get("status", {}).get("conditions", [])
                for cond in conditions:
                    if cond.get("type") == condition and cond.get("status") == status:
                        return
                    if stop_condition and cond.get("type") == stop_condition and cond.get("status") == "True":
                        raise StopConditionError(
                            resource=f"{self.kind}/{self.name}", field="condition", value=stop_condition
                        )
            except OCError:
                pass
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"{self.kind}/{self.name}: {condition}={status} not met within {timeout}s")

    async def _async_wait_for_status(
        self,
        status: str,
        timeout: int = 300,
        poll_interval: int = 5,
        stop_status: str | None = None,
    ) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                phase = (await self._get_json()).get("status", {}).get("phase")
                if phase == status:
                    return
                if stop_status and phase == stop_status:
                    raise StopConditionError(resource=f"{self.kind}/{self.name}", field="phase", value=stop_status)
            except OCError:
                pass
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"{self.kind}/{self.name}: status {status} not reached within {timeout}s")

    async def _async_wait_deleted(self, timeout: int = 120, poll_interval: int = 5) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if not await self._async_exists():
                return
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"{self.kind}/{self.name}: still exists after {timeout}s")

    @classmethod
    async def _async_list_resources(
        cls,
        label_selector: str | None = None,
        client: ApiClient | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ClusterScopedResource]:
        if client:
            params = {}
            if label_selector:
                params["labelSelector"] = label_selector
            result = await client.get(path=cls._collection_path(), **params)
            items = result.get("items", [result])
            for item in items:
                yield cls(name=item["metadata"]["name"], client=client)
            return
        result = await oc_get_json(kind=cls._qualified_kind(), label_selector=label_selector)
        items = result if isinstance(result, list) else [result]
        for item in items:
            yield cls(name=item["metadata"]["name"])

    # ======================================================================
    # SYNC public API
    # ======================================================================

    def __enter__(self) -> Self:
        return _run_sync(coro=self.__aenter__())

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        _run_sync(coro=self.__aexit__(exc_type, exc_val, exc_tb))

    @property
    def instance(self) -> ResourceDict:
        return _run_sync(coro=self._async_instance())

    @property
    def exists(self) -> bool:
        return _run_sync(coro=self._async_exists())

    def ensure_exists(self) -> None:
        _run_sync(coro=self._async_ensure_exists())

    def wait_until_exists(self, timeout: int = 300, poll_interval: int = 5) -> None:
        _run_sync(coro=self._async_wait_until_exists(timeout=timeout, poll_interval=poll_interval))

    def output(self, output_format: str = "yaml") -> str:
        return _run_sync(coro=self._async_output(output_format=output_format))

    def describe(self) -> str:
        return _run_sync(coro=self._async_describe())

    def create(self, from_dict: dict[str, Any] | None = None) -> Self:
        return _run_sync(coro=self._async_create(from_dict=from_dict))

    def apply(self, resource_dict: dict[str, Any] | None = None) -> Self:
        return _run_sync(coro=self._async_apply(resource_dict=resource_dict))

    def delete(self, wait: bool = False, timeout: int = 120, ignore_not_found: bool = False) -> None:
        _run_sync(coro=self._async_delete(wait=wait, timeout=timeout, ignore_not_found=ignore_not_found))

    def label(self, labels: dict[str, str | None], overwrite: bool = True) -> None:
        _run_sync(coro=self._async_label(labels=labels, overwrite=overwrite))

    def annotate(self, annotations: dict[str, str | None], overwrite: bool = True) -> None:
        _run_sync(coro=self._async_annotate(annotations=annotations, overwrite=overwrite))

    def edit(self, patch: dict[str, Any], subresource: str | None = None) -> None:
        _run_sync(coro=self._async_edit(patch=patch, subresource=subresource))

    @contextlib.contextmanager
    def patch_and_restore(self, patch: dict[str, Any], subresource: str | None = None) -> Generator[Self]:
        snapshot = copy.deepcopy(_run_sync(coro=self._get_json()))
        _run_sync(coro=self._apply_patch(patch=patch, subresource=subresource))
        try:
            yield self
        finally:
            restore = _build_restore_patch(snapshot, patch)
            if restore:
                _run_sync(coro=self._apply_patch(restore, subresource=subresource))

    def wait_for_condition(
        self,
        condition: str,
        status: str = "True",
        timeout: int = 300,
        poll_interval: int = 5,
        stop_condition: str | None = None,
    ) -> None:
        _run_sync(
            coro=self._async_wait_for_condition(
                condition=condition,
                status=status,
                timeout=timeout,
                poll_interval=poll_interval,
                stop_condition=stop_condition,
            )
        )

    def wait_for_status(
        self,
        status: str,
        timeout: int = 300,
        poll_interval: int = 5,
        stop_status: str | None = None,
    ) -> None:
        _run_sync(
            coro=self._async_wait_for_status(
                status=status,
                timeout=timeout,
                poll_interval=poll_interval,
                stop_status=stop_status,
            )
        )

    def wait_deleted(self, timeout: int = 120, poll_interval: int = 5) -> None:
        _run_sync(coro=self._async_wait_deleted(timeout=timeout, poll_interval=poll_interval))

    @classmethod
    def list_resources(
        cls,
        label_selector: str | None = None,
        client: ApiClient | None = None,
        **kwargs: Any,
    ) -> Generator[ClusterScopedResource]:
        items = _run_sync(
            coro=_collect(
                async_gen=cls._async_list_resources(
                    label_selector=label_selector,
                    client=client,
                    **kwargs,
                )
            )
        )
        yield from items

    # ======================================================================
    # Internal (shared by both sync and async paths)
    # ======================================================================

    @classmethod
    def _qualified_kind(cls) -> str:
        if cls.api_group:
            return f"{cls.kind}.{cls.api_group}"
        return cls.kind

    def _base_args(self, verb: str) -> list[str]:
        return [verb, f"{self._qualified_kind()}/{self.name}"]

    async def _get_json(self) -> dict[str, Any]:
        if self._client:
            return await self._client.get(path=self._api_path())
        result = await oc_get_json(kind=self._qualified_kind(), name=self.name)
        if isinstance(result, list):
            if not result:
                raise OCError(command=["get"], returncode=1, stderr=f"{self.kind}/{self.name} not found")
            return result[0]
        return result

    async def _apply_patch(self, patch: dict[str, Any], subresource: str | None = None) -> None:
        if self._client:
            path = self._api_path()
            if subresource:
                path = f"{path}/{subresource}"
            await self._client.patch(path=path, body=patch)
            return
        args = self._base_args("patch") + ["--type", "merge", "-p", json.dumps(patch)]
        if subresource:
            args.extend(["--subresource", subresource])
        await run_oc(args=args)

    def _build_dict(self) -> dict[str, Any]:
        resource: dict[str, Any] = {
            "apiVersion": self.full_api_version(),
            "kind": self.kind,
            "metadata": {"name": self.name},
        }
        if self._label:
            resource["metadata"]["labels"] = self._label
        if self._annotations:
            resource["metadata"]["annotations"] = self._annotations
        return resource

    async def _write(self, resource_dict: dict[str, Any], verb: str) -> Self:
        if self._client:
            if verb == "create":
                await self._client.post(path=self._api_path(collection=True), body=resource_dict)
            elif verb == "apply":
                await self._client.server_side_apply(self._api_path(), body=resource_dict)
            return self
        await run_oc(args=[verb, "-f", "-", "-o", "json"], input=json.dumps(resource_dict))
        return self


_RESTORE_SKIP_KEYS = {"resourceVersion", "uid", "creationTimestamp", "generation", "managedFields"}


def _build_restore_patch(snapshot: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    restore: dict[str, Any] = {}
    for key, value in patch.items():
        if key in _RESTORE_SKIP_KEYS:
            continue
        if isinstance(value, dict) and isinstance(snapshot.get(key), dict):
            nested = _build_restore_patch(snapshot=snapshot[key], patch=value)
            if nested:
                restore[key] = nested
        else:
            restore[key] = snapshot.get(key, None)
    return restore
