"""Base class for cluster-scoped Kubernetes/OpenShift resources."""

from __future__ import annotations

import contextlib
import copy
import json
import tempfile
import time
import types
from collections.abc import Generator
from typing import Any, Self

from utilities.openshift_resources.oc import OCError, oc_get_json, run_oc
from utilities.openshift_resources.resource_dict import ResourceDict


class ClusterScopedResource:
    """Base for cluster-scoped resources (Namespace, Node, ClusterRole, etc.).

    Subclasses set ``kind`` to the Kubernetes kind name.
    """

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
        """Return the full apiVersion string (e.g., 'apps/v1', 'v1')."""
        if cls.api_group:
            return f"{cls.api_group}/{cls.api_version}"
        return cls.api_version

    def __init__(
        self,
        name: str | None = None,
        label: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
        teardown: bool = True,
        kind_dict: dict[str, Any] | None = None,
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
        self._label = label or {}
        self._annotations = annotations or {}
        self._teardown = teardown

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    def __enter__(self) -> Self:
        self.create()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._teardown:
            self.delete(ignore_not_found=True)

    @property
    def exists(self) -> bool:
        """Check if the resource exists on the cluster."""
        try:
            self._get_json()
            return True
        except OCError as err:
            if "NotFound" in err.stderr or "not found" in err.stderr:
                return False
            raise

    @property
    def instance(self) -> ResourceDict:
        """Fetch the live resource from the cluster and cache it."""
        self._instance = ResourceDict(self._get_json())  # noqa: FCN001 - dict subclass constructor
        return self._instance

    def to_dict(self) -> dict[str, Any]:
        """Return the local resource definition as a dict (without fetching from cluster)."""
        return self._build_dict()

    def create(self, from_dict: dict[str, Any] | None = None) -> Self:
        """Create the resource. Returns self."""
        return self._write(resource_dict=from_dict or self._kind_dict or self._build_dict(), verb="create")

    def apply(self, resource_dict: dict[str, Any] | None = None) -> Self:
        """Apply (create or update) the resource. Returns self."""
        return self._write(resource_dict=resource_dict or self._build_dict(), verb="apply")

    def output(self, format: str = "yaml") -> str:
        """Return the resource in the given format (yaml, wide, name, jsonpath=..., etc.)."""
        args = self._base_args("get") + ["-o", format]
        return run_oc(*args).stdout.strip()

    def describe(self) -> str:
        """Return the human-readable describe output."""
        return run_oc(*self._base_args("describe")).stdout

    def delete(
        self,
        wait: bool = False,
        timeout: int = 120,
        ignore_not_found: bool = False,
    ) -> None:
        """Delete the resource."""
        args = self._base_args(verb="delete")
        if ignore_not_found:
            args.append("--ignore-not-found")
        if wait:
            args.extend(["--wait=true", "--timeout", f"{timeout}s"])
        run_oc(*args)

    def label(self, labels: dict[str, str | None], overwrite: bool = True) -> None:
        """Add or remove labels. Set a value to None to remove."""
        args = self._base_args(verb="label")
        if overwrite:
            args.append("--overwrite")
        for key, value in labels.items():
            args.append(f"{key}-" if value is None else f"{key}={value}")
        run_oc(*args)

    def annotate(self, annotations: dict[str, str | None], overwrite: bool = True) -> None:
        """Add or remove annotations. Set a value to None to remove."""
        args = self._base_args(verb="annotate")
        if overwrite:
            args.append("--overwrite")
        for key, value in annotations.items():
            args.append(f"{key}-" if value is None else f"{key}={value}")
        run_oc(*args)

    def edit(self, patch: dict[str, Any], subresource: str | None = None) -> None:
        """Patch the resource with a merge patch.

        Args:
            patch: The merge patch dict.
            subresource: Optional subresource to target (e.g., "status", "scale").
        """
        self._apply_patch(patch=patch, subresource=subresource)

    @contextlib.contextmanager
    def patch_and_restore(self, patch: dict[str, Any], subresource: str | None = None) -> Generator[Self]:
        """Apply a merge patch on enter, auto-restore original state on exit.

        with resource.patch_and_restore({"metadata": {"labels": {"env": "staging"}}}) as resource:
            assert resource.instance.metadata.labels["env"] == "staging"
        # original state restored here
        """
        snapshot = copy.deepcopy(self._get_json())
        self._apply_patch(patch=patch, subresource=subresource)
        try:
            yield self
        finally:
            restore = _build_restore_patch(snapshot, patch)
            if restore:
                self._apply_patch(restore, subresource=subresource)

    def _apply_patch(self, patch: dict[str, Any], subresource: str | None = None) -> None:
        args = self._base_args("patch") + ["--type", "merge", "-p", json.dumps(patch)]
        if subresource:
            args.extend(["--subresource", subresource])
        run_oc(*args)

    # -- waiting -----------------------------------------------------------

    def wait_for_condition(
        self,
        condition: str,
        status: str = "True",
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> None:
        """Wait until a status condition matches."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                conditions = self._get_json().get("status", {}).get("conditions", [])
                for cond in conditions:
                    if cond.get("type") == condition and cond.get("status") == status:
                        return
            except OCError:
                pass
            time.sleep(poll_interval)
        raise TimeoutError(f"{self.kind}/{self.name}: {condition}={status} not met within {timeout}s")

    def wait_for_status(self, status: str, timeout: int = 300, poll_interval: int = 5) -> None:
        """Wait until status.phase matches."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self._get_json().get("status", {}).get("phase") == status:
                    return
            except OCError:
                pass
            time.sleep(poll_interval)
        raise TimeoutError(f"{self.kind}/{self.name}: status {status} not reached within {timeout}s")

    def wait_deleted(self, timeout: int = 120, poll_interval: int = 5) -> None:
        """Wait until the resource no longer exists."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.exists:
                return
            time.sleep(poll_interval)
        raise TimeoutError(f"{self.kind}/{self.name}: still exists after {timeout}s")

    # -- class methods -----------------------------------------------------

    @classmethod
    def list_resources(
        cls,
        label_selector: str | None = None,
        **kwargs: Any,
    ) -> Generator[ClusterScopedResource]:
        """Yield resource objects of this kind."""
        result = oc_get_json(kind=cls._qualified_kind(), label_selector=label_selector)
        items = result if isinstance(result, list) else [result]
        for item in items:
            yield cls(name=item["metadata"]["name"])

    # -- internal ----------------------------------------------------------

    @classmethod
    def _qualified_kind(cls) -> str:
        if cls.api_group:
            return f"{cls.kind}.{cls.api_group}"
        return cls.kind

    def _base_args(self, verb: str) -> list[str]:
        return [verb, f"{self._qualified_kind()}/{self.name}"]

    def _get_json(self) -> dict[str, Any]:
        result = oc_get_json(kind=self._qualified_kind(), name=self.name)
        if isinstance(result, list):
            if not result:
                raise OCError(command=["get"], returncode=1, stderr=f"{self.kind}/{self.name} not found")
            return result[0]
        return result

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

    def _write(self, resource_dict: dict[str, Any], verb: str) -> Self:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as tmp:
            json.dump(resource_dict, tmp)
            tmp.flush()
            run_oc(verb, "-f", tmp.name, "-o", "json")  # noqa: FCN001 - run_oc uses *args
            return self


_RESTORE_SKIP_KEYS = {"resourceVersion", "uid", "creationTimestamp", "generation", "managedFields"}


def _build_restore_patch(snapshot: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Build a patch that restores snapshot values for every key touched by patch."""
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
