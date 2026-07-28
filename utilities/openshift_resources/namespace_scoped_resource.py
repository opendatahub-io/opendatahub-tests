"""Base class for namespace-scoped Kubernetes/OpenShift resources."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from typing import Any, Self

from utilities.openshift_resources.cluster_scoped_resource import ClusterScopedResource
from utilities.openshift_resources.oc import OCError, oc_get_json, run_oc


class NamespaceScopedResource(ClusterScopedResource):
    """Base for namespace-scoped resources (Pod, Deployment, Secret, etc.).

    Subclasses set ``kind`` to the Kubernetes kind name.
    """

    def __init__(
        self,
        name: str | None = None,
        namespace: str | None = None,
        label: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
        teardown: bool = True,
        kind_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if kind_dict and not namespace:
            namespace = kind_dict.get("metadata", {}).get("namespace")
        super().__init__(
            name=name, label=label, annotations=annotations, teardown=teardown, kind_dict=kind_dict, **kwargs
        )
        self.namespace = namespace or ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, namespace={self.namespace!r})"

    def _base_args(self, verb: str) -> list[str]:
        return [verb, f"{self._qualified_kind()}/{self.name}", "-n", self.namespace]

    def _get_json(self) -> dict[str, Any]:
        result = oc_get_json(kind=self._qualified_kind(), name=self.name, namespace=self.namespace)
        if isinstance(result, list):
            if not result:
                raise OCError(command=["get"], returncode=1, stderr=f"{self.kind}/{self.name} not found")
            return result[0]
        return result

    def _build_dict(self) -> dict[str, Any]:
        resource = super()._build_dict()
        resource["metadata"]["namespace"] = self.namespace
        return resource

    def _write(self, resource_dict: dict[str, Any], verb: str) -> Self:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as tmp:
            json.dump(resource_dict, tmp)
            tmp.flush()
            run_oc(verb, "-f", tmp.name, "-n", self.namespace, "-o", "json")  # noqa: FCN001 - run_oc uses *args
            return self

    @classmethod
    def list_resources(  # type: ignore[override]
        cls,
        namespace: str | None = None,
        label_selector: str | None = None,
        **kwargs: Any,
    ) -> Generator[NamespaceScopedResource]:
        """Yield resource objects. If namespace is None, lists across all namespaces."""
        args = ["get", cls._qualified_kind(), "-o", "json"]
        if namespace:
            args.extend(["-n", namespace])
        else:
            args.append("--all-namespaces")
        if label_selector:
            args.extend(["-l", label_selector])
        result = run_oc(*args)
        data = json.loads(result.stdout)
        items = data.get("items", []) if data.get("kind") == "List" else [data]
        for item in items:
            item_ns = item.get("metadata", {}).get("namespace", namespace or "")
            yield cls(name=item["metadata"]["name"], namespace=item_ns)
