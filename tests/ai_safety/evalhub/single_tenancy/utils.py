"""Utilities for EvalHub single-tenancy integration tests."""

from __future__ import annotations

from ocp_resources.evalhub import EvalHub


class SingleTenantEvalHub(EvalHub):
    """EvalHub subclass that injects spec.tenancy: single into the resource dict.

    The auto-generated ocp_resources.EvalHub class has no tenancy parameter.
    This subclass overrides to_dict() to inject the field after the parent
    populates spec from the standard kwargs.
    """

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            self.res.setdefault("spec", {})["tenancy"] = "single"
