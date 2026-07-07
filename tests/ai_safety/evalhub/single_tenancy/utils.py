"""Utilities for EvalHub single-tenancy integration tests."""

from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.evalhub import EvalHub


def _is_evalhub_crd_available(admin_client: DynamicClient) -> bool:
    """Check if EvalHub CRD is installed on the cluster."""
    crd_name = "evalhubs.trustyai.opendatahub.io"
    crd = CustomResourceDefinition(
        client=admin_client,
        name=crd_name,
    )
    return crd.exists is not None


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
