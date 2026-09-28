"""Regression tests verifying existing vLLM templates remain unmodified after Omni addition."""

from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.template import Template
from pytest_testconfig import config as py_config

from utilities.constants import RuntimeTemplates

LOGGER = structlog.get_logger(name=__name__)


# Flag that must NOT appear in any non-Omni vLLM template's container args.
OMNI_FLAG: str = "--omni"


def _get_template(admin_client: DynamicClient, template_name: str) -> Template:
    """Return a Template resource object from the RHOAI applications namespace.

    Args:
        admin_client: Kubernetes dynamic client with admin privileges.
        template_name: Name of the ServingRuntime template to retrieve.

    Returns:
        Template: The requested Template resource (existence not yet asserted).
    """
    return Template(
        client=admin_client,
        name=template_name,
        namespace=py_config["applications_namespace"],
    )


def _get_first_container_args(template: Template) -> list[str]:
    """Extract the first container's args list from a ServingRuntime template.

    The RHOAI operator stores ServingRuntime definitions as Template objects
    where the first object in ``template.instance.objects`` is the runtime spec.

    Args:
        template: A Template resource whose first embedded object is a ServingRuntime.

    Returns:
        list[str]: The container args, or an empty list when none are defined.
    """
    model_dict: dict[str, Any] = template.instance.objects[0].to_dict()
    containers: list[dict[str, Any]] = model_dict.get("spec", {}).get("containers", [])
    if not containers:
        return []
    return list(containers[0].get("args") or [])


class TestVllmOmniDoesNotBreakExistingCudaTemplate:
    """vLLM-Omni addition must not modify the standard vLLM CUDA template.

    Both the vLLM CUDA and vLLM-Omni templates must co-exist independently in
    the RHOAI applications namespace.  The CUDA template's container args must
    remain exactly as shipped — in particular, the '--omni' flag must not
    appear in them.
    """

    @pytest.mark.smoke
    def test_vllm_omni_does_not_break_existing_vllm(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify the vLLM CUDA template is unchanged after the Omni template is added.

        Given:
            Both the standard vLLM CUDA and the new vLLM-Omni ServingRuntime
            templates are registered in the RHOAI applications namespace.
        When:
            Retrieving each template and inspecting the CUDA template's first
            container args.
        Then:
            - The vLLM CUDA template exists.
            - The vLLM-Omni template exists as a distinct, additive entry.
            - The vLLM CUDA template's container args do NOT contain '--omni'.
        """
        applications_namespace: str = py_config["applications_namespace"]

        cuda_template = _get_template(
            admin_client=admin_client,
            template_name=RuntimeTemplates.VLLM_CUDA,
        )
        omni_template = _get_template(
            admin_client=admin_client,
            template_name=RuntimeTemplates.VLLM_OMNI_CUDA,
        )

        assert cuda_template.exists, (
            f"vLLM CUDA template '{RuntimeTemplates.VLLM_CUDA}' was not found in namespace "
            f"'{applications_namespace}'. The vLLM-Omni addition must not remove or rename "
            "existing templates."
        )
        assert omni_template.exists, (
            f"vLLM-Omni template '{RuntimeTemplates.VLLM_OMNI_CUDA}' was not found in namespace "
            f"'{applications_namespace}'. Ensure the RHOAI operator version includes the "
            "vLLM-Omni runtime template."
        )

        cuda_args = _get_first_container_args(template=cuda_template)
        omni_args = [arg for arg in cuda_args if arg.startswith(OMNI_FLAG)]
        assert not omni_args, (
            f"vLLM CUDA template '{RuntimeTemplates.VLLM_CUDA}' must not contain '{OMNI_FLAG}' "
            f"in its container args, but found {omni_args} in: {cuda_args}. "
            "This indicates the CUDA template was incorrectly modified."
        )

        LOGGER.info(
            event="vLLM CUDA template present and unmodified",
            cuda_template=RuntimeTemplates.VLLM_CUDA,
            omni_template=RuntimeTemplates.VLLM_OMNI_CUDA,
            cuda_args=cuda_args,
        )
