"""RBAC tests for vLLM-Omni serving runtime."""

import pytest
import structlog
from kubernetes.client.rest import ApiException
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.template import Template
from pytest_testconfig import config as py_config

from utilities.constants import RuntimeTemplates

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "vllm-omni-rbac-ds-user"},
            id="test_vllm_omni_ds_project_user_rbac",
            marks=[pytest.mark.smoke],
        ),
    ],
    indirect=True,
)
class TestVllmOmniDSProjectUserRBAC:
    """DS Project User is forbidden from modifying the cluster-level ServingRuntime."""

    def test_vllm_omni_ds_project_user_cannot_modify_cluster_runtime(
        self,
        model_namespace: Namespace,
        unprivileged_client: DynamicClient,
    ) -> None:
        """Verify DS Project User receives 403 when patching a cluster-level Template.

        Given the vLLM-Omni runtime Template in the RHOAI applications namespace
        When a DS Project User (project-scoped only) attempts to PATCH it
        Then the operation returns HTTP 403 Forbidden.
        """
        cluster_runtime = Template(
            client=unprivileged_client,
            name=RuntimeTemplates.VLLM_OMNI_CUDA,
            namespace=py_config["applications_namespace"],
        )
        with (
            pytest.raises(ApiException) as exc_info,
            ResourceEditor(
                patches={cluster_runtime: {"metadata": {"annotations": {"test.rbac.unauthorized/attempt": "true"}}}}
            ),
        ):
            pass  # ResourceEditor applies the patch; the exception surfaces at context entry

        assert exc_info.value.status == 403, (
            f"Expected HTTP 403 Forbidden for DS Project User patching cluster Template, "
            f"got HTTP {exc_info.value.status}. "
            "DS Project Users must not modify shared cluster-level runtime templates."
        )
        LOGGER.info(
            event="DS Project User correctly denied PATCH on cluster Template",
            http_status=exc_info.value.status,
        )
