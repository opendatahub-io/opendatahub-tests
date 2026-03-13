from typing import Self

import pytest
from ocp_resources.deployment import Deployment
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

from tests.model_registry.constants import MR_INSTANCE_NAME, MR_POSTGRES_DEPLOYMENT_NAME_STR
from tests.model_registry.scc.utils import (
    validate_deployment_scc,
    validate_pod_scc,
)

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_registry_metadata_db_resources, model_registry_instance",
    [
        pytest.param({}, {}),
        pytest.param({"db_name": "default"}, {"db_name": "default"}),
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_metadata_db_resources",
    "model_registry_instance",
)
@pytest.mark.custom_namespace
@pytest.mark.skip_must_gather
class TestModelRegistrySecurityContextValidation:
    @pytest.mark.parametrize(
        "deployment_model_registry_ns",
        [
            pytest.param({"deployment_name": MR_INSTANCE_NAME}),
            pytest.param({"deployment_name": MR_POSTGRES_DEPLOYMENT_NAME_STR}),
        ],
        indirect=["deployment_model_registry_ns"],
    )
    @pytest.mark.tier2
    def test_model_registry_deployment_security_context_validation(
        self: Self,
        skip_if_not_valid_check: None,
        deployment_model_registry_ns: Deployment,
    ):
        """
        Validate that model registry deployment does not set runAsUser/runAsGroup
        """
        validate_deployment_scc(deployment=deployment_model_registry_ns)

    @pytest.mark.parametrize(
        "pod_model_registry_ns",
        [
            pytest.param({"deployment_name": MR_INSTANCE_NAME}, id="test_pod_scc_deployment_mr"),
            pytest.param({"deployment_name": MR_POSTGRES_DEPLOYMENT_NAME_STR}, id="test_pod_scc_deployment_postgres"),
        ],
        indirect=["pod_model_registry_ns"],
    )
    @pytest.mark.tier2
    def test_model_registry_pod_security_context_validation(
        self: Self,
        skip_if_not_valid_check: None,
        pod_model_registry_ns: Pod,
        model_registry_scc_namespace: dict[str, str],
    ):
        """
        Validate that model registry pod gets runAsUser/runAsGroup from openshift and the values matches namespace
        annotations
        """
        validate_pod_scc(pod=pod_model_registry_ns, model_registry_scc_namespace=model_registry_scc_namespace)
