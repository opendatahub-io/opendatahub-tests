import pytest
from typing import Self

from simple_logger.logger import get_logger
from _pytest.fixtures import FixtureRequest

from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.deployment import Deployment
from tests.model_registry.scc.constants import MODEL_CATALOG_STR
from tests.model_registry.scc.utils import (
    get_uid_from_namespace,
    validate_pod_security_context,
    KEYS_TO_VALIDATE,
    validate_containers_pod_security_context,
    get_pod_by_deployment_name,
    validate_deployment_scc,
    validate_pod_scc,
)
from tests.model_registry.constants import MODEL_DICT, MR_INSTANCE_NAME, MODEL_REGISTRY_POD_FILTER

from kubernetes.dynamic import DynamicClient

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
)
@pytest.mark.custom_namespace
class TestModelCatalogSecurityContextValidation:
    @pytest.mark.parametrize(
        "deployment_model_registry_ns",
        [
            pytest.param({"deployment_name": MODEL_CATALOG_STR}),
            pytest.param({"deployment_name": f"{MODEL_CATALOG_STR}-postgres"}),
        ],
        indirect=["deployment_model_registry_ns"],
    )
    @pytest.mark.sanity
    def test_model_catalog_deployment_security_context_validation(
        self: Self,
        deployment_model_registry_ns: Deployment,
    ):
        """
        Validate that model catalog deployment does not set runAsUser/runAsGroup
        """
        validate_deployment_scc(deployment=deployment_model_registry_ns)

    @pytest.mark.parametrize(
        "pod_model_registry_ns",
        [
            pytest.param({"deployment_name": MODEL_CATALOG_STR}),
            pytest.param({"deployment_name": f"{MODEL_CATALOG_STR}-postgres"}),
        ],
        indirect=["pod_model_registry_ns"],
    )
    @pytest.mark.sanity
    def test_model_catalog_pod_security_context_validation(
        self: Self,
        pod_model_registry_ns: Pod,
        model_registry_scc_namespace: dict[str, str],
    ):
        """
        Validate that model catalog pod gets runAsUser/runAsGroup from openshift and the values matches namespace
        annotations
        """
        validate_pod_scc(pod=pod_model_registry_ns, model_registry_scc_namespace=model_registry_scc_namespace)