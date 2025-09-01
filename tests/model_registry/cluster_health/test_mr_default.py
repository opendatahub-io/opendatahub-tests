import pytest
from utilities.infra import get_data_science_cluster
from utilities.constants import DscComponents
from kubernetes.dynamic import DynamicClient

from ocp_resources.namespace import Namespace

from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger


LOGGER = get_logger(name=__name__)


class TestMrDefault:
    @pytest.mark.cluster_health
    def test_mr_default(self, admin_client: DynamicClient):
        """
        Verify MODELREGISTRY managementState is MANAGED in DSC,
        the namespace is Active, and the MR ready condition in DSC is True.
        """
        dsc_resource = get_data_science_cluster(client=admin_client)
        assert (
            dsc_resource.instance.spec.components[DscComponents.MODELREGISTRY].managementState
            == DscComponents.ManagementState.MANAGED
        )

        namespace = Namespace(name=py_config["model_registry_namespace"], ensure_exists=True)
        assert namespace.instance.status.phase == Namespace.Status.ACTIVE

        for condition in dsc_resource.instance.status.conditions:
            if condition.type == DscComponents.COMPONENT_MAPPING[DscComponents.MODELREGISTRY]:
                LOGGER.info(f"MR ready in DSC: {condition.status}")
                assert condition.status == "True"
                break
        else:
            pytest.fail("MR not ready in DSC")
