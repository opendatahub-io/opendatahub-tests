import pytest
from simple_logger.logger import get_logger

from utilities.constants import DscComponents

LOGGER = get_logger(name=__name__)
TIMEOUT = 6 * 60


pytestmark = pytest.mark.smoke


@pytest.mark.usefixtures("managed_modelmesh_kserve_in_dsc")
class TestKserveModelmeshCoexist:
    def test_model_mesh_state_in_dsc(self, dsc_resource):
        """Verify ModelMesh Serving state in DSC is managed when kserve is enabled"""
        LOGGER.info(f"Verify {DscComponents.MODELMESHSERVING} state in DSC is {DscComponents.ManagementState.MANAGED}")
        dsc_resource.wait_for_condition(
            condition=DscComponents.ConditionType.MODEL_MESH_SERVING_READY,
            status="True",
            timeout=TIMEOUT,
        )

    def test_kserve_state_in_dsc(self, dsc_resource):
        """Verify kserve Serving state in DSC is managed when ModelMesh is enabled"""
        LOGGER.info(f"Verify {DscComponents.KSERVE} state in DSC is {DscComponents.ManagementState.MANAGED}")
        dsc_resource.wait_for_condition(
            condition=DscComponents.ConditionType.KSERVE_READY,
            status="True",
            timeout=TIMEOUT,
        )
