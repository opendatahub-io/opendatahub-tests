import pytest

from tests.model_serving.model_server.llmd.llmd_configs import TinyLlamaS3ConnectionConfig
from tests.model_serving.model_server.llmd.utils import (
    assert_llmisvc_connection_cleared,
    ns_from_file,
    wait_for_llmisvc_connection_cleared,
)
from tests.model_serving.model_server.utils import remove_connection_annotations
from utilities.constants import Timeout
from utilities.resources.llm_inference_service import LLMInferenceService

pytestmark = [pytest.mark.tier1]

NAMESPACE = ns_from_file(file=__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [pytest.param({"name": NAMESPACE}, TinyLlamaS3ConnectionConfig, id="s3-connection-remove")],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestLlmdConnectionRemove:
    """Manual 1.9: removing the ConnectionsAPI annotation clears injected LLMISVC fields."""

    def test_llmd_connection_remove_clears_model(self, llmisvc: LLMInferenceService):
        """Test steps:

        1. Assert the S3 connection was injected on CREATE.
        2. Remove the ConnectionsAPI annotations, exercising the UPDATE-remove path.
        3. Assert the injected fields are cleared and the service is no longer Ready.
        """
        llmisvc.connections_config.verify_injection(llmisvc=llmisvc)

        remove_connection_annotations(resource=llmisvc)
        wait_for_llmisvc_connection_cleared(llmisvc=llmisvc, timeout=Timeout.TIMEOUT_2MIN)
        assert_llmisvc_connection_cleared(llmisvc=llmisvc)

        llmisvc.wait_for_condition(condition="Ready", status="False", timeout=Timeout.TIMEOUT_2MIN)
