import pytest
import requests
from simple_logger.logger import get_logger

from tests.model_serving.model_server.maas_billing.utils import verify_chat_completions

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures("maas_unprivileged_model_namespace", "maas_controller_enabled_latest")
class TestMaasEndpoints:
    @pytest.mark.sanity
    def test_model(
        self,
        maas_models: list,
    ) -> None:
        """Verify /v1/models endpoint returns at least one model."""
        assert isinstance(maas_models, list)
        assert maas_models, "no models returned from /v1/models"

        first = maas_models[0]
        assert "id" in first, "model entry missing 'id'"

    @pytest.mark.sanity
    def test_chat_completions(
        self,
        request_session_http: requests.Session,
        model_url: str,
        maas_headers: dict,
        maas_models: list,
    ) -> None:
        """Verify /llm/<deployment>/v1/chat/completions responds to a simple prompt."""
        verify_chat_completions(
            request_session_http=request_session_http,
            model_url=model_url,
            headers=maas_headers,
            models_list=maas_models,
            log_prefix="MaaS Endpoint Test",
        )
