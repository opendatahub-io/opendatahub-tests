import pytest
from simple_logger.logger import get_logger
from utilities.plugins.constant import OpenAIEnpoints
from tests.model_serving.model_server.maas_billing.utils import (
    verify_chat_completions,
)

LOGGER = get_logger(name=__name__)

MODELS_INFO = OpenAIEnpoints.MODELS_INFO
CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS

ACTORS = [
    {"type": "admin"},
    {"type": "free"},
    {"type": "premium"},
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {
                "name": "llm",
                "modelmesh-enabled": False,
            },
            id="maas-billing-namespace",
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("maas_free_group", "maas_premium_group")
@pytest.mark.parametrize(
    "ocp_token_for_actor",
    ACTORS,
    indirect=True,
)
class TestMaasRBACE2E:
    """
    For each actor (admin / free / premium) verify:
    - can mint a MaaS token
    - can list models
    - can call /v1/chat/completions
    """

    @pytest.mark.sanity
    def test_mint_token_for_actors(
        self,
        ocp_token_for_actor,
        maas_token_for_actor: str,
    ) -> None:
        LOGGER.info(f"MaaS RBAC: using already minted MaaS token length={len(maas_token_for_actor)}")

    @pytest.mark.sanity
    def test_models_visible_for_actors(
        self,
        model_url: str,
        maas_models_response_for_actor,
    ) -> None:
        """Use fixture for /v1/models response."""
        response = maas_models_response_for_actor
        models = response.json().get("data", [])
        assert isinstance(models, list) and models, "no models returned from /v1/models"

    @pytest.mark.sanity
    def test_chat_completions_for_actors(
        self,
        request_session_http,
        model_url: str,
        maas_headers_for_actor: dict,
        maas_models_response_for_actor,
        ocp_token_for_actor,
    ) -> None:
        """
        Reuse the models fixture instead of duplicating the /v1/models logic,
        then call /v1/chat/completions with the first model id using the
        common verify_chat_completions helper.
        """
        models_response = maas_models_response_for_actor
        models_list = models_response.json().get("data", [])
        assert models_list, "no models returned from /v1/models"

        verify_chat_completions(
            request_session_http=request_session_http,
            model_url=model_url,
            headers=maas_headers_for_actor,
            models_list=models_list,
            prompt_text="Hello",
            max_tokens=16,
            request_timeout_seconds=60,
            log_prefix="MaaS RBAC",
        )
