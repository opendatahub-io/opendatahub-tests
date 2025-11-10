from typing import Generator

import pytest
import requests
from simple_logger.logger import get_logger
from utilities.plugins.constant import RestHeader, OpenAIEnpoints


from tests.model_serving.model_server.maas_billing.utils import (
    detect_scheme_via_llmisvc,
    host_from_ingress_domain,
    mint_token,
    llmis_name,
)

LOGGER = get_logger(name=__name__)
MODELS_INFO = OpenAIEnpoints.MODELS_INFO
CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS


@pytest.fixture(scope="session")
def request_session_http() -> Generator[requests.Session, None, None]:
    session = requests.Session()
    session.headers.update({"User-Agent": "odh-maas-billing-tests/1"})
    session.verify = False
    yield session
    session.close()


@pytest.fixture(scope="class")
def minted_token(request_session_http, base_url: str, current_client_token: str) -> str:
    """Mint a MaaS token once per test class and reuse it."""
    resp, body = mint_token(
        base_url=base_url,
        oc_user_token=current_client_token,
        minutes=30,
        http_session=request_session_http,
    )
    LOGGER.info("Mint token response status=%s", resp.status_code)
    assert resp.status_code in (200, 201), f"mint failed: {resp.status_code} {resp.text[:200]}"
    token = body.get("token", "")
    assert isinstance(token, str) and len(token) > 10, f"no usable token in response: {body}"
    LOGGER.info(f"Minted MaaS token len={len(token)}")
    return token


@pytest.fixture(scope="module")
def base_url(admin_client) -> str:
    scheme = detect_scheme_via_llmisvc(client=admin_client)
    host = host_from_ingress_domain(client=admin_client)
    return f"{scheme}://{host}/maas-api"


@pytest.fixture(scope="session")
def model_url(admin_client) -> str:
    """
    MODEL_URL:http(s)://<host>/llm/<deployment>/v1/chat/completions
    """
    scheme = detect_scheme_via_llmisvc(client=admin_client)
    host = host_from_ingress_domain(client=admin_client)
    deployment = llmis_name(client=admin_client)
    return f"{scheme}://{host}/llm/{deployment}{CHAT_COMPLETIONS}"


@pytest.fixture
def maas_headers(minted_token: str) -> dict:
    """Common headers for MaaS API calls."""
    return {"Authorization": f"Bearer {minted_token}", **RestHeader.HEADERS}


@pytest.fixture
def maas_models(
    request_session_http: requests.Session,
    base_url: str,
    maas_headers: dict,
):
    """
    Call /v1/models once and return the list of models.

    """
    models_url = f"{base_url}{MODELS_INFO}"
    resp = request_session_http.get(models_url, headers=maas_headers, timeout=60)

    assert resp.status_code == 200, f"/v1/models failed: {resp.status_code} {resp.text[:200]}"

    models = resp.json().get("data", [])
    assert models, "no models available"
    return models
