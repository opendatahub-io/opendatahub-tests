from typing import Generator

import pytest
import requests
from simple_logger.logger import get_logger

from tests.model_serving.model_server.maas_billing.utils import (
    detect_scheme_via_llmisvc,
    host_from_ingress_domain,
    mint_token,
)

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session")
def request_session_http() -> Generator[requests.Session, None, None]:
    session = requests.Session()
    session.verify = False
    session.headers.update({"User-Agent": "odh-maas-billing-tests/1"})
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
    scheme = detect_scheme_via_llmisvc(client=admin_client, namespace="llm")
    host = host_from_ingress_domain(client=admin_client)
    return f"{scheme}://{host}/maas-api"
