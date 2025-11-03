from __future__ import annotations

from typing import Generator

import pytest
import requests

from tests.model_serving.model_server.maas_billing.utils import choose_scheme_via_gateway, host_from_ingress_domain


@pytest.fixture(scope="session")
def request_session_http() -> Generator[requests.Session, None, None]:
    s = requests.Session()
    s.verify = False
    s.headers.update({"User-Agent": "odh-maas-billing-tests/1"})
    try:
        yield s
    finally:
        s.close()


@pytest.fixture(scope="session", name="maas_user_token_for_api_calls")
def maas_user_token_for_api_calls(current_client_token: str) -> str:
    assert current_client_token, "Empty token from current_client_token()"
    return current_client_token


@pytest.fixture(scope="module")
def base_url(admin_client) -> str:
    scheme = choose_scheme_via_gateway(client=admin_client)
    host = host_from_ingress_domain(client=admin_client)
    return f"{scheme}://{host}/maas-api"
