from typing import Generator

import pytest
import requests

from tests.model_serving.model_server.maas_billing.utils import choose_scheme_via_gateway, host_from_ingress_domain


@pytest.fixture(scope="session")
def request_session_http() -> Generator[requests.Session, None, None]:
    session = requests.Session()
    session.verify = False
    session.headers.update({"User-Agent": "odh-maas-billing-tests/1"})
    yield session
    session.close()


@pytest.fixture(scope="module")
def base_url(admin_client) -> str:
    scheme = choose_scheme_via_gateway(client=admin_client)
    host = host_from_ingress_domain(client=admin_client)
    return f"{scheme}://{host}/maas-api"
