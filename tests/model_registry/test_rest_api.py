import schemathesis
import pytest
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)

schema = schemathesis.pytest.from_fixture("generated_schema")


@pytest.mark.skip(reason="Only run manually for now")
@pytest.mark.fuzzer
@schema.parametrize()
def test_mr_api_stateless(case, current_client_token):
    case.headers["Authorization"] = f"Bearer {current_client_token}"
    case.headers["Content-Type"] = "application/json"
    case.call_and_validate(verify=False)
