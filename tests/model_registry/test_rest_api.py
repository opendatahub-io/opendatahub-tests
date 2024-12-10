import schemathesis
import pytest
from schemathesis.checks import ignored_auth
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)

schema = schemathesis.from_pytest_fixture("generate_schema")


@pytest.mark.skip(reason="Only run manually for now")
@schema.parametrize()
def test_mr_api(case, admin_client_token):
    case.headers["Authorization"] = f"Bearer {admin_client_token}"
    case.headers["Content-Type"] = "application/json"
    LOGGER.warning(case)
    response = case.call_and_validate(excluded_checks=(ignored_auth,), verify=False)
    LOGGER.warning(response)
    # case.validate_response(response, verify=False)
