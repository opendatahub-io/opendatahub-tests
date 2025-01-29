import schemathesis
import pytest
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)

schema = schemathesis.from_pytest_fixture("generated_schema")


# TODO: This is a Stateless test due to how the openAPI spec is currently defined in upstream.
# Once it is updated to support Stateful testing of the API we can enable this to run every time,
# but for now having it run manually to check the existing failures is more than enough.
@pytest.mark.skip(reason="Only run manually for now")
@schema.parametrize()
def test_mr_api(case, current_client_token):
    case.headers["Authorization"] = f"Bearer {current_client_token}"
    case.headers["Content-Type"] = "application/json"
    case.call_and_validate(verify=False)
