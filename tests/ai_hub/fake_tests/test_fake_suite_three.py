import pytest

pytestmark = [pytest.mark.fake]


class TestFakeSuiteThree:
    def test_fails_key_error(self):
        """Fake test that fails with a key error."""
        data: dict = {"existing_key": "value"}
        _ = data["missing_key"]
