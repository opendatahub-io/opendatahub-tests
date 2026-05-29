import pytest

pytestmark = [pytest.mark.fake]


class TestFakeSuiteTwo:
    def test_fails_with_value_error(self):
        """Fake test that fails with a value error."""
        raise ValueError("Intentional value error for must-gather testing")

    def test_fails_comparison(self):
        """Fake test that fails with a comparison mismatch."""
        expected = "expected_value"
        actual = "wrong_value"
        assert actual == expected, f"Expected '{expected}', got '{actual}'"
