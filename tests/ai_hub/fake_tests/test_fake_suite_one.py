import pytest

pytestmark = [pytest.mark.fake]


class TestFakeSuiteOne:
    def test_always_fails_assertion(self):
        """Fake test that always fails with an assertion error."""
        assert False, "Intentional failure for must-gather testing"

    def test_always_fails_exception(self):
        """Fake test that always fails with a runtime error."""
        raise RuntimeError("Intentional exception for must-gather testing")
