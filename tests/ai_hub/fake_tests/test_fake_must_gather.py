import pytest

pytestmark = [pytest.mark.fake]


class TestFakeMustGather:
    def test_always_fails(self):
        """Fake test that always fails to trigger must-gather collection."""
        assert False, "Intentional failure for must-gather testing"
