import pytest


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-fairness-loan-model", "modelmesh-enabled": True},
        )
    ],
    indirect=True,
)
class TestFairnessMetrics:
    def test_dummy(self):
        pass
