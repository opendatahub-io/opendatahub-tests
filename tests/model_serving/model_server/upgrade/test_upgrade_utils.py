import pytest

# Import the infrastructure module first to match the normal test-suite import
# order and avoid the existing general/infra compatibility cycle.
import utilities.infra  # noqa: F401
from tests.model_serving.model_server.upgrade import utils


def test_load_isvc_upgrade_baseline_skips_when_pre_upgrade_baseline_is_missing(monkeypatch):
    def missing_baseline(**_kwargs):
        raise AssertionError("pre-upgrade baseline is unavailable")

    monkeypatch.setattr(utils, "load_baseline_from_configmap", missing_baseline)

    with pytest.raises(pytest.skip.Exception, match="pre-upgrade baseline is unavailable"):
        utils.load_isvc_upgrade_baseline(
            client=object(),
            namespace="upgrade-model-server",
            isvc_name="ovms-model-mesh",
        )
