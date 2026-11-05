import pytest
from timeout_sampler import TimeoutExpiredError

from utilities.minio import wait_for_minio_deployment


def test_wait_for_minio_deployment_collects_diagnostics_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostics = []

    class FailingDeployment:
        namespace = "test-guardrails"
        name = "minio"

        def wait_for_replicas(self, **_: object) -> None:
            raise TimeoutExpiredError("MinIO did not become ready")

    monkeypatch.setattr(
        "utilities.minio.log_minio_diagnostics",
        lambda **kwargs: diagnostics.append(kwargs),
    )

    with pytest.raises(TimeoutExpiredError, match="MinIO did not become ready"):
        wait_for_minio_deployment(deployment=FailingDeployment())

    assert diagnostics == [{"namespace": "test-guardrails", "name": "minio"}]
