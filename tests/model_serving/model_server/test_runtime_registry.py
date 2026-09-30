"""Unit checks for architecture-aware model-serving configuration."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

# These unit tests exercise fixture factories without creating cluster resources.
from tests.model_serving.model_server import conftest as model_server_fixtures  # noqa: NIC001
from tests.model_serving.model_server.kserve.authentication import conftest as auth_fixtures  # noqa: NIC001
from tests.model_serving.model_server.kserve.authentication.conftest import auth_model_config  # noqa: NIC001
from tests.model_serving.model_server.kserve.storage.pvc import conftest as pvc_fixtures  # noqa: NIC001
from tests.model_serving.model_server.runtime_registry import get_runtime_profile, resolve_cluster_arch
from tests.model_serving.model_server.utils import arch_onnx_s3_path
from utilities.constants import ModelFormat
from utilities.image_constants import SharedImages


def test_runtime_profiles_use_compatible_model_cars() -> None:
    """Given each architecture, select a compatible ONNX model car."""
    assert get_runtime_profile("amd64").model_car_image == SharedImages.MODELCAR_MNIST_8_1
    assert get_runtime_profile("amd64").model_car_format == ModelFormat.OPENVINO
    assert get_runtime_profile("arm64").model_car_image == SharedImages.MLSERVER_ONNX
    assert get_runtime_profile("arm64").model_car_format == ModelFormat.ONNX


@pytest.mark.parametrize(
    ("configured", "detected", "expected"),
    [("auto", "amd64", "amd64"), ("auto", "arm64", "arm64"), ("arm64", None, "arm64")],
)
def test_resolve_cluster_arch(configured: str, detected: str | None, expected: str) -> None:
    """Given a valid override or detection result, resolve the architecture."""
    assert resolve_cluster_arch(configured=configured, detected=detected) == expected


@pytest.mark.parametrize(("configured", "detected"), [("auto", None), ("auto", "s390x"), ("bogus", "amd64")])
def test_resolve_cluster_arch_rejects_unknown(configured: str, detected: str | None) -> None:
    """Given no supported architecture, fail rather than guessing AMD64."""
    with pytest.raises(ValueError, match="architecture"):
        resolve_cluster_arch(configured=configured, detected=detected)


@pytest.mark.parametrize(
    ("arch", "path", "input_name"),
    [("amd64", "test-dir", "Input3"), ("arm64", "mlserver/model_repository/resnet-50-onnx", "pixel_values")],
)
def test_auth_model_matches_architecture(arch: str, path: str, input_name: str) -> None:
    """Select a runtime-compatible S3 model and request for auth feature tests."""
    model = auth_model_config.__wrapped__(cluster_arch=arch)
    assert model["path"] == path
    assert arch_onnx_s3_path(arch) == path
    assert model["template"] == get_runtime_profile(arch).template
    query = (
        model["request"]["inputs"]
        if arch == "arm64"
        else model["inference"]["default_query_model"]["infer"]["query_input"]
    )
    assert query[0]["name"] == input_name
    if arch == "arm64":
        assert len(query[0]["data"]) == 3 * 224 * 224


@pytest.mark.parametrize("arch", ["amd64", "arm64"])
def test_arch_marked_runtime_uses_matching_template(arch: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An opted-in MNIST feature test selects the runtime supported on its architecture."""
    runtime_kwargs = {}

    @contextmanager
    def fake_runtime(**kwargs):
        runtime_kwargs.update(kwargs)
        yield SimpleNamespace(name="test-runtime")

    monkeypatch.setattr(model_server_fixtures, "ServingRuntimeFromTemplate", fake_runtime)
    request = SimpleNamespace(
        param={
            "runtime-name": "test-runtime",
            "runtime-image": "custom-image",
            "model-format": "custom-format",
            "supported-model-formats": {"custom-format": "1"},
        },
        node=SimpleNamespace(get_closest_marker=lambda name: True if name == "arch_runtime" else None),
        getfixturevalue=lambda argname: arch,
    )
    runtime = model_server_fixtures.ovms_kserve_serving_runtime.__wrapped__(
        request=request,
        unprivileged_client=object(),
        unprivileged_model_namespace=SimpleNamespace(name="test-namespace"),
    )
    next(runtime)
    with pytest.raises(StopIteration):
        next(runtime)
    assert runtime_kwargs["template_name"] == get_runtime_profile(arch).template
    if arch == "amd64":
        assert runtime_kwargs["runtime_image"] == "custom-image"
        assert runtime_kwargs["model_format_name"] == "custom-format"
        assert runtime_kwargs["supported_model_formats"] == {"custom-format": "1"}
    else:
        assert "runtime_image" not in runtime_kwargs
        assert "model_format_name" not in runtime_kwargs
        assert "supported_model_formats" not in runtime_kwargs


@pytest.mark.parametrize("arch", ["amd64", "arm64"])
@pytest.mark.parametrize("second_model", [False, True])
def test_auth_storage_preserves_x86_parameters(arch: str, second_model: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve x86 model parameters and fetch the models secret only for ARM."""
    captured = {}
    requested = []

    @contextmanager
    def fake_isvc(**kwargs):
        captured.update(kwargs)
        yield SimpleNamespace(name="isvc")

    def getfixturevalue(name):
        requested.append(name)
        assert arch == "arm64"
        assert name == "models_endpoint_s3_secret"
        return SimpleNamespace(name="models-secret")

    monkeypatch.setattr(auth_fixtures, "create_isvc", fake_isvc)
    runtime = SimpleNamespace(
        name="runtime",
        instance=SimpleNamespace(spec=SimpleNamespace(supportedModelFormats=[SimpleNamespace(name="original-format")])),
    )
    kwargs = {
        "request": SimpleNamespace(param={"model-dir": "custom-model"}, getfixturevalue=getfixturevalue),
        "unprivileged_client": object(),
        "unprivileged_model_namespace": SimpleNamespace(name="namespace"),
        "http_s3_ovms_serving_runtime": runtime,
        "ci_s3_bucket_name": "ci-bucket",
        "ci_endpoint_s3_secret": SimpleNamespace(name="ci-secret"),
        "cluster_arch": arch,
        "auth_model_config": auth_model_config.__wrapped__(arch),
        "model_service_account_2" if second_model else "model_service_account": SimpleNamespace(name="sa"),
    }
    fixture = (
        auth_fixtures.http_s3_ovms_raw_inference_service_2
        if second_model
        else auth_fixtures.http_s3_ovms_raw_inference_service
    )
    assert len(list(fixture.__wrapped__(**kwargs))) == 1
    assert captured["storage_path"] == f"/{arch_onnx_s3_path(arch) if arch == 'arm64' else 'custom-model'}/"
    assert captured["storage_key"] == ("models-secret" if arch == "arm64" else "ci-secret")
    assert captured["model_format"] == (ModelFormat.ONNX if arch == "arm64" else "original-format")
    assert requested == (["models_endpoint_s3_secret"] if arch == "arm64" else [])


def test_x86_pvc_download_preserves_requested_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parameterized x86 PVC test must download its requested model, not a hardcoded MNIST path."""
    captured = {}
    monkeypatch.setattr(pvc_fixtures, "download_model_data", lambda **kwargs: captured.update(kwargs))
    pvc_fixtures.ci_bucket_downloaded_model_data.__wrapped__(
        request=SimpleNamespace(param={"model-dir": "custom-model"}),
        admin_client=object(),
        aws_access_key_id="test",
        aws_secret_access_key="test",  # pragma: allowlist secret
        unprivileged_model_namespace=SimpleNamespace(name="namespace"),
        model_pvc=SimpleNamespace(name="pvc"),
        ci_s3_bucket_name="ci-bucket",
        ci_s3_bucket_endpoint="endpoint",
        ci_s3_bucket_region="region",
        cluster_arch="amd64",
    )
    assert captured["model_path"] == "custom-model"
    assert captured["bucket_name"] == "ci-bucket"
