"""Unit checks for architecture-aware model-serving configuration."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tests.model_serving.model_server import conftest as model_server_fixtures
from tests.model_serving.model_server.conftest import arch_onnx_s3_path
from tests.model_serving.model_server.kserve.authentication.conftest import auth_model_config
from tests.model_serving.model_server.runtime_registry import get_runtime_profile, resolve_cluster_arch
from utilities.constants import ModelFormat
from utilities.image_constants import SharedImages


def test_runtime_profiles_use_compatible_model_cars() -> None:
    """Given each architecture, select a compatible ONNX model car."""
    assert get_runtime_profile("amd64").model_car_image == SharedImages.MODELCAR_MNIST_8_1
    assert get_runtime_profile("amd64").model_car_format == ModelFormat.OPENVINO
    assert get_runtime_profile("arm64").model_car_image == SharedImages.MLSERVER_ONNX
    assert get_runtime_profile("arm64").model_car_format == ModelFormat.ONNX
    assert get_runtime_profile("arm64", "openvino_ir") is None


@pytest.mark.parametrize(
    ("configured", "detected", "expected"),
    [("auto", "amd64", "amd64"), ("auto", "arm64", "arm64"), ("arm64", None, "arm64")],
)
def test_resolve_cluster_arch(configured: str, detected: str | None, expected: str) -> None:
    """Given a valid override or detection result, resolve the architecture."""
    assert resolve_cluster_arch(configured, detected) == expected


@pytest.mark.parametrize(("configured", "detected"), [("auto", None), ("auto", "s390x"), ("bogus", "amd64")])
def test_resolve_cluster_arch_rejects_unknown(configured: str, detected: str | None) -> None:
    """Given no supported architecture, fail rather than guessing AMD64."""
    with pytest.raises(ValueError, match="architecture"):
        resolve_cluster_arch(configured, detected)


@pytest.mark.parametrize(
    ("arch", "path", "input_name"),
    [("amd64", "test-dir", "Input3"), ("arm64", "mlserver/model_repository/resnet-50-onnx", "pixel_values")],
)
def test_auth_model_matches_architecture(arch: str, path: str, input_name: str) -> None:
    """Select a runtime-compatible S3 model and request for auth feature tests."""
    model = auth_model_config.__wrapped__(arch)
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
        param={"runtime-name": "test-runtime"},
        node=SimpleNamespace(get_closest_marker=lambda name: True if name == "arch_runtime" else None),
        getfixturevalue=lambda name: arch,
    )
    runtime = model_server_fixtures.ovms_kserve_serving_runtime.__wrapped__(
        request, object(), SimpleNamespace(name="test-namespace")
    )
    next(runtime)
    with pytest.raises(StopIteration):
        next(runtime)
    assert runtime_kwargs["template_name"] == get_runtime_profile(arch).template
