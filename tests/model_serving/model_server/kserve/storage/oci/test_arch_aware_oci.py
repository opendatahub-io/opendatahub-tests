"""Arch-aware OCI Model Car inference test.

Demonstrates the arch-runtime-selection pattern: same test logic runs with
OVMS on x86_64 and MLServer on ARM64, resolved automatically from cluster
node architecture.
"""

import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import KServeDeploymentType, ModelCarImage, ModelFormat, Protocols
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG


@pytest.mark.arch_runtime
@pytest.mark.tier1
@pytest.mark.parametrize(
    "unprivileged_model_namespace, arch_runtime_profile",
    [
        pytest.param(
            {"name": f"{ModelFormat.ONNX}-arch-model-car"},
            "onnx",
        ),
    ],
    indirect=True,
)
class TestArchAwareModelCar:
    """Validate KServe model serving using OCI Model Car with arch-aware runtime.

    On x86_64 clusters: uses OVMS runtime (kserve-ovms template)
    On ARM64 clusters: uses MLServer runtime (mlserver-runtime-template)

    The test logic is identical — only the runtime differs based on architecture.
    """

    @pytest.fixture(scope="class")
    def arch_model_car_isvc(
        self,
        arch_serving_runtime,
        unprivileged_client,
        unprivileged_model_namespace,
    ):
        from utilities.inference_utils import create_isvc

        with create_isvc(
            client=unprivileged_client,
            name="arch-model-car-raw",
            namespace=unprivileged_model_namespace.name,
            runtime=arch_serving_runtime.name,
            storage_uri=ModelCarImage.MNIST_8_1,
            model_format=arch_serving_runtime.instance.spec.supportedModelFormats[0].name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            external_route=True,
            wait_for_predictor_pods=False,
        ) as isvc:
            yield isvc

    def test_arch_model_car_no_restarts(self, arch_model_car_isvc):
        """Verify that model pod doesn't experience excessive restarts."""
        from utilities.infra import get_pods_by_isvc_label

        pod = get_pods_by_isvc_label(
            client=arch_model_car_isvc.client,
            isvc=arch_model_car_isvc,
        )[0]
        restarted_containers = [
            container.name for container in pod.instance.status.containerStatuses if container.restartCount > 2
        ]
        assert not restarted_containers, f"Containers {restarted_containers} restarted"

    def test_arch_model_car_inference_rest(self, arch_model_car_isvc):
        """Verify model inference via REST — runtime resolved per cluster arch."""
        verify_inference_response(
            inference_service=arch_model_car_isvc,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    def test_arch_model_car_status_loaded(self, arch_model_car_isvc):
        """Verify model status is Loaded and UpToDate."""
        model_status = arch_model_car_isvc.instance.status.modelStatus
        assert model_status.states.activeModelState == "Loaded"
        assert model_status.states.targetModelState == "Loaded"
        assert model_status.transitionStatus == "UpToDate"
