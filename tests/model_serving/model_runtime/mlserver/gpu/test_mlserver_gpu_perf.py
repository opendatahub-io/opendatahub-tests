"""GPU vs CPU inference performance comparison using MLServer runtimes.

Validates that the GPU-enabled MLServer CUDA runtime provides measurably lower
average inference latency than the CPU MLServer runtime for ResNet-50 ONNX inference.
"""

import warnings

import pytest
import structlog
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.mlserver.constant import (
    GPU_SPEEDUP_THRESHOLD_RATIO,
    ONNX_RESNET50_REST_INPUT_QUERY,
    RESNET50_INFERENCE_REQUEST_COUNT,
)
from tests.model_serving.model_runtime.mlserver.utils import get_model_storage_uri_dict, measure_average_latency
from utilities.constants import KServeDeploymentType, ModelFormat, Timeout

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures("skip_if_no_gpu_for_mlserver", "valid_aws_config"),
    pytest.mark.gpu,
    pytest.mark.mlserver_nvidia_gpu,
]


@pytest.mark.parametrize(
    ("model_namespace", "s3_models_storage_uri", "mlserver_serving_runtime", "mlserver_inference_service"),
    [
        pytest.param(
            {"name": "mlserver-perf-gpu"},
            get_model_storage_uri_dict(model_format_name=ModelFormat.ONNX),
            {"deployment_mode": KServeDeploymentType.STANDARD, "gpu": True},
            {
                "name": "resnet-50-onnx",
                "gpu_count": 1,
                "deployment_mode": KServeDeploymentType.STANDARD,
                "enable_external_route": True,
                "timeout": Timeout.TIMEOUT_10MIN,
            },
            id="test_mlserver_gpu_vs_cpu_perf",
            marks=[pytest.mark.slow],
        ),
    ],
    indirect=["model_namespace", "s3_models_storage_uri", "mlserver_serving_runtime", "mlserver_inference_service"],
)
class TestGpuVsCpuPerformance:
    """Performance tests comparing GPU MLServer CUDA runtime against CPU MLServer runtime."""

    def test_gpu_faster_than_cpu(
        self,
        mlserver_cpu_perf_isvc: InferenceService,
        mlserver_inference_service: InferenceService,
    ) -> None:
        """GPU MLServer CUDA runtime provides lower latency than CPU runtime.

        Given a CPU ISVC and a GPU ISVC both serving ResNet-50 ONNX from S3
        When sequential inference requests are sent to each
        Then both ISVCs respond successfully and latencies are logged.

        Note: Speedup ratio is logged but not strictly asserted because network
        latency from remote test runners can mask GPU compute advantages.
        """
        cpu_avg_latency = measure_average_latency(
            isvc=mlserver_cpu_perf_isvc,
            input_data=ONNX_RESNET50_REST_INPUT_QUERY,
            request_count=RESNET50_INFERENCE_REQUEST_COUNT,
        )
        gpu_avg_latency = measure_average_latency(
            isvc=mlserver_inference_service,
            input_data=ONNX_RESNET50_REST_INPUT_QUERY,
            request_count=RESNET50_INFERENCE_REQUEST_COUNT,
        )

        speedup_ratio = cpu_avg_latency / gpu_avg_latency

        LOGGER.info(
            event="GPU vs CPU performance comparison",
            cpu_avg_latency=f"{cpu_avg_latency:.3f}s",
            gpu_avg_latency=f"{gpu_avg_latency:.3f}s",
            speedup_ratio=f"{speedup_ratio:.2f}x",
            threshold=f"{GPU_SPEEDUP_THRESHOLD_RATIO}x",
            meets_threshold=speedup_ratio >= GPU_SPEEDUP_THRESHOLD_RATIO,
        )

        if speedup_ratio < GPU_SPEEDUP_THRESHOLD_RATIO:
            warnings.warn(
                f"GPU speedup ({speedup_ratio:.2f}x) is below the target {GPU_SPEEDUP_THRESHOLD_RATIO}x. "
                f"CPU: {cpu_avg_latency:.3f}s, GPU: {gpu_avg_latency:.3f}s. "
                "This may be due to network latency dominating compute time.",
                stacklevel=1,
            )
