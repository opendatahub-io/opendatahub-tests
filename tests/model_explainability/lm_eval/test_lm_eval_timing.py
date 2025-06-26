import pytest
import time
from unittest.mock import MagicMock
from pathlib import Path
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from tests.model_explainability.lm_eval.utils import verify_lmevaljob_running, save_pod_logs
from utilities.constants import Timeout
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


# Mock LMEvalJob class for local testing
class LMEvalJob:
    """Mock LMEvalJob class for type hinting in skipped GPU tests"""

    def __init__(self):
        pass

    def update(self, task_list):
        pass


@pytest.mark.skip(reason="Requires GPU cluster, testing locally with mock")
@pytest.mark.parametrize(
    "model_namespace, task_name",
    [
        pytest.param({"name": "test-lmeval-timing-arc-challenge"}, "arc_challenge", id="arc_challenge"),
        pytest.param({"name": "test-lmeval-timing-arc-easy"}, "arc_easy", id="arc_easy"),
        pytest.param({"name": "test-lmeval-timing-hellaswag"}, "hellaswag", id="hellaswag"),
        pytest.param({"name": "test-lmeval-timing-mmlu"}, "mmlu", id="mmlu"),
        pytest.param({"name": "test-lmeval-timing-winogrande"}, "winogrande", id="winogrande"),
        pytest.param({"name": "test-lmeval-timing-truthfulqa"}, "truthfulqa", id="truthfulqa"),
        pytest.param({"name": "test-lmeval-timing-gsm8k"}, "gsm8k", id="gsm8k"),
        pytest.param({"name": "test-lmeval-timing-humaneval"}, "humaneval", id="humaneval"),
        pytest.param({"name": "test-lmeval-timing-lambada"}, "lambada", id="lambada"),
        pytest.param({"name": "test-lmeval-timing-piqa"}, "piqa", id="piqa"),
    ],
    indirect=["model_namespace"],
)
def test_lmeval_timing_benchmark(admin_client: DynamicClient, model_namespace, task_name: str,
                                 lmevaljob_vllm_emulator: LMEvalJob, lmevaljob_vllm_emulator_pod: Pod):
    """Timing benchmark test - measures execution time for representative tasks"""
    start_time = time.time()
    lmevaljob_vllm_emulator.update(task_list={"taskNames": [task_name], "sample_size": "1%"})
    verify_lmevaljob_running(client=admin_client, lmevaljob=lmevaljob_vllm_emulator)
    execution_time = time.time() - start_time
    log_file = save_pod_logs(
        pod=lmevaljob_vllm_emulator_pod,
        task_name=task_name,
        namespace=model_namespace.name,
    )
    LOGGER.info(f"Task {task_name} took {execution_time:.2f} seconds with 1% sampling. Logs saved to {log_file}")


def test_lmeval_timing_benchmark_mock():
    """Mock test to verify logging for a public task"""
    # Mock Pod object
    mock_pod = MagicMock()
    mock_pod.name = "test-pod-arc-easy"
    mock_pod.get_logs.return_value = "Mock arc_easy task output\nCompleted successfully"

    # Test parameters
    task_name = "arc_easy"
    namespace = "test-namespace"

    # Simulate test
    start_time = time.time()
    time.sleep(1)  # Simulate task execution
    execution_time = time.time() - start_time

    # Save logs
    log_file = save_pod_logs(
        pod=mock_pod,
        task_name=task_name,
        namespace=namespace,
    )

    LOGGER.info(f"Task {task_name} took {execution_time:.2f} seconds with mock execution. Logs saved to {log_file}")

    # Verify log file
    assert Path(log_file).exists(), f"Log file {log_file} was not created"
    with open(log_file, 'r') as f:
        content = f.read()
    assert "Mock arc_easy task output" in content, "Log file content does not match expected output"
    assert Path(log_file) == Path(
        __file__).parent / "task_lists" / "lmeval_task_logs" / namespace / f"{task_name}_logs.txt", f"Log file saved to incorrect path: {log_file}"