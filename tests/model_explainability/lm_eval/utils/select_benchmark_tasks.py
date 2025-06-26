#!/usr/bin/env python3
"""
Select better tasks for timing benchmark
Choose popular/common tasks that are representative
"""

import json
from pathlib import Path


def load_configmap_tasks():
    """Load tasks from the extracted ConfigMap"""
    configmap_path = Path(__file__).parent.parent / "task_lists" / "configmap_tasks.json"
    try:
        with open(configmap_path, "r") as f:
            data = json.load(f)
        return data["tasks"]
    except FileNotFoundError:
        print(f"❌ {configmap_path} not found. Run parse_configmap.py first.")
        raise


def select_timing_benchmark_tasks(all_tasks, count=10):
    """Select representative tasks for timing benchmark"""
    priority_patterns = [
        "arc_challenge",
        "arc_easy",
        "hellaswag",
        "mmlu",
        "winogrande",
        "truthfulqa",
        "gsm8k",
        "humaneval",
        "lambada",
        "piqa",
        "boolq",
        "copa",
        "wic",
        "rte",
        "cb",
    ]
    selected_tasks = []
    for pattern in priority_patterns:
        matching_tasks = [task for task in all_tasks if pattern in task.lower()]
        if matching_tasks:
            best_match = min(matching_tasks, key=len)
            if best_match not in selected_tasks:
                selected_tasks.append(best_match)
                print(f"✅ Found: {best_match}")
        if len(selected_tasks) >= count:
            break
    if len(selected_tasks) < count:
        print(f"\n🔄 Need {count - len(selected_tasks)} more tasks, adding diverse ones...")
        additional_patterns = ["squad", "drop", "race", "openbookqa", "commonsenseqa", "anli", "xnli"]
        for pattern in additional_patterns:
            matching_tasks = [task for task in all_tasks if pattern in task.lower()]
            if matching_tasks:
                best_match = min(matching_tasks, key=len)
                if best_match not in selected_tasks:
                    selected_tasks.append(best_match)
                    print(f"➕ Added: {best_match}")
                if len(selected_tasks) >= count:
                    break
    if len(selected_tasks) < count:
        remaining_tasks = [task for task in all_tasks if task not in selected_tasks]
        remaining_tasks.sort(key=len)
        for task in remaining_tasks[: count - len(selected_tasks)]:
            selected_tasks.append(task)
            print(f"📝 Filled: {task}")
    return selected_tasks[:count]


def save_benchmark_tasks(tasks):
    """Save the selected benchmark tasks"""
    output_dir = Path(__file__).parent.parent / "task_lists"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "selected_timing_benchmark.py", "w") as f:
        f.write("# Selected tasks for timing benchmark\n")
        f.write("# These are popular/representative LMEval tasks\n\n")
        f.write("TIMING_BENCHMARK_TASKS = [\n")
        for task in tasks:
            f.write(f'    "{task}",\n')
        f.write("]\n\n")
        f.write(f"# Total: {len(tasks)} tasks\n")
        f.write("# Estimated time: ~25min × 10 = 4+ hours for 1% sampling\n")
    with open(output_dir / "timing_benchmark.json", "w") as f:
        json.dump(
            {
                "purpose": "Timing benchmark for LMEval systematic testing",
                "sampling_percentage": "1%",
                "estimated_time_per_task": "25 minutes",
                "total_estimated_time": f"{len(tasks) * 25} minutes",
                "tasks": tasks,
            },
            f,
            indent=2,
        )
    print("\n💾 Saved benchmark task selection:")
    print(f"   🐍 Python: {output_dir / 'selected_timing_benchmark.py'}")
    print(f"   📄 JSON: {output_dir / 'timing_benchmark.json'}")


def generate_test_code(tasks):
    """Generate pytest code for the timing benchmark"""
    test_code = """# Timing benchmark test for LMEval systematic testing
import pytest
import time
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from tests.model_explainability.lm_eval.utils import verify_lmevaljob_running, save_pod_logs
from utilities.constants import Timeout
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)

@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_namespace, task_name",
    [
"""
    for task in tasks:
        test_code += f'''
        pytest.param(
            {{"name": "test-lmeval-timing-{task.replace("_", "-")}"}},
            "{task}",
            id="{task}"
        ),'''
    test_code += '''
    ],
    indirect=["model_namespace"],
)
def test_lmeval_timing_benchmark(admin_client: DynamicClient, model_namespace, task_name: str, lmevaljob_vllm_emulator: LMEvalJob, lmevaljob_vllm_emulator_pod: Pod):
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
'''
    output_dir = Path(__file__).parent.parent
    with open(output_dir / "test_lm_eval_timing.py", "w") as f:
        f.write(test_code)
    print(f"   🧪 Test code: {output_dir / 'test_lm_eval_timing.py'}")


def main():
    print("🎯 Selecting optimal tasks for timing benchmark")
    print("=" * 60)
    all_tasks = load_configmap_tasks()
    print(f"📋 Loaded {len(all_tasks)} total tasks from ConfigMap")
    print("\n🔍 Selecting 10 representative tasks...")
    selected_tasks = select_timing_benchmark_tasks(all_tasks, count=10)
    print(f"\n✅ Selected {len(selected_tasks)} tasks for timing benchmark:")
    for i, task in enumerate(selected_tasks, 1):
        print(f"   {i:2d}. {task}")
    save_benchmark_tasks(selected_tasks)
    generate_test_code(selected_tasks)
    print("\n📊 Next Steps:")
    print("   1. Review tests/model_explainability/lm_eval/task_lists/selected_timing_benchmark.py")
    print(f"   2. Run timing test: ~{len(selected_tasks) * 25} minutes estimated")
    print("   3. Adjust sampling % based on results")
    print("   4. Scale to all tasks")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
