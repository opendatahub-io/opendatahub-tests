#!/usr/bin/env python3
"""
Parse trustyai-lmeval-tasks.yaml ConfigMap from GitHub to extract task list
"""

import json
import re
import sys
from pathlib import Path

import requests
import yaml


def parse_configmap(
    github_url: str = "https://raw.githubusercontent.com/ruivieira/trustyai-service-operator/RHOAIENG-26883/config/configmaps/trustyai-lmeval-tasks.yaml",
):
    """Parse the ConfigMap from GitHub to extract LMEval tasks"""
    try:
        # Fetch YAML content from GitHub
        response = requests.get(github_url)
        response.raise_for_status()  # Raise exception for bad status codes
        configmap = yaml.safe_load(response.text)

        print(f"✅ Successfully loaded ConfigMap from: {github_url}")

        # Print structure to understand the format
        print("\n📋 ConfigMap Structure:")
        print(f"   Kind: {configmap.get('kind')}")
        print(f"   Metadata name: {configmap.get('metadata', {}).get('name')}")

        data_keys = list(configmap.get("data", {}).keys())
        print(f"   Data keys: {data_keys}")

        # Extract task information
        tasks = []
        task_details = []
        data = configmap.get("data", {})

        print(f"\n🔍 Found data keys: {list(data.keys())}")

        if "tasks" in data:
            tasks_json = data["tasks"]
            print(f"   📝 Tasks data type: {type(tasks_json)}")
            print(f"   📝 Tasks data preview: {tasks_json[:300]}...")

            try:
                # Parse the JSON string
                parsed_data = json.loads(tasks_json)
                print("   ✅ Successfully parsed JSON")
                print(f"   📊 Top-level keys: {list(parsed_data.keys())}")

                # Extract tasks from "lm-evaluation-harness" array
                if "lm-evaluation-harness" in parsed_data:
                    lm_eval_tasks = parsed_data["lm-evaluation-harness"]
                    print(f"   📋 Found {len(lm_eval_tasks)} lm-evaluation-harness tasks")

                    for task_obj in lm_eval_tasks:
                        if isinstance(task_obj, dict) and "name" in task_obj:
                            task_name = task_obj["name"]
                            task_desc = task_obj.get("description", "No description")
                            tasks.append(task_name)
                            task_details.append({"name": task_name, "description": task_desc})

                # Check for other task categories
                for key, value in parsed_data.items():
                    if key != "lm-evaluation-harness" and isinstance(value, list):
                        print(f"   📋 Found additional category '{key}' with {len(value)} tasks")
                        for task_obj in value:
                            if isinstance(task_obj, dict) and "name" in task_obj:
                                task_name = task_obj["name"]
                                task_desc = task_obj.get("description", "No description")
                                tasks.append(task_name)
                                task_details.append({"name": task_name, "description": task_desc, "category": key})

            except json.JSONDecodeError as e:
                print(f"   ❌ JSON parsing failed: {e}")
                # Fallback to regex extraction
                name_matches = re.findall(r'"name":\s*"([^"]+)"', tasks_json)
                tasks.extend(name_matches)
                print(f"   🔄 Fallback: extracted {len(name_matches)} task names using regex")

        # Clean and deduplicate tasks
        clean_tasks = [task.strip() for task in tasks if isinstance(task, str) and task.strip()]
        unique_tasks = sorted(set(clean_tasks))

        print(f"\n✅ Extracted {len(unique_tasks)} unique tasks from ConfigMap")

        # Show categories if found
        categories = {detail["category"] for detail in task_details if "category" in detail}
        if categories:
            print(f"   📂 Task categories found: {', '.join(categories)}")

        return unique_tasks, task_details, configmap

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch ConfigMap from GitHub: {e}")
        print(f"💡 Ensure the GitHub URL is accessible: {github_url}")
        return None, None, None
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML: {e}")
        return None, None, None


def save_extracted_tasks(tasks, task_details, configmap, output_dir: str = None):
    """Save extracted tasks and analysis"""
    if output_dir is None:
        # Default to task_lists relative to this script
        output_dir = Path(__file__).parent.parent / "task_lists"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save task list with details
    task_file = output_dir / "configmap_tasks.json"
    with open(task_file, "w") as f:
        json.dump(
            {
                "source": "trustyai-lmeval-tasks.yaml ConfigMap (PR #488)",
                "total_tasks": len(tasks),
                "tasks": tasks,
                "task_details": task_details,
            },
            f,
            indent=2,
        )

    # Save just task names for easy use
    names_file = output_dir / "task_names_only.txt"
    with open(names_file, "w") as f:
        f.write(f"# LMEval Task Names from ConfigMap - Total: {len(tasks)}\n")
        for task in tasks:
            f.write(f"{task}\n")

    # Save task list as Python module
    py_file = output_dir / "configmap_tasks.py"
    with open(py_file, "w") as f:
        f.write("# Tasks extracted from trustyai-lmeval-tasks.yaml ConfigMap\n")
        f.write("# Source: PR #488\n")
        f.write(f"# Total tasks: {len(tasks)}\n\n")
        f.write("CONFIGMAP_TASKS = [\n")
        for task in tasks:
            f.write(f'    "{task}",\n')
        f.write("]\n\n")
        f.write(f"TOTAL_TASKS = {len(tasks)}\n")

    # Save first 10 tasks for timing benchmark
    sample_file = output_dir / "timing_benchmark_tasks.py"
    sample_tasks = tasks[:10]
    with open(sample_file, "w") as f:
        f.write("# First 10 tasks for timing benchmark\n")
        f.write("TIMING_BENCHMARK_TASKS = [\n")
        for task in sample_tasks:
            f.write(f'    "{task}",\n')
        f.write("]\n")

    print("\n💾 Saved files:")
    print(f"   📄 Complete data: {task_file}")
    print(f"   📄 Task names only: {names_file}")
    print(f"   🐍 Python module: {py_file}")
    print(f"   ⏱️ Timing benchmark: {sample_file}")

    return {"complete": task_file, "names": names_file, "python": py_file, "benchmark": sample_file}


def compare_with_lmeval_list():
    """Suggest comparison with lm-eval --tasks list"""
    print("\n🔄 Next Steps:")
    print("   1. Compare with lm-eval harness:")
    print("      pip install lm-eval")
    print("      lm-eval --tasks list > lmeval_official_tasks.txt")
    print("   2. Choose tasks for timing benchmark")
    print("   3. Implement test modifications")


def main():
    print("🔍 Parsing trustyai-lmeval-tasks.yaml ConfigMap from GitHub")
    print("=" * 60)

    # Parse the ConfigMap
    tasks, task_details, configmap = parse_configmap()

    if not tasks:
        print("❌ Failed to extract tasks from ConfigMap")
        return 1

    # Show preview
    print(f"\n📋 Task Preview (first 20 of {len(tasks)}):")
    for i, task in enumerate(tasks[:20], 1):
        print(f"   {i:2d}. {task}")

    if len(tasks) > 20:
        print(f"   ... and {len(tasks) - 20} more")

    # Save results
    files = save_extracted_tasks(tasks, task_details, configmap)

    # Next steps
    compare_with_lmeval_list()

    return 0


if __name__ == "__main__":
    sys.exit(main())
