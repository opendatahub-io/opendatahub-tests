# We use wrapper library to interact with openshift cluster kinds.
# This script looks for calls bypassing wrapper library: https://github.com/RedHatQE/openshift-python-wrapper/
# created with help from claude
import os
import re
import sys
from pathlib import Path

PROHIBITED_PATTERNS = [
    r"\.get\((.*)api_version=(.*),\)",
    r"\.resources\.get\((.*)kind=(.*),\)",
    r"client\.resources\.get(.*)kind=(.*)",
]
KIND_PATTERN = r'kind="(.*)"'


def find_all_python_files(root_dir: Path) -> list[str]:
    skip_folders = {".tox", "venv", ".pytest_cache", "site-packages", ".git", ".local"}

    py_files = [
        file_name
        for file_name in Path(os.path.abspath(root_dir)).rglob("*.py")
        if not any(any(folder_name in part for folder_name in skip_folders) for part in file_name.parts)
    ]
    return [str(file_name) for file_name in py_files]


def check_file_for_violations(filepath: str) -> dict[str, set[str]]:
    with open(filepath, "r") as f:
        content = f.read()
    violations = set()
    kinds = set()
    for line_num, line in enumerate(content.split("\n"), 1):
        line = line.strip()
        for pattern in PROHIBITED_PATTERNS:
            if re.search(pattern, line):
                kind_match = re.search(KIND_PATTERN, line)
                if kind_match:
                    kinds.add(kind_match.group(1))
                violation_str = f"{filepath}:{line_num} - {line}"
                violations.add(violation_str)

    return {"violations": violations, "kind": kinds}


if __name__ == "__main__":
    all_violations = set()
    all_kinds = set()
    all_files = find_all_python_files(root_dir=Path(__file__).parent.parent)
    for filepath in all_files:
        result = check_file_for_violations(filepath=filepath)
        if result["violations"]:
            all_violations.update(result["violations"])
        if result["kind"]:
            all_kinds.update(result["kind"])
    if all_violations:
        print("Prohibited patterns found:")
        for violation in all_violations:
            print(f"  {violation}")
        if all_kinds:
            print(
                "\n\nPlease check if the following kinds exists in "
                "https://github.com/RedHatQE/openshift-python-wrapper/tree/main/ocp_resources:"
            )
            print(
                "For details about why we need such resources in openshift-python-wrapper, please check: "
                "https://github.com/opendatahub-io/opendatahub-tests/blob/main/docs/DEVELOPER_GUIDE.md#"
                "interacting-with-kubernetesopenshift-apis"
            )
            for kind in all_kinds:
                print(f"  {kind}")
    if all_kinds or all_violations:
        sys.exit(1)
    sys.exit(0)
