"""Boundary-case tests for the startup-probe CHECK_REPOSITORY_POPULATED shell switch.

Validates the unguarded boundaries of the startup probe's ``case`` statement
on ``CHECK_REPOSITORY_POPULATED`` and the ``ls -A "$MLSERVER_MODELS_DIR"``
check. A regression in any of these silently changes pod startup semantics.

Backs RHAI-60151. Follow-up from the PR #1026 review.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import requests
import structlog
import yaml

LOGGER = structlog.get_logger(__name__)

ODHMC_RAW_BASE = "https://raw.githubusercontent.com/opendatahub-io/odh-model-controller/incubating/config/runtimes"
TEMPLATE_FILE = "mlserver-template.yaml"
REQUEST_TIMEOUT = 30


def _extract_startup_probe_command() -> str:
    """Extract the startup probe shell command from the MLServer template."""
    url = f"{ODHMC_RAW_BASE}/{TEMPLATE_FILE}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    assert response.status_code == 200, f"Failed to fetch template: HTTP {response.status_code}"
    parsed = yaml.safe_load(response.text)
    objects = parsed.get("objects", [])
    assert objects, "Template has no objects"
    spec = objects[0].get("spec", {})
    containers = spec.get("containers", [])
    assert containers, "No containers in spec"
    startup_probe = containers[0].get("startupProbe", {})
    exec_config = startup_probe.get("exec", {})
    command = exec_config.get("command", [])
    assert len(command) >= 3, f"Unexpected command array: {command}"
    return command[2]


def _run_probe(
    command: str,
    *,
    check_populated: str | None,
    models_dir: str | None,
) -> int:
    """Run the startup probe command with controlled environment variables."""
    env = {k: v for k, v in os.environ.items() if k not in ("CHECK_REPOSITORY_POPULATED", "MLSERVER_MODELS_DIR")}
    if check_populated is not None:
        env["CHECK_REPOSITORY_POPULATED"] = check_populated
    if models_dir is not None:
        env["MLSERVER_MODELS_DIR"] = models_dir
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        env=env,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return result.returncode


@pytest.fixture(scope="module")
def probe_command() -> str:
    """Extract the startup probe command once for all tests in this module."""
    return _extract_startup_probe_command()


@pytest.fixture()
def temp_dir() -> Generator[Path]:
    """Provide a temporary directory that is cleaned up after each test."""
    d = Path(tempfile.mkdtemp(prefix="probe-test-"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestStartupProbeBoundaries:
    """Boundary-case validation for the startup-probe shell switch."""

    def test_unset_check_populated_skips_check(self, probe_command: str) -> None:
        """Given CHECK_REPOSITORY_POPULATED is unset, when the probe runs,
        then it must exit 0 (check skipped, dynamic-profile default)."""
        exit_code = _run_probe(probe_command, check_populated=None, models_dir="/tmp")
        assert exit_code == 0, f"Unset should skip: exit {exit_code}"

    def test_false_skips_check(self, probe_command: str) -> None:
        """Given CHECK_REPOSITORY_POPULATED=false, when the probe runs,
        then it must exit 0 (check skipped explicitly)."""
        exit_code = _run_probe(probe_command, check_populated="false", models_dir="/tmp")
        assert exit_code == 0, f"false should skip: exit {exit_code}"

    def test_true_with_populated_dir_passes(self, probe_command: str, temp_dir: Path) -> None:
        """Given CHECK_REPOSITORY_POPULATED=true and a non-empty models dir,
        when the probe runs, then it must exit 0."""
        (temp_dir / "model-settings.json").write_text("{}")
        exit_code = _run_probe(probe_command, check_populated="true", models_dir=str(temp_dir))
        assert exit_code == 0, f"true + populated should pass: exit {exit_code}"

    def test_true_with_empty_dir_fails(self, probe_command: str, temp_dir: Path) -> None:
        """Given CHECK_REPOSITORY_POPULATED=true and an empty models dir,
        when the probe runs, then it must exit non-zero."""
        exit_code = _run_probe(probe_command, check_populated="true", models_dir=str(temp_dir))
        assert exit_code != 0, f"true + empty should fail: exit {exit_code}"

    def test_true_with_missing_dir_fails(self, probe_command: str, temp_dir: Path) -> None:
        """Given CHECK_REPOSITORY_POPULATED=true and a nonexistent models dir,
        when the probe runs, then it must exit non-zero."""
        missing_dir = temp_dir / "does-not-exist"
        exit_code = _run_probe(probe_command, check_populated="true", models_dir=str(missing_dir))
        assert exit_code != 0, f"true + missing should fail: exit {exit_code}"

    def test_unset_models_dir_does_not_succeed_against_cwd(self, probe_command: str, temp_dir: Path) -> None:
        """Given CHECK_REPOSITORY_POPULATED=true and MLSERVER_MODELS_DIR unset,
        when the probe runs from a non-empty CWD, then `ls -A ""` must not
        succeed against the working directory."""
        (temp_dir / "placeholder").write_text("x")
        env = {k: v for k, v in os.environ.items() if k not in ("CHECK_REPOSITORY_POPULATED", "MLSERVER_MODELS_DIR")}
        env["CHECK_REPOSITORY_POPULATED"] = "true"
        result = subprocess.run(
            ["/bin/sh", "-c", probe_command],
            env=env,
            capture_output=True,
            timeout=10,
            check=False,
            cwd=str(temp_dir),
        )
        assert result.returncode != 0 or not (temp_dir / "placeholder").exists(), (
            f"Unset dir + non-empty CWD should not pass: exit {result.returncode}"
        )

    def test_hidden_only_contents_count_as_populated(self, probe_command: str, temp_dir: Path) -> None:
        """Given CHECK_REPOSITORY_POPULATED=true and a models dir containing
        only hidden files, when the probe runs, then `ls -A` includes
        hidden files so the dir counts as populated (exit 0). Pins current behavior."""
        (temp_dir / ".keep").write_text("")
        exit_code = _run_probe(probe_command, check_populated="true", models_dir=str(temp_dir))
        assert exit_code == 0, f"Hidden-only should count as populated: exit {exit_code}"

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1"])
    def test_case_insensitive_true_variants_enforce_check(self, probe_command: str, temp_dir: Path, value: str) -> None:
        """Given CHECK_REPOSITORY_POPULATED in any case variant of true or 1,
        when the probe runs against an empty dir, then the check is enforced."""
        exit_code = _run_probe(probe_command, check_populated=value, models_dir=str(temp_dir))
        assert exit_code != 0, f"{value!r} should enforce on empty dir: exit {exit_code}"

    @pytest.mark.parametrize("value", ["yes", "on", "enabled", "random-text"])
    def test_non_boolean_values_skip_check(self, probe_command: str, value: str) -> None:
        """Given CHECK_REPOSITORY_POPULATED set to a non-boolean value,
        when the probe runs, then the check is skipped (exit 0)."""
        exit_code = _run_probe(probe_command, check_populated=value, models_dir="/definitely/not/here")
        assert exit_code == 0, f"{value!r} should skip: exit {exit_code}"
