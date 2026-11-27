"""CSR/template parity validation for MLServer ServingRuntime pairs.

Validates that the CSR variants and OpenShift Template variants of the
MLServer ServingRuntime remain in sync: equal env lists and equal probe
configurations (startup, readiness, liveness — endpoint paths, timings,
and exec commands). A drift between a pair surfaces only at runtime on
one platform, so this test catches it at CI time instead.

Backs RHAI-60150. Follow-up from the PR #1026 review.
"""

from __future__ import annotations

import requests
import structlog
import yaml

LOGGER = structlog.get_logger(__name__)

ODHMC_RAW_BASE = "https://raw.githubusercontent.com/opendatahub-io/odh-model-controller/incubating/config/runtimes"

TEMPLATE_PAIRS = [
    ("csr-kserve-mlserver.yaml", "mlserver-template.yaml"),
    ("csr-kserve-mlserver-cuda.yaml", "mlserver-cuda-template.yaml"),
]

REQUEST_TIMEOUT = 30


def _fetch_template(filename: str) -> dict:
    """Fetch a ServingRuntime template YAML from the odh-model-controller repo."""
    url = f"{ODHMC_RAW_BASE}/{filename}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    assert response.status_code == 200, f"Failed to fetch {filename}: HTTP {response.status_code}"
    parsed = yaml.safe_load(response.text)
    assert isinstance(parsed, dict), f"{filename} did not parse as a YAML dict"
    return parsed


def _extract_container_spec(template: dict, *, is_openshift_template: bool) -> dict:
    """Extract the kserve-container spec from either template format."""
    if is_openshift_template:
        objects = template.get("objects", [])
        assert objects, "OpenShift Template has no objects"
        spec = objects[0].get("spec", {})
    else:
        spec = template.get("spec", {})
    containers = spec.get("containers", [])
    assert containers, "No containers found in spec"
    for container in containers:
        if container.get("name") == "kserve-container":
            return container
    return containers[0]


def _normalize_env(env_list: list) -> list[tuple[str, str]]:
    """Normalize an env var list to a sorted list of (name, value) tuples."""
    return sorted((item["name"], str(item.get("value", ""))) for item in env_list)


def _extract_probe(container: dict, probe_name: str) -> dict:
    """Extract a normalized probe config from a container spec."""
    probe = container.get(probe_name)
    if probe is None:
        return {}
    normalized: dict = {}
    if "httpGet" in probe:
        normalized["httpGet_path"] = probe["httpGet"].get("path")
        normalized["httpGet_port"] = probe["httpGet"].get("port")
    if "exec" in probe:
        normalized["exec_command"] = probe["exec"].get("command")
    for key in ("initialDelaySeconds", "periodSeconds", "failureThreshold", "timeoutSeconds"):
        if key in probe:
            normalized[key] = probe[key]
    return normalized


class TestTemplateParity:
    """CSR/template parity validation for MLServer ServingRuntime pairs."""

    def test_env_list_parity(self) -> None:
        """Given the CSR and template variants, when env lists are compared,
        then they must be identical within each pair."""
        for csr_file, template_file in TEMPLATE_PAIRS:
            csr_yaml = _fetch_template(csr_file)
            template_yaml = _fetch_template(template_file)
            csr_container = _extract_container_spec(csr_yaml, is_openshift_template=False)
            template_container = _extract_container_spec(template_yaml, is_openshift_template=True)
            csr_env = _normalize_env(csr_container.get("env", []))
            template_env = _normalize_env(template_container.get("env", []))
            assert csr_env == template_env, (
                f"Env list mismatch between {csr_file} and {template_file}:\n"
                f"  CSR only: {set(csr_env) - set(template_env)}\n"
                f"  Template only: {set(template_env) - set(csr_env)}"
            )
            LOGGER.info("parity-check.passed", pair=f"{csr_file} ↔ {template_file}", check="env_list")

    def test_startup_probe_parity(self) -> None:
        """Given the CSR and template variants, when startup probes are compared,
        then their configurations must be identical within each pair."""
        for csr_file, template_file in TEMPLATE_PAIRS:
            csr_yaml = _fetch_template(csr_file)
            template_yaml = _fetch_template(template_file)
            csr_container = _extract_container_spec(csr_yaml, is_openshift_template=False)
            template_container = _extract_container_spec(template_yaml, is_openshift_template=True)
            csr_probe = _extract_probe(csr_container, "startupProbe")
            template_probe = _extract_probe(template_container, "startupProbe")
            assert csr_probe == template_probe, (
                f"Startup probe mismatch between {csr_file} and {template_file}:\n"
                f"  CSR: {csr_probe}\n  Template: {template_probe}"
            )
            LOGGER.info("parity-check.passed", pair=f"{csr_file} ↔ {template_file}", check="startup_probe")

    def test_readiness_probe_parity(self) -> None:
        """Given the CSR and template variants, when readiness probes are compared,
        then their configurations must be identical within each pair."""
        for csr_file, template_file in TEMPLATE_PAIRS:
            csr_yaml = _fetch_template(csr_file)
            template_yaml = _fetch_template(template_file)
            csr_container = _extract_container_spec(csr_yaml, is_openshift_template=False)
            template_container = _extract_container_spec(template_yaml, is_openshift_template=True)
            csr_probe = _extract_probe(csr_container, "readinessProbe")
            template_probe = _extract_probe(template_container, "readinessProbe")
            assert csr_probe == template_probe, (
                f"Readiness probe mismatch between {csr_file} and {template_file}:\n"
                f"  CSR: {csr_probe}\n  Template: {template_probe}"
            )
            LOGGER.info("parity-check.passed", pair=f"{csr_file} ↔ {template_file}", check="readiness_probe")

    def test_liveness_probe_parity(self) -> None:
        """Given the CSR and template variants, when liveness probes are compared,
        then their configurations must be identical within each pair."""
        for csr_file, template_file in TEMPLATE_PAIRS:
            csr_yaml = _fetch_template(csr_file)
            template_yaml = _fetch_template(template_file)
            csr_container = _extract_container_spec(csr_yaml, is_openshift_template=False)
            template_container = _extract_container_spec(template_yaml, is_openshift_template=True)
            csr_probe = _extract_probe(csr_container, "livenessProbe")
            template_probe = _extract_probe(template_container, "livenessProbe")
            assert csr_probe == template_probe, (
                f"Liveness probe mismatch between {csr_file} and {template_file}:\n"
                f"  CSR: {csr_probe}\n  Template: {template_probe}"
            )
            LOGGER.info("parity-check.passed", pair=f"{csr_file} ↔ {template_file}", check="liveness_probe")
