"""vLLM-Omni ServingRuntime template validation tests."""

from typing import Any, Self

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.template import Template
from pytest_testconfig import config as py_config

from utilities.constants import RuntimeTemplates
from utilities.operator_utils import get_csv_related_images

LOGGER = structlog.get_logger(name=__name__)

# Metadata annotation and label key constants
_DASHBOARD_LABEL_KEY: str = "opendatahub.io/dashboard"
_SUPPORT_STATUS_KEY: str = "opendatahub.io/support-status"
_RECOMMENDED_ACCELERATORS_KEY: str = "opendatahub.io/recommended-accelerators"
_RUNTIME_VERSION_KEY: str = "opendatahub.io/runtime-version"
_KSERVE_RUNTIME_KEY: str = "opendatahub.io/kserve-runtime"
_PROMETHEUS_PATH_KEY: str = "prometheus.io/path"
_PROMETHEUS_PORT_KEY: str = "prometheus.io/port"
_MONITORING_SCRAPE_KEY: str = "monitoring.opendatahub.io/scrape"


@pytest.fixture(scope="module")
def vllm_omni_stable_template(admin_client: DynamicClient) -> Template:
    """Return the stable vLLM-Omni ServingRuntime Template object from the applications namespace."""
    applications_namespace: str = py_config["applications_namespace"]
    return Template(
        client=admin_client,
        name=RuntimeTemplates.VLLM_OMNI_CUDA,
        namespace=applications_namespace,
    )


@pytest.fixture(scope="module")
def vllm_omni_runtime_spec(vllm_omni_stable_template: Template) -> dict[str, Any]:
    """Extract the ServingRuntime spec dictionary from the stable vLLM-Omni template objects list."""
    if not vllm_omni_stable_template.exists:
        pytest.fail(
            f"Template '{RuntimeTemplates.VLLM_OMNI_CUDA}' not found in namespace "
            f"'{py_config['applications_namespace']}'"
        )
    objects = vllm_omni_stable_template.instance.objects
    if not objects:
        pytest.fail(f"Template '{RuntimeTemplates.VLLM_OMNI_CUDA}' has no objects defined")
    runtime_dict: dict[str, Any] = objects[0].to_dict()
    return runtime_dict


class TestVllmOmniTemplateStructure:
    """Validate the vLLM-Omni ServingRuntime template spec, metadata, and container configuration."""

    @pytest.mark.smoke
    def test_vllm_omni_runtime_template_exists(
        self: Self,
        admin_client: DynamicClient,
        vllm_omni_stable_template: Template,
        vllm_omni_runtime_spec: dict[str, Any],
        vllm_omni_runtime_image: str | None,
    ) -> None:
        """Confirm the vLLM-Omni template is deployed with a CSV-tracked sha256 digest image.

        Given the RHOAI operator is installed with vllm-omni-cuda-runtime-template
        When the Template resource is retrieved from the applications namespace
        Then the template exists in the namespace
        And spec.containers[0].image uses a @sha256: digest reference
        And the image path contains 'vllm-omni'
        And the image is present in the operator CSV relatedImages list.
        """
        assert vllm_omni_stable_template.exists, (
            f"Template '{RuntimeTemplates.VLLM_OMNI_CUDA}' not found "
            f"in namespace '{py_config['applications_namespace']}'"
        )

        containers: list[dict[str, Any]] = vllm_omni_runtime_spec.get("spec", {}).get("containers", [])
        assert containers, (
            f"No containers found in vLLM-Omni template '{RuntimeTemplates.VLLM_OMNI_CUDA}' spec. "
            "Verify the template object has a valid spec.containers array."
        )

        container_image: str = containers[0].get("image", "")
        assert container_image, "spec.containers[0].image is not set in the vLLM-Omni template"
        assert "@sha256:" in container_image, (
            f"Container image '{container_image}' does not use @sha256: digest format. "
            f"Image must be pinned to a digest via RELATED_IMAGE from the operator CSV."
        )
        assert "vllm-omni" in container_image, (
            f"Container image '{container_image}' does not contain 'vllm-omni' in the path"
        )

        if vllm_omni_runtime_image:
            LOGGER.info(
                event="Skipping CSV relatedImages check — custom runtime image override is active",
                custom_image=vllm_omni_runtime_image,
                template_image=container_image,
            )
            return

        csv_images = {
            entry["image"] for entry in get_csv_related_images(admin_client=admin_client) if entry.get("image")
        }
        omni_csv_images = sorted(img for img in csv_images if ("vllm-omni" in img or "vllm_omni" in img))
        LOGGER.info(
            event="CSV relatedImages check",
            template_image=container_image,
            csv_total=len(csv_images),
            omni_csv_images=omni_csv_images or "none found",
        )
        assert omni_csv_images, (
            f"No vllm-omni image found in CSV relatedImages ({len(csv_images)} total). "
            "The operator CSV must include a vllm-omni-cuda image entry."
        )
        assert container_image in omni_csv_images, (
            f"Template image does not match CSV vllm-omni image.\n"
            f"  Template: {container_image}\n"
            f"  CSV:      {omni_csv_images}"
        )

    @pytest.mark.smoke
    def test_vllm_omni_dashboard_discovery_annotations(
        self: Self,
        vllm_omni_runtime_spec: dict[str, Any],
    ) -> None:
        """Confirm vLLM-Omni template has required labels, annotations, and supportedModelFormats.

        Given the vLLM-Omni ServingRuntime template is deployed
        When metadata labels, metadata annotations, spec annotations, and spec.supportedModelFormats are inspected
        Then opendatahub.io/dashboard: 'true' is in metadata.labels (not metadata.annotations)
        And metadata.annotations contains support-status='unsupported', recommended-accelerators, and runtime-version
        And spec.annotations contains kserve-runtime='vllm-omni', prometheus path/port, and monitoring.scrape='true'
        And spec.supportedModelFormats[0].name equals 'vLLM'.
        """
        metadata: dict[str, Any] = vllm_omni_runtime_spec.get("metadata", {})
        metadata_labels: dict[str, str] = metadata.get("labels", {})
        metadata_annotations: dict[str, str] = metadata.get("annotations", {})
        spec: dict[str, Any] = vllm_omni_runtime_spec.get("spec", {})
        spec_annotations: dict[str, str] = spec.get("annotations", {})

        assert metadata_labels.get(_DASHBOARD_LABEL_KEY) == "true", (
            f"'{_DASHBOARD_LABEL_KEY}: true' not found in metadata.labels. Labels: {metadata_labels}"
        )
        assert metadata_annotations.get(_SUPPORT_STATUS_KEY) == "unsupported", (
            f"'{_SUPPORT_STATUS_KEY}' must be 'unsupported' in metadata.annotations. "
            f"Found: {metadata_annotations.get(_SUPPORT_STATUS_KEY)!r}"
        )
        assert _RECOMMENDED_ACCELERATORS_KEY in metadata_annotations, (
            f"'{_RECOMMENDED_ACCELERATORS_KEY}' not found in metadata.annotations. "
            f"Available keys: {sorted(metadata_annotations.keys())}"
        )
        assert _RUNTIME_VERSION_KEY in metadata_annotations, (
            f"'{_RUNTIME_VERSION_KEY}' not found in metadata.annotations. "
            f"Available keys: {sorted(metadata_annotations.keys())}"
        )

        assert spec_annotations.get(_KSERVE_RUNTIME_KEY) == "vllm-omni", (
            f"'{_KSERVE_RUNTIME_KEY}' must be 'vllm-omni' in spec.annotations. "
            f"Found: {spec_annotations.get(_KSERVE_RUNTIME_KEY)!r}"
        )
        assert spec_annotations.get(_PROMETHEUS_PATH_KEY) == "/metrics", (
            f"'{_PROMETHEUS_PATH_KEY}' must be '/metrics' in spec.annotations. "
            f"Found: {spec_annotations.get(_PROMETHEUS_PATH_KEY)!r}"
        )
        assert spec_annotations.get(_PROMETHEUS_PORT_KEY) == "8080", (
            f"'{_PROMETHEUS_PORT_KEY}' must be '8080' in spec.annotations. "
            f"Found: {spec_annotations.get(_PROMETHEUS_PORT_KEY)!r}"
        )
        assert spec_annotations.get(_MONITORING_SCRAPE_KEY) == "true", (
            f"'{_MONITORING_SCRAPE_KEY}' must be 'true' in spec.annotations. "
            f"Found: {spec_annotations.get(_MONITORING_SCRAPE_KEY)!r}"
        )

        supported_formats: list[dict[str, Any]] = spec.get("supportedModelFormats", [])
        assert supported_formats, (
            f"spec.supportedModelFormats is empty in vLLM-Omni template '{RuntimeTemplates.VLLM_OMNI_CUDA}'. "
            "Verify the template defines at least one supportedModelFormat entry with name='vLLM'."
        )
        vllm_format = next(
            (fmt for fmt in supported_formats if fmt.get("name") == "vLLM"),
            None,
        )
        assert vllm_format is not None, (
            f"No entry with name='vLLM' found in supportedModelFormats. "
            f"Found: {[fmt.get('name') for fmt in supported_formats]}"
        )

    @pytest.mark.smoke
    def test_vllm_omni_container_command_and_args(
        self: Self,
        vllm_omni_runtime_spec: dict[str, Any],
    ) -> None:
        """Confirm the vLLM-Omni container command, args, and HF_HOME env var are correct.

        Given the vLLM-Omni ServingRuntime template is deployed
        When spec.containers[0] command, args, and env are inspected
        Then command is ['vllm', 'serve']
        And args include --port=8080, --model=/mnt/models, and --omni
        And HF_HOME env var is set to /tmp/hf_home.
        """
        containers: list[dict[str, Any]] = vllm_omni_runtime_spec.get("spec", {}).get("containers", [])
        assert containers, (
            f"No containers found in vLLM-Omni template '{RuntimeTemplates.VLLM_OMNI_CUDA}' spec. "
            "Verify the template object has a valid spec.containers array."
        )
        container: dict[str, Any] = containers[0]

        command: list[str] = container.get("command", [])
        assert command == ["vllm", "serve"], f"Container command must be ['vllm', 'serve']. Found: {command}"

        args: list[str] = container.get("args", [])
        for expected_arg in ("--port=8080", "--model=/mnt/models", "--omni"):
            assert expected_arg in args, f"Expected arg '{expected_arg}' not found in container args: {args}"

        env_vars: list[dict[str, str]] = container.get("env", [])
        hf_home_entry = next((env for env in env_vars if env.get("name") == "HF_HOME"), None)
        assert hf_home_entry is not None, (
            f"Environment variable 'HF_HOME' not found in container env. Env vars: {env_vars}"
        )
        assert hf_home_entry.get("value") == "/tmp/hf_home", (
            f"HF_HOME value must be '/tmp/hf_home'. Found: {hf_home_entry.get('value')!r}"
        )

    @pytest.mark.smoke
    def test_vllm_omni_probe_configuration(
        self: Self,
        vllm_omni_runtime_spec: dict[str, Any],
    ) -> None:
        """Confirm all three container probes are configured with correct values.

        Given the vLLM-Omni ServingRuntime template is deployed
        When the container's startupProbe, readinessProbe, and livenessProbe are inspected
        Then startupProbe has failureThreshold=40 and periodSeconds=30
        And readinessProbe has periodSeconds=10 and failureThreshold=3
        And livenessProbe has periodSeconds=15 and failureThreshold=3
        And none of the three probes has initialDelaySeconds set.
        """
        containers: list[dict[str, Any]] = vllm_omni_runtime_spec.get("spec", {}).get("containers", [])
        assert containers, (
            f"No containers found in vLLM-Omni template '{RuntimeTemplates.VLLM_OMNI_CUDA}' spec. "
            "Verify the template object has a valid spec.containers array."
        )
        container: dict[str, Any] = containers[0]

        startup_probe: dict[str, Any] = container.get("startupProbe", {})
        assert startup_probe, "startupProbe is not defined in the vLLM-Omni container spec"
        assert startup_probe.get("failureThreshold") == 40, (
            f"startupProbe.failureThreshold must be 40. Found: {startup_probe.get('failureThreshold')}"
        )
        assert startup_probe.get("periodSeconds") == 30, (
            f"startupProbe.periodSeconds must be 30. Found: {startup_probe.get('periodSeconds')}"
        )

        readiness_probe: dict[str, Any] = container.get("readinessProbe", {})
        assert readiness_probe, "readinessProbe is not defined in the vLLM-Omni container spec"
        assert readiness_probe.get("periodSeconds") == 10, (
            f"readinessProbe.periodSeconds must be 10. Found: {readiness_probe.get('periodSeconds')}"
        )
        assert readiness_probe.get("failureThreshold") == 3, (
            f"readinessProbe.failureThreshold must be 3. Found: {readiness_probe.get('failureThreshold')}"
        )

        liveness_probe: dict[str, Any] = container.get("livenessProbe", {})
        assert liveness_probe, "livenessProbe is not defined in the vLLM-Omni container spec"
        assert liveness_probe.get("periodSeconds") == 15, (
            f"livenessProbe.periodSeconds must be 15. Found: {liveness_probe.get('periodSeconds')}"
        )
        assert liveness_probe.get("failureThreshold") == 3, (
            f"livenessProbe.failureThreshold must be 3. Found: {liveness_probe.get('failureThreshold')}"
        )

        for probe_name, probe in (
            ("startupProbe", startup_probe),
            ("readinessProbe", readiness_probe),
            ("livenessProbe", liveness_probe),
        ):
            assert "initialDelaySeconds" not in probe, (
                f"{probe_name} must not define initialDelaySeconds. Found: {probe.get('initialDelaySeconds')}"
            )

    @pytest.mark.smoke
    def test_vllm_omni_supported_model_formats(
        self: Self,
        vllm_omni_runtime_spec: dict[str, Any],
    ) -> None:
        """Confirm spec.supportedModelFormats has exactly one entry with name 'vLLM'.

        Given the vLLM-Omni ServingRuntime template is deployed
        When spec.supportedModelFormats is inspected
        Then the list contains exactly one entry
        And the entry's name is 'vLLM' (exact case-sensitive match required for KServe dashboard lookup).
        """
        spec: dict[str, Any] = vllm_omni_runtime_spec.get("spec", {})
        supported_formats: list[dict[str, Any]] = spec.get("supportedModelFormats", [])

        assert len(supported_formats) == 1, (
            f"spec.supportedModelFormats must have exactly 1 entry. Found {len(supported_formats)}: {supported_formats}"
        )
        assert supported_formats[0].get("name") == "vLLM", (
            f"supportedModelFormats[0].name must be 'vLLM' (case-sensitive). "
            f"Found: {supported_formats[0].get('name')!r}"
        )

    @pytest.mark.smoke
    def test_vllm_omni_runtime_version_annotation(
        self: Self,
        vllm_omni_runtime_spec: dict[str, Any],
    ) -> None:
        """Confirm the opendatahub.io/runtime-version annotation is present and non-empty.

        Given the vLLM-Omni ServingRuntime template is deployed
        When metadata.annotations is inspected
        Then the opendatahub.io/runtime-version key is present
        And the annotation value is non-empty (not None or empty string).
        """
        metadata_annotations: dict[str, str] = vllm_omni_runtime_spec.get("metadata", {}).get("annotations", {})

        assert _RUNTIME_VERSION_KEY in metadata_annotations, (
            f"'{_RUNTIME_VERSION_KEY}' not found in metadata.annotations. "
            f"Available keys: {sorted(metadata_annotations.keys())}"
        )
        runtime_version: str = metadata_annotations[_RUNTIME_VERSION_KEY]
        assert runtime_version, f"'{_RUNTIME_VERSION_KEY}' annotation is present but has an empty value"
        LOGGER.info(event="vLLM-Omni runtime-version annotation found", runtime_version=runtime_version)

    @pytest.mark.smoke
    def test_vllm_omni_security_context(
        self: Self,
        vllm_omni_runtime_spec: dict[str, Any],
    ) -> None:
        """Confirm the container securityContext enforces privilege and non-root hardening.

        Given the vLLM-Omni ServingRuntime template is deployed
        When spec.containers[0].securityContext is inspected
        Then allowPrivilegeEscalation is False
        And privileged is False
        And runAsNonRoot is True
        And capabilities.drop contains 'ALL'.
        """
        containers: list[dict[str, Any]] = vllm_omni_runtime_spec.get("spec", {}).get("containers", [])
        assert containers, (
            f"No containers found in vLLM-Omni template '{RuntimeTemplates.VLLM_OMNI_CUDA}' spec. "
            "Verify the template object has a valid spec.containers array."
        )

        security_context: dict[str, Any] = containers[0].get("securityContext", {})
        assert security_context, "securityContext is not defined in the vLLM-Omni container spec"

        assert security_context.get("allowPrivilegeEscalation") is False, (
            f"allowPrivilegeEscalation must be False. Found: {security_context.get('allowPrivilegeEscalation')!r}"
        )
        assert security_context.get("privileged") is False, (
            f"privileged must be False. Found: {security_context.get('privileged')!r}"
        )
        assert security_context.get("runAsNonRoot") is True, (
            f"runAsNonRoot must be True. Found: {security_context.get('runAsNonRoot')!r}"
        )

        capabilities: dict[str, Any] = security_context.get("capabilities", {})
        drop_list: list[str] = capabilities.get("drop", [])
        assert "ALL" in drop_list, f"capabilities.drop must contain 'ALL'. Found: {drop_list}"


@pytest.mark.smoke
def test_vllm_omni_csv_related_images(admin_client: DynamicClient) -> None:
    """Confirm the RHOAI operator CSV relatedImages list includes a vLLM-Omni image.

    Given the RHOAI operator is installed on the cluster
    When the operator CSV spec.relatedImages section is inspected
    Then an entry with 'vllm-omni' in the image path is present
    And the image reference uses @sha256: digest format (not a mutable tag).
    """
    related_images: list[dict[str, str]] = get_csv_related_images(admin_client=admin_client)
    assert related_images, (
        "CSV spec.relatedImages is empty — expected at least one image entry. "
        "Verify the RHOAI operator CSV is installed and its spec.relatedImages section is populated."
    )

    vllm_omni_entries: list[dict[str, str]] = [img for img in related_images if "vllm-omni" in img.get("image", "")]
    assert vllm_omni_entries, (
        "No entry with 'vllm-omni' found in CSV spec.relatedImages. "
        f"Inspected {len(related_images)} entries. "
        f"Image paths: {[img.get('image', '') for img in related_images]}"
    )

    for entry in vllm_omni_entries:
        image: str = entry.get("image", "")
        assert "@sha256:" in image, f"vLLM-Omni CSV relatedImages entry '{image}' does not use @sha256: digest format"
    LOGGER.info(
        event="vLLM-Omni image entries found in CSV relatedImages",
        count=len(vllm_omni_entries),
        entries=[entry.get("image", "") for entry in vllm_omni_entries],
    )


@pytest.mark.smoke
def test_vllm_omni_template_variants_present(admin_client: DynamicClient) -> None:
    """Confirm the stable vLLM-Omni template is deployed in the applications namespace.

    Given the odh-model-controller kustomization.yaml includes vllm-omni-cuda-template
    When the Template resource is queried from the applications namespace
    Then the stable template must exist.

    Note: fast-1 and fast-2 variants are tested in fast/test_vllm_omni_fast_inference.py
    and deselected by pytest_collection_modifyitems until fast images are available.
    """
    applications_namespace: str = py_config["applications_namespace"]
    template = Template(
        client=admin_client,
        name=RuntimeTemplates.VLLM_OMNI_CUDA,
        namespace=applications_namespace,
    )
    assert template.exists, (
        f"vLLM-Omni template '{RuntimeTemplates.VLLM_OMNI_CUDA}' not found in namespace '{applications_namespace}'. "
        f"Verify it is listed in odh-model-controller kustomization.yaml resources."
    )
    LOGGER.info(
        event="vLLM-Omni stable template found",
        template_name=RuntimeTemplates.VLLM_OMNI_CUDA,
        namespace=applications_namespace,
    )
