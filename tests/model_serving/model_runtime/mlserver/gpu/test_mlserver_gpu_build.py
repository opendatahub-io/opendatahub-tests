"""Tests to verify GPU-enabled MLServer runtime build artifacts.

Validates that the RHOAI operator CSV includes both CPU and GPU MLServer images
in digest format and that the mlserver-cuda-runtime template has correct GPU
resource limits and container configuration.
"""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.template import Template
from pytest_testconfig import config as py_config

from utilities.constants import RHOAI_OPERATOR_NAMESPACE, Containers, RuntimeTemplates
from utilities.operator_utils import get_cluster_service_version
from utilities.serving_runtime import get_runtime_image_from_template

pytestmark = [
    pytest.mark.usefixtures("skip_if_no_gpu_for_mlserver"),
    pytest.mark.gpu,
    pytest.mark.mlserver_nvidia_gpu,
    pytest.mark.downstream_only,
]

_ODH_MLSERVER_IMAGE_NAME: str = "odh_mlserver_image"
_ODH_MLSERVER_CUDA_IMAGE_NAME: str = "odh_mlserver_cuda_image"


class TestMLServerGPUBuildArtifacts:
    """Validate GPU runtime build artifacts in RHOAI operator CSV and templates."""

    def test_csv_includes_mlserver_cuda_image(self, admin_client: DynamicClient) -> None:
        """Verify RHOAI operator CSV relatedImages includes MLServer image entries in digest format.

        Given the RHOAI operator is installed
        When the operator CSV relatedImages section is inspected
        Then odh_mlserver_image and odh_mlserver_cuda_image entries are present
        And both images use @sha256: digest format
        And the MLServer and MLServer-CUDA template images match the respective CSV entries.
        """
        csv = get_cluster_service_version(
            client=admin_client,
            prefix="rhods-operator",
            namespace=RHOAI_OPERATOR_NAMESPACE,
        )

        related_images = csv.instance.spec.relatedImages
        assert related_images, "CSV spec.relatedImages is empty"
        images_by_name: dict[str, str] = {img.name: img.image for img in related_images if hasattr(img, "name")}

        applications_namespace: str = py_config["applications_namespace"]
        image_template_pairs = [
            (_ODH_MLSERVER_IMAGE_NAME, RuntimeTemplates.MLSERVER),
            (_ODH_MLSERVER_CUDA_IMAGE_NAME, RuntimeTemplates.MLSERVER_CUDA),
        ]

        for image_name, template_name in image_template_pairs:
            assert image_name in images_by_name, (
                f"Entry '{image_name}' not found in CSV relatedImages. Available: {sorted(images_by_name.keys())}"
            )
            csv_image = images_by_name[image_name]
            assert "@sha256:" in csv_image, f"Image '{csv_image}' does not use @sha256: digest format"

            template_image = get_runtime_image_from_template(
                client=admin_client,
                template_name=template_name,
                namespace=applications_namespace,
            )
            assert template_image == csv_image, (
                f"Template '{template_name}' image '{template_image}' does not match "
                f"CSV relatedImages '{image_name}' entry: '{csv_image}'"
            )

    def test_cuda_template_has_correct_config(self, admin_client: DynamicClient) -> None:
        """Verify mlserver-cuda-runtime template exists with correct container configuration.

        Given the mlserver-cuda-runtime-template exists
        When the template's container list is inspected
        Then a container named kserve-container must be present
        And the container image must use @sha256: digest format.

        Note: GPU resource limits (nvidia.com/gpu) are not set in the template.
        They are injected by the RHOAI Dashboard when an accelerator profile is selected.
        """
        applications_namespace: str = py_config["applications_namespace"]

        template = Template(
            client=admin_client,
            name=RuntimeTemplates.MLSERVER_CUDA,
            namespace=applications_namespace,
        )
        assert template.exists, (
            f"Template '{RuntimeTemplates.MLSERVER_CUDA}' not found in namespace '{applications_namespace}'"
        )

        objects = template.instance.objects
        assert objects, f"Template '{RuntimeTemplates.MLSERVER_CUDA}' has no objects"

        template_dict: dict[str, Any] = objects[0].to_dict()
        containers: list[dict[str, Any]] = template_dict.get("spec", {}).get("containers", [])
        assert containers, f"Template '{RuntimeTemplates.MLSERVER_CUDA}' has no containers defined"

        kserve_container: dict[str, Any] | None = next(
            (container for container in containers if container.get("name") == Containers.KSERVE_CONTAINER_NAME),
            None,
        )
        assert kserve_container is not None, (
            f"Container '{Containers.KSERVE_CONTAINER_NAME}' not found in template. "
            f"Found containers: {[c.get('name') for c in containers]}"
        )

        container_image: str | None = kserve_container.get("image")
        assert container_image is not None, f"Container '{Containers.KSERVE_CONTAINER_NAME}' has no image in template"
        assert "@sha256:" in container_image, f"Container image '{container_image}' does not use @sha256: digest format"
