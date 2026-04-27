from typing import Self

import pytest
from ocp_resources.pod import Pod

from utilities.general import validate_container_images


@pytest.mark.usefixtures("unprivileged_ogx_distribution")
@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server_config",
    [
        pytest.param(
            {"name": "test-ogx-operator", "randomize_name": True},
            {"ogx_storage_size": "2Gi"},
        ),
    ],
    indirect=True,
)
@pytest.mark.downstream_only
@pytest.mark.ogx
class TestOgxDistribution:
    """
    Test class that implements multiple tests to verify OGX distribution functionality.

    This class contains tests that validate various aspects of the OGX operator
    and its distribution components, including image validation and configuration checks.
    """

    @pytest.mark.smoke
    def test_ogxdistribution_verify_images(
        self: Self,
        ogx_distribution_pods: Pod,
        related_images_refs: set[str],
    ) -> None:
        """
        Verify that OgxDistribution container images meet the requirements:
        1. Images are hosted in registry.redhat.io
        2. Images use sha256 digest instead of tags
        3. Images are listed in the CSV's relatedImages section
        """
        validation_errors = []
        for pod in [ogx_distribution_pods]:
            validation_errors.extend(validate_container_images(pod=pod, valid_image_refs=related_images_refs))

        if validation_errors:
            pytest.fail("\n".join(validation_errors))
