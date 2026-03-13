from typing import Self

import pytest
from ocp_resources.pod import Pod

from utilities.general import validate_container_images


@pytest.mark.usefixtures("unprivileged_llama_stack_distribution")
@pytest.mark.parametrize(
    "unprivileged_model_namespace, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-llamastack-operator", "randomize_name": True},
            {"llama_stack_storage_size": "2Gi"},
        ),
    ],
    indirect=True,
)
@pytest.mark.downstream_only
@pytest.mark.llama_stack
class TestLlamaStackDistribution:
    """
    Test class that implements multiple tests to verify LlamaStack distribution functionality.

    This class contains tests that validate various aspects of the LlamaStack operator
    and its distribution components, including image validation and configuration checks.
    """

    @pytest.mark.smoke
    def test_llamastackdistribution_verify_images(
        self: Self,
        llama_stack_distribution_pods: Pod,
        related_images_refs: set[str],
    ) -> None:
        """
        Verify that LlamaStackDistribution container images meet the requirements:
        1. Images are hosted in registry.redhat.io
        2. Images use sha256 digest instead of tags
        3. Images are listed in the CSV's relatedImages section
        """
        validation_errors = []
        for pod in [llama_stack_distribution_pods]:
            validation_errors.extend(validate_container_images(pod=pod, valid_image_refs=related_images_refs))

        if validation_errors:
            pytest.fail("\n".join(validation_errors))
