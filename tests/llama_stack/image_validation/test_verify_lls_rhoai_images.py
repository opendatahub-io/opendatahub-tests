import pytest
from typing import Self, Set
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient

from utilities.general import (
    validate_container_images,
)
from ocp_resources.pod import Pod
from ocp_resources.namespace import Namespace


LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures(
    "enabled_llama_stack_operator"
)

@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-llamastack-image-validation"},
        )
    ],
    indirect=True,
)

@pytest.mark.rawdeployment
@pytest.mark.downstream_only
class TestLLSImages:
    """
    Tests to verify that all LLS (LlamaStack) component images meet the requirements:
    1. Images are hosted in registry.redhat.io
    2. Images use sha256 digest instead of tags
    3. Images are listed in the CSV's relatedImages section
    """

    @pytest.mark.smoke
    def test_verify_lls_operator_images(
        self: Self,
        admin_client: DynamicClient,
        lls_operator_pod_by_label: Pod,
        related_images_refs: Set[str],
    ):
        validation_errors = []
        for pod in [lls_operator_pod_by_label]:
            validation_errors.extend(
                validate_container_images(
                    pod=pod, valid_image_refs=related_images_refs, skip_patterns=["openshift-service-mesh"]
                )
            )

        if validation_errors:
            pytest.fail("\n".join(validation_errors))
