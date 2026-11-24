"""
Tests to verify that Data Science Pipelines images meet the requirements:
1. Images are hosted in registry.redhat.io
2. Images use sha256 digest instead of tags
3. Images are listed in the CSV's relatedImages section

The test creates one pipeline server and checks the operator controller plus the
Deployments that server creates. Pipeline-run pods are not created.
"""

from typing import Self

import pytest
from ocp_resources.pod import Pod

from tests.data_science_pipelines.image_validation.utils import validate_images

pytestmark = [pytest.mark.downstream_only, pytest.mark.skip_must_gather, pytest.mark.tier1]


class TestDataSciencePipelinesImages:
    def test_verify_data_science_pipelines_images(
        self: Self,
        data_science_pipelines_pods: list[Pod],
        related_images_refs: set[str],
    ) -> None:
        """
        Given AI Pipelines is Managed and one DataSciencePipelinesApplication exists
        When the operator and pipeline-server pods are created
        Then every container and init container uses a registry.redhat.io sha256 image listed in the CSV
        """
        validate_images(pods_to_validate=data_science_pipelines_pods, related_images_refs=related_images_refs)
