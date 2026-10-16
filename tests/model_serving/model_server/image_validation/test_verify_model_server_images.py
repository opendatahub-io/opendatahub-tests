"""
Tests to verify that Model Server and Metrics team (Serving Orchestration)
component images meet the requirements:
1. Images are hosted in registry.redhat.io
2. Images use sha256 digest instead of tags
3. Images are listed in the CSV's relatedImages section

Covered components:
- kserve control plane deployments (odh-model-controller, kserve-controller-manager);
  the kserve-controller-manager pod includes the kube-rbac-proxy (metrics) sidecar.
- InferenceService model server pods (predictor and storage-initializer images).
- InferenceGraph raw deployment router pods (router image).

Resources are created with placeholder storage; no actual model or inference is
required and pods do not need to be Ready. Pod images are validated against the
cluster CSV (relatedImages) at runtime; no hardcoded image SHAs.
"""

from typing import Self

import pytest
import structlog
from ocp_resources.pod import Pod

from tests.model_serving.model_server.image_validation.constant import KSERVE_CONTROL_PLANE_DEPLOYMENTS
from utilities.general import validate_container_images

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.downstream_only,
    pytest.mark.skip_must_gather,
    pytest.mark.tier1,
    pytest.mark.no_image_registry_mirroring,
]


@pytest.mark.parametrize(
    "kserve_control_plane_pods",
    [
        pytest.param(deployment, id=f"test_{deployment['name']}_pods_images")
        for deployment in KSERVE_CONTROL_PLANE_DEPLOYMENTS
    ],
    indirect=True,
)
class TestKServeControlPlaneImages:
    """
    Verify images of pods belonging to the kserve control plane deployments.
    """

    def test_verify_control_plane_pod_images(
        self: Self,
        kserve_control_plane_pods: list[Pod],
        related_images_refs: set[str],
    ) -> None:
        """
        Given pods of a kserve control plane deployment
        When validating their container images
        Then the images use registry.redhat.io, sha256 digest, and are listed in CSV relatedImages
        """
        validation_errors = []
        for pod in kserve_control_plane_pods:
            LOGGER.info(f"Validating {pod.name} in {pod.namespace}")
            validation_errors.extend(
                validate_container_images(
                    pod=pod,
                    valid_image_refs=related_images_refs,
                )
            )

        if validation_errors:
            pytest.fail("\n".join(validation_errors))
        LOGGER.info("control plane pods images : passed")


@pytest.mark.usefixtures("model_server_image_validation_namespace")
class TestModelServerImages:
    """
    Verify images of pods created by the kserve controllers (model server and router).
    """

    def test_verify_isvc_model_server_pod_images(
        self: Self,
        isvc_model_server_pods: list[Pod],
        related_images_refs: set[str],
    ) -> None:
        """
        Given an InferenceService created with a placeholder storage URI
        When validating its model server pod images (predictor and storage-initializer)
        Then the images use registry.redhat.io, sha256 digest, and are listed in CSV relatedImages
        """
        validation_errors = []
        for pod in isvc_model_server_pods:
            LOGGER.info(f"Validating {pod.name} in {pod.namespace}")
            validation_errors.extend(
                validate_container_images(
                    pod=pod,
                    valid_image_refs=related_images_refs,
                )
            )

        if validation_errors:
            pytest.fail("\n".join(validation_errors))
        LOGGER.info("isvc model server pod images : passed")

    @pytest.mark.rawdeployment
    def test_verify_inference_graph_router_pod_images(
        self: Self,
        inference_graph_router_pods: list[Pod],
        related_images_refs: set[str],
    ) -> None:
        """
        Given a raw InferenceGraph routing to a placeholder InferenceService
        When validating its router pod images
        Then the images use registry.redhat.io, sha256 digest, and are listed in CSV relatedImages
        """
        validation_errors = []
        for pod in inference_graph_router_pods:
            LOGGER.info(f"Validating {pod.name} in {pod.namespace}")
            validation_errors.extend(
                validate_container_images(
                    pod=pod,
                    valid_image_refs=related_images_refs,
                )
            )

        if validation_errors:
            pytest.fail("\n".join(validation_errors))
        LOGGER.info("inference graph router pod images : passed")
