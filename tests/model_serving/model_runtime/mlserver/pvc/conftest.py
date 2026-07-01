"""
Fixtures for MLServer PVC (Persistent Volume Claim) storage tests.

This module provides fixtures for testing MLServer inference with models
stored on PVC instead of S3 storage.
"""

from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from pytest import FixtureRequest

from utilities.constants import KServeDeploymentType
from utilities.general import download_model_data
from utilities.inference_utils import create_isvc


@pytest.fixture(scope="class")
def mlserver_model_pvc(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """
    Create a PersistentVolumeClaim for storing MLServer model data.

    Args:
        request: Pytest request with parameters:
            - pvc-size: Size of the PVC (e.g., "5Gi")
            - access-modes: Access mode (default: "ReadWriteOnce")
            - storage-class-name: Optional storage class
        admin_client: Kubernetes admin client
        model_namespace: Namespace for the PVC

    Yields:
        PersistentVolumeClaim: Created PVC resource
    """
    pvc_kwargs: dict[str, Any] = {
        "name": "mlserver-model-pvc",
        "namespace": model_namespace.name,
        "client": admin_client,
        "size": request.param["pvc-size"],
        "accessmodes": request.param.get("access-modes", "ReadWriteOnce"),
    }
    if storage_class_name := request.param.get("storage-class-name"):
        pvc_kwargs["storage_class"] = storage_class_name

    with PersistentVolumeClaim(**pvc_kwargs) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def mlserver_pvc_downloaded_model_data(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_model_pvc: PersistentVolumeClaim,
    aws_secret_access_key: str,
    aws_access_key_id: str,
    models_s3_bucket_name: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    """
    Download MLServer model data from S3 bucket into the PVC.

    Uses the download_model_data utility to populate the PVC with model
    files from S3 storage before creating the InferenceService.

    Args:
        request: Pytest request with parameters:
            - model-dir: Path in S3 bucket (e.g., "sklearn")
        admin_client: Kubernetes admin client
        model_namespace: Namespace for the download job
        mlserver_model_pvc: PVC to download into
        aws_secret_access_key: AWS secret access key
        aws_access_key_id: AWS access key ID
        models_s3_bucket_name: S3 bucket name
        models_s3_bucket_endpoint: S3 endpoint URL
        models_s3_bucket_region: S3 region

    Returns:
        str: Model path within the PVC (used in storage_uri)
    """
    return download_model_data(
        client=admin_client,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        model_namespace=model_namespace.name,
        model_pvc_name=mlserver_model_pvc.name,
        bucket_name=models_s3_bucket_name,
        aws_endpoint_url=models_s3_bucket_endpoint,
        aws_default_region=models_s3_bucket_region,
        model_path=request.param["model-dir"],
        use_sub_path=True,
        restricted_scc_init=True,
    )


@pytest.fixture(scope="class")
def mlserver_pvc_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_serving_runtime,
    mlserver_model_pvc: PersistentVolumeClaim,
    mlserver_pvc_downloaded_model_data: str,
) -> Generator[InferenceService, Any, Any]:
    """
    Create an MLServer InferenceService using PVC storage.

    This fixture creates an InferenceService that loads the model from
    PVC storage instead of S3. The storage URI format is:
    pvc://{pvc_name}/{model_path}

    Args:
        request: Pytest request with parameters:
            - name: InferenceService name
            - deployment_mode: Deployment mode (default: STANDARD)
            - timeout: Optional timeout in seconds
            - min_replicas: Optional minimum replicas
        admin_client: Kubernetes admin client
        model_namespace: Namespace for the InferenceService
        mlserver_serving_runtime: MLServer runtime
        mlserver_model_pvc: PVC containing the model
        pvc_downloaded_model_data: Model path within PVC

    Yields:
        InferenceService: Created InferenceService resource
    """
    deployment_mode = request.param.get("deployment_mode", KServeDeploymentType.STANDARD)
    storage_uri = f"pvc://{mlserver_model_pvc.name}/{mlserver_pvc_downloaded_model_data}"

    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": mlserver_serving_runtime.name,
        "storage_uri": storage_uri,
        "model_format": mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "deployment_mode": deployment_mode,
        "wait_for_predictor_pods": True,
    }

    if timeout := request.param.get("timeout"):
        isvc_kwargs["timeout"] = timeout

    if (min_replicas := request.param.get("min_replicas")) is not None:
        isvc_kwargs["min_replicas"] = min_replicas

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc
