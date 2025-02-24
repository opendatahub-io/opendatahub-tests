from __future__ import annotations

import base64
import os
from functools import cache

from kubernetes.dynamic import DynamicClient
from ocp_resources.secret import Secret
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from utilities.constants import (
    ISTIO_CA_BUNDLE_FILENAME,
    KServeDeploymentType,
    OPENSHIFT_CA_BUNDLE_FILENAME,
)
from utilities.infra import is_managed_cluster, is_self_managed_operator


LOGGER = get_logger(name=__name__)


def create_ca_bundle_file(client: DynamicClient, ca_type: str) -> str:
    """
    Creates a ca bundle file from a secret

    Args:
        client (DynamicClient): DynamicClient object
        ca_type (str): The type of ca bundle to create. Can be "knative" or "openshift"

    Returns:
        str: The path to the ca bundle file. If cert is not created, return empty string

    Raises:
        ValueError: If ca_type is not "knative" or "openshift"

    """
    if ca_type == "knative":
        certs_secret = Secret(
            client=client,
            name="knative-serving-cert",
            namespace="istio-system",
        )
        filename = ISTIO_CA_BUNDLE_FILENAME

    elif ca_type == "openshift":
        certs_secret = Secret(
            client=client,
            name="router-certs-default",
            namespace="openshift-ingress",
        )
        filename = OPENSHIFT_CA_BUNDLE_FILENAME

    else:
        raise ValueError("Invalid ca_type")

    if certs_secret.exists:
        bundle = base64.b64decode(certs_secret.instance.data["tls.crt"]).decode()
        filepath = os.path.join(py_config["tmp_base_dir"], filename)
        with open(filepath, "w") as fd:
            fd.write(bundle)

        return filepath

    LOGGER.warning(f"Could not find {certs_secret.name} secret")
    return ""


@cache
def get_ca_bundle(client: DynamicClient, deployment_mode: str) -> str:
    """
    Get the ca bundle for the given deployment mode.

    If running on managed cluster and deployment in serverless or raw deployment, return empty string.
    If running on self-managed operator and deployment is model mesh, return ca bundle.

    Args:
        client (DynamicClient): DynamicClient object
        deployment_mode (str): The deployment mode. Can be "serverless", "model-mesh" or "raw-deployment"

    Returns:
        str: The path to the ca bundle file. If cert is not created, return empty string

    Raises:
            ValueError: If deployment_mode is not "serverless", "model-mesh" or "raw-deployment"

    """
    if deployment_mode in (
        KServeDeploymentType.SERVERLESS,
        KServeDeploymentType.RAW_DEPLOYMENT,
    ):
        if is_managed_cluster(client):
            LOGGER.info("Running on managed cluster, not using ca bundle")
            return ""
        else:
            return create_ca_bundle_file(client=client, ca_type="knative")

    elif deployment_mode == KServeDeploymentType.MODEL_MESH:
        if is_self_managed_operator(client=client):
            return create_ca_bundle_file(client=client, ca_type="openshift")

        return ""

    else:
        raise ValueError(f"Unknown deployment mode: {deployment_mode}")
