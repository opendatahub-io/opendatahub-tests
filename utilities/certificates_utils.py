from __future__ import annotations

import base64
import os
from functools import cache

from kubernetes.dynamic import DynamicClient
from ocp_resources.catalog_source import CatalogSource
from ocp_resources.secret import Secret
from pytest_testconfig import config as py_config

from utilities.constants import ISTIO_CA_BUNDLE_FILENAME, KServeDeploymentType, OPENSHIFT_CA_BUNDLE_FILENAME


@cache
def is_self_managed_operator(client: DynamicClient) -> bool:
    """
    Check if the operator is self-managed.
    """
    if py_config["distribution"] == "upstream":
        return True

    if CatalogSource(
        client=client,
        name="addon-managed-odh-catalog",
        namespace=py_config["applications_namespace"],
    ).exists:
        return True

    return False


def create_ca_bundle_file(client: DynamicClient, ca_type: str) -> str:
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

    raise Exception(f"Could not find {certs_secret.name} secret")


@cache
def get_ca_bundle(client: DynamicClient, deployment_mode: str) -> str:
    if deployment_mode in (KServeDeploymentType.SERVERLESS, KServeDeploymentType.RAW_DEPLOYMENT):
        return create_ca_bundle_file(client=client, ca_type="knative")

    elif deployment_mode == KServeDeploymentType.MODEL_MESH and is_self_managed_operator(client=client):
        return create_ca_bundle_file(client=client, ca_type="openshift")

    else:
        raise ValueError(f"Unknown deployment mode: {deployment_mode}")
