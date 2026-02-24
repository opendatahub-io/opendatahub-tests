import base64
import os
from functools import cache

from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.secret import Secret
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from utilities.constants import (
    ISTIO_CA_BUNDLE_FILENAME,
    OPENSHIFT_CA_BUNDLE_FILENAME,
    KServeDeploymentType,
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


def create_k8s_secret(
    client: DynamicClient,
    namespace: str,
    name: str,
    file_path: str,
    key_name: str,
) -> Secret:
    """
    Creates a Kubernetes secret from a file.

    Args:
        client: The admin client to create the ConfigMap
        namespace: The namespace of the model registry
        name: The name of the secret
        file_path: The path to the file to create the secret from
        key_name: The key name to use for the secret

    Returns:
        Secret: The created secret
    """
    with open(file_path, "rb") as f:
        file_content_raw_bytes = f.read()
        file_content_b64_string = base64.b64encode(file_content_raw_bytes).decode("utf-8")
        data = {key_name: file_content_b64_string}
    secret = Secret(
        client=client,
        name=name,
        namespace=namespace,
        data_dict=data,
        wait_for_resource=True,
    )
    secret.create()
    return secret


def create_ca_bundle_with_router_cert(
    client: DynamicClient,
    namespace: str,
    ca_bundle_path: str,
    cert_name: str,
) -> None:
    """
    Creates a CA bundle file by fetching the CA bundle from a ConfigMap and appending the router CA from a Secret.

    Args:
        client: The client to get the CA bundle from a ConfigMap and append the router CA from a Secret.
        namespace: The namespace of the ConfigMap and Secret.
        ca_bundle_path: The path to the CA bundle file.
        cert_name: The name of the certificate in the ConfigMap.

    Returns:
        None
    """
    cm = ConfigMap(client=client, name="odh-trusted-ca-bundle", namespace=namespace, ensure_exists=True)
    ca_bundle_content = cm.instance.data.get(cert_name)
    with open(ca_bundle_path, "w", encoding="utf-8") as f:
        f.write(ca_bundle_content)

    router_secret = Secret(client=client, name="router-ca", namespace="openshift-ingress-operator", ensure_exists=True)
    router_ca_b64 = router_secret.instance.data.get("tls.crt")
    if router_ca_b64:
        router_ca_content = base64.b64decode(router_ca_b64).decode("utf-8")
        with open(ca_bundle_path, "r", encoding="utf-8") as bundle:
            bundle_content = bundle.read()
        if router_ca_content not in bundle_content:
            with open(ca_bundle_path, "a", encoding="utf-8") as bundle_append:
                bundle_append.write("\n" + router_ca_content)
