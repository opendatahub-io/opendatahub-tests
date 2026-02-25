"""Fixtures for Model Registry Python Client Signing Tests."""

import json
from collections.abc import Generator
from typing import Any

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.subscription import Subscription
from ocp_utilities.operators import install_operator, uninstall_operator
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_registry.model_registry.python_client.signing.constants import (
    SECURESIGN_API_VERSION,
    SECURESIGN_NAME,
    SECURESIGN_NAMESPACE,
    TAS_CONNECTION_TYPE_NAME,
)
from tests.model_registry.model_registry.python_client.signing.utils import (
    create_connection_type_field,
    get_organization_config,
    get_tas_service_urls,
    is_securesign_ready,
)
from utilities.constants import OPENSHIFT_OPERATORS, Timeout
from utilities.infra import get_openshift_token
from utilities.resources.securesign import Securesign

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def oidc_issuer_url(admin_client: DynamicClient, api_server_url: str) -> str:
    """Get the OIDC issuer URL from cluster's .well-known/openid-configuration endpoint.

    Args:
        admin_client: Kubernetes dynamic client
        api_server_url: Kubernetes API server URL

    Returns:
        str: OIDC issuer URL for keyless signing authentication
    """
    token = get_openshift_token(client=admin_client)
    url = f"{api_server_url}/.well-known/openid-configuration"
    headers = {"Authorization": f"Bearer {token}"}

    LOGGER.info(f"Fetching OIDC configuration from {url}")
    response = requests.get(url=url, headers=headers, verify=False, timeout=30)
    response.raise_for_status()

    oidc_config = response.json()
    issuer = oidc_config.get("issuer")

    assert issuer, "'issuer' field not found or empty in OIDC configuration response"

    LOGGER.info(f"Retrieved OIDC issuer URL: {issuer}")
    return issuer


@pytest.fixture(scope="class")
def installed_tas_operator(admin_client: DynamicClient) -> Generator[None, Any]:
    """Install Red Hat Trusted Artifact Signer (RHTAS/TAS) operator if not already installed.

    This fixture checks if TAS operator subscription exists in openshift-operators
    namespace. If not found, installs the operator from the appropriate catalog
    and removes it on teardown.
    If already installed, leaves it as-is without cleanup.

    Args:
        admin_client: Kubernetes dynamic client

    Yields:
        None: Operator is ready for use
    """
    distribution = py_config["distribution"]
    operator_ns = Namespace(name=OPENSHIFT_OPERATORS, ensure_exists=True)
    package_name = "rhtas-operator"

    # Determine operator source: ODH uses community-operators, RHOAI uses redhat-operators
    operator_source = "community-operators" if distribution == "upstream" else "redhat-operators"

    tas_operator_subscription = Subscription(client=admin_client, namespace=operator_ns.name, name=package_name)

    if not tas_operator_subscription.exists:
        LOGGER.info(f"TAS operator not found in {OPENSHIFT_OPERATORS}. Installing from {operator_source}...")
        install_operator(
            admin_client=admin_client,
            target_namespaces=None,  # All Namespaces
            name=package_name,
            channel="stable",
            source=operator_source,
            operator_namespace=operator_ns.name,
            timeout=Timeout.TIMEOUT_10MIN,
            install_plan_approval="Manual",  # TAS operator requires manual approval
        )

        # Wait for operator deployment to be ready
        deployment = Deployment(
            client=admin_client,
            namespace=operator_ns.name,
            name="rhtas-operator-controller-manager",
            wait_for_resource=True,
        )
        deployment.wait_for_replicas()
        LOGGER.info("TAS operator successfully installed")

        yield

        LOGGER.info("Uninstalling TAS operator (we installed it)")
        uninstall_operator(
            admin_client=admin_client,
            name=package_name,
            operator_namespace=operator_ns.name,
            clean_up_namespace=False,
        )
    else:
        LOGGER.info(f"TAS operator already installed in {OPENSHIFT_OPERATORS}. Using existing installation.")
        yield


@pytest.fixture(scope="class")
def securesign_instance(
    admin_client: DynamicClient, installed_tas_operator: None, oidc_issuer_url: str
) -> Generator[Securesign, Any]:
    """Create a Securesign instance with all Sigstore components in the trusted-artifact-signer namespace

    with the following components enabled:
    - Fulcio: Certificate authority with OIDC authentication
    - Rekor: Transparency log for signature records
    - CTLog: Certificate transparency log
    - TUF: The Update Framework for trust root distribution
    - TSA: Timestamp authority for RFC 3161 timestamps

    All components have external access enabled. Waits up to 5 minutes for the
    instance to reach Ready condition before yielding.

    Args:
        admin_client: Kubernetes dynamic client
        installed_tas_operator: TAS operator fixture ensuring operator is installed
        oidc_issuer_url: OIDC issuer URL for keyless signing authentication

    Yields:
        Resource: Securesign resource instance
    """
    # Ensure namespace exists for Securesign
    Namespace(name=SECURESIGN_NAMESPACE, ensure_exists=True)

    # Build Securesign CR spec
    org_config = get_organization_config()
    securesign_dict = {
        "apiVersion": SECURESIGN_API_VERSION,
        "kind": "Securesign",
        "metadata": {
            "name": SECURESIGN_NAME,
            "namespace": SECURESIGN_NAMESPACE,
        },
        "spec": {
            "fulcio": {
                "externalAccess": {"enabled": True},
                "certificate": org_config,
                "config": {
                    "MetaIssuers": [
                        {
                            "ClientID": oidc_issuer_url,
                            "Issuer": oidc_issuer_url,
                            "Type": "kubernetes",
                        }
                    ]
                },
            },
            "rekor": {
                "externalAccess": {"enabled": True},
            },
            "ctlog": {},
            "tuf": {
                "externalAccess": {"enabled": True},
            },
            "tsa": {
                "externalAccess": {"enabled": True},
                "signer": {
                    "certificateChain": {
                        "rootCA": org_config,
                        "intermediateCA": [org_config],
                        "leafCA": org_config,
                    }
                },
            },
        },
    }

    # Create Securesign instance using custom Securesign class
    with Securesign(kind_dict=securesign_dict, client=admin_client) as securesign:
        LOGGER.info(f"Securesign instance '{SECURESIGN_NAME}' created in namespace '{SECURESIGN_NAMESPACE}'")

        # Wait for the Securesign instance to become ready
        try:
            for sample in TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_5MIN,
                sleep=5,
                func=lambda: securesign.instance.to_dict(),
            ):
                if sample and is_securesign_ready(sample):
                    LOGGER.info(f"Securesign instance '{SECURESIGN_NAME}' is ready")
                    break
        except TimeoutExpiredError:
            LOGGER.error(f"Timeout waiting for Securesign instance '{SECURESIGN_NAME}' to become ready")
            raise

        yield securesign

    # Cleanup is handled automatically by the context manager
    LOGGER.info(f"Securesign instance '{SECURESIGN_NAME}' cleanup completed")


@pytest.fixture(scope="class")
def tas_connection_type(admin_client: DynamicClient, securesign_instance: Securesign) -> Generator[ConfigMap, Any]:
    """Create ODH Connection Type ConfigMap for TAS (Trusted Artifact Signer).

    Provides TAS service endpoints for programmatic access to signing services.
    The ConfigMap includes URLs for all Sigstore components (Fulcio, Rekor, TSA, TUF)

    Args:
        admin_client: Kubernetes dynamic client
        securesign_instance: Securesign instance fixture ensuring infrastructure is ready

    Yields:
        ConfigMap: TAS Connection Type ConfigMap
    """
    app_namespace = py_config["applications_namespace"]

    # Get Securesign instance to extract service URLs from status
    LOGGER.info("Retrieving TAS service URLs from Securesign instance...")
    securesign_data = securesign_instance.instance.to_dict()

    # Extract service URLs from Securesign status
    service_urls = get_tas_service_urls(securesign_instance=securesign_data)

    # Log and validate all URLs
    for service, url in service_urls.items():
        assert url, f"{service.replace('_', ' ').title()} URL not found"
        LOGGER.info(f"{service.replace('_', ' ').title()} URL: {url}")

    # Define Connection Type field specifications
    field_specs = [
        (
            "Fulcio URL",
            "Certificate authority service URL for keyless signing",
            "SIGSTORE_FULCIO_URL",
            service_urls["fulcio"],
            True,
        ),
        (
            "Rekor URL",
            "Transparency log service URL for signature verification",
            "SIGSTORE_REKOR_URL",
            service_urls["rekor"],
            True,
        ),
        ("TSA URL", "Timestamp Authority service URL (RFC 3161)", "SIGSTORE_TSA_URL", service_urls["tsa"], True),
        ("TUF URL", "Trust root distribution service URL", "SIGSTORE_TUF_URL", service_urls["tuf"], True),
    ]

    # Build Connection Type fields
    fields = [create_connection_type_field(name, desc, env, url, req) for name, desc, env, url, req in field_specs]

    # Create ConfigMap for Connection Type
    configmap_data = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": TAS_CONNECTION_TYPE_NAME,
            "namespace": app_namespace,
            "labels": {
                "opendatahub.io/connection-type": "true",
                "opendatahub.io/dashboard": "true",
                "app.opendatahub.io/dashboard": "true",
                "app": "odh-dashboard",
                "app.kubernetes.io/part-of": "dashboard",
            },
            "annotations": {
                "openshift.io/display-name": "Red Hat Trusted Artifact Signer",
                "openshift.io/description": "Connect to RHTAS for keyless signing and verification using Sigstore svc",
            },
        },
        "data": {
            "category": json.dumps(["Artifact signing"]),
            "fields": json.dumps(fields),
        },
    }

    with ConfigMap(kind_dict=configmap_data, client=admin_client) as connection_type:
        LOGGER.info(f"TAS Connection Type '{TAS_CONNECTION_TYPE_NAME}' created in namespace '{app_namespace}'")
        yield connection_type

    LOGGER.info(f"TAS Connection Type '{TAS_CONNECTION_TYPE_NAME}' deleted from namespace '{app_namespace}'")
