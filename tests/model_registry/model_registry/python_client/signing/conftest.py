"""Fixtures for Model Registry Python Client Signing Tests."""

import json
import logging
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import requests
import shortuuid
import structlog
from huggingface_hub import snapshot_download
from kubernetes.dynamic import DynamicClient
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.signing import Signer
from model_registry.types import RegisteredModel
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.subscription import Subscription
from ocp_utilities.operators import install_operator, uninstall_operator
from pyhelper_utils.shell import run_command
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutSampler

from tests.model_registry.model_registry.async_job.constants import (
    ASYNC_UPLOAD_JOB_NAME,
    MODEL_SYNC_CONFIG,
)
from tests.model_registry.model_registry.async_job.utils import get_latest_job_pod
from tests.model_registry.model_registry.python_client.signing.constants import (
    IDENTITY_TOKEN_MOUNT_PATH,
    NATIVE_SIGNING_REPO,
    NATIVE_SIGNING_TAG,
    SECURESIGN_API_VERSION,
    SECURESIGN_NAME,
    SECURESIGN_NAMESPACE,
    SIGNING_MODEL_DATA,
    SIGNING_OCI_REPO_NAME,
    SIGNING_OCI_TAG,
    TAS_CONNECTION_TYPE_NAME,
)
from tests.model_registry.model_registry.python_client.signing.utils import (
    create_async_upload_job,
    create_connection_type_field,
    generate_token,
    get_base_async_job_env_vars,
    get_model_registry_host,
    get_oci_image_with_digest,
    get_oci_internal_endpoint,
    get_organization_config,
    get_root_checksum,
    get_tas_service_urls,
    run_minio_uploader_pod,
)
from utilities.constants import (
    OPENSHIFT_OPERATORS,
    ApiGroups,
    Labels,
    MinIo,
    ModelCarImage,
    OCIRegistry,
    Timeout,
)
from utilities.general import b64_encoded_string, get_s3_secret_dict
from utilities.infra import get_openshift_token, is_managed_cluster
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from utilities.resources.route import Route
from utilities.resources.securesign import Securesign

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="package")
def skip_if_not_managed_cluster(admin_client: DynamicClient) -> None:
    """
    Skip tests if the cluster is not managed.
    """
    if not is_managed_cluster(admin_client):
        pytest.skip("Skipping tests - cluster is not managed")

    LOGGER.info("Cluster is managed - proceeding with tests")


@pytest.fixture(scope="package")
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


@pytest.fixture(scope="package")
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
    operator_ns = Namespace(client=admin_client, name=OPENSHIFT_OPERATORS, ensure_exists=True)
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
        # Ensure namespace exists for Securesign
        ns = Namespace(client=admin_client, name=SECURESIGN_NAMESPACE)
        if ns.exists:
            ns.delete(wait=True)
    else:
        LOGGER.info(f"TAS operator already installed in {OPENSHIFT_OPERATORS}. Using existing installation.")
        yield


@pytest.fixture(scope="package")
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
    ns = Namespace(client=admin_client, name=SECURESIGN_NAMESPACE)
    ns.wait_for_status(status=Namespace.Status.ACTIVE)

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
                "enabled": True,
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
                "enabled": True,
                "externalAccess": {"enabled": True},
            },
            "ctlog": {
                "enabled": True,
            },
            "tuf": {
                "enabled": True,
                "externalAccess": {"enabled": True},
            },
            "tsa": {
                "enabled": True,
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
        securesign.wait_for_condition(condition="Ready", status="True")
        yield securesign

    # Cleanup is handled automatically by the context manager
    LOGGER.info(f"Securesign instance '{SECURESIGN_NAME}' cleanup completed")


@pytest.fixture(scope="package")
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


@pytest.fixture(scope="class")
def oci_registry_pod(
    admin_client: DynamicClient,
    oci_namespace: Namespace,
) -> Generator[Pod, Any]:
    """Create a simple OCI registry (Zot) pod with local emptyDir storage.

    Unlike oci_registry_pod_with_minio, this does not require MinIO — data is
    stored in an emptyDir volume, which is sufficient for signing test scenarios.

    Args:
        admin_client: Kubernetes dynamic client
        oci_namespace: Namespace for the OCI registry pod

    Yields:
        Pod: Ready OCI registry pod
    """
    with Pod(
        client=admin_client,
        name=OCIRegistry.Metadata.NAME,
        namespace=oci_namespace.name,
        containers=[
            {
                "env": [
                    {"name": "ZOT_HTTP_ADDRESS", "value": OCIRegistry.Metadata.DEFAULT_HTTP_ADDRESS},
                    {"name": "ZOT_HTTP_PORT", "value": str(OCIRegistry.Metadata.DEFAULT_PORT)},
                    {"name": "ZOT_LOG_LEVEL", "value": "info"},
                ],
                "image": OCIRegistry.PodConfig.REGISTRY_IMAGE,
                "name": OCIRegistry.Metadata.NAME,
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "volumeMounts": [
                    {
                        "name": "zot-data",
                        "mountPath": "/var/lib/registry",
                    }
                ],
            }
        ],
        volumes=[
            {
                "name": "zot-data",
                "emptyDir": {},
            }
        ],
        label={
            Labels.Openshift.APP: OCIRegistry.Metadata.NAME,
            "maistra.io/expose-route": "true",
        },
    ) as oci_pod:
        oci_pod.wait_for_condition(condition="Ready", status="True")
        yield oci_pod


@pytest.fixture(scope="class")
def ai_hub_oci_registry_route(admin_client: DynamicClient, oci_registry_service: Service) -> Generator[Route, Any]:
    """Override the default Route with edge TLS termination.

    Cosign requires HTTPS. Edge termination lets the OpenShift router handle TLS
    and forward plain HTTP to the Zot backend.
    """
    with Route(
        client=admin_client,
        name=OCIRegistry.Metadata.NAME,
        namespace=oci_registry_service.namespace,
        to={"kind": "Service", "name": oci_registry_service.name},
        tls={"termination": "edge", "insecureEdgeTerminationPolicy": "Redirect"},
    ) as oci_route:
        yield oci_route


@pytest.fixture(scope="class")
def ai_hub_oci_registry_host(ai_hub_oci_registry_route: Route) -> str:
    """Get the OCI registry host from the route."""
    return ai_hub_oci_registry_route.instance.spec.host


@pytest.fixture(scope="class")
def downloaded_model_dir() -> Path:
    """Download a test model from Hugging Face to a temporary directory.

    Downloads the jonburdo/public-test-model-1 model to a temporary directory
    and yields the path to the downloaded model directory.

    Yields:
        Path: Path to the temporary directory containing the downloaded model
    """
    model_dir = Path(py_config["tmp_base_dir"]) / "model"
    model_dir.mkdir(exist_ok=True)

    LOGGER.info(f"Downloading model to temporary directory: {model_dir}")
    snapshot_download(repo_id="jonburdo/public-test-model-1", local_dir=str(model_dir))
    LOGGER.info(f"Model downloaded successfully to: {model_dir}")

    return model_dir


@pytest.fixture(scope="class")
def set_environment_variables(securesign_instance: Securesign) -> Generator[None, Any]:
    """
    Create a service account token and save it to a temporary directory.
    Automatically cleans up environment variables when fixture scope ends.
    """
    # Set up environment variables
    securesign_data = securesign_instance.instance.to_dict()
    service_urls = get_tas_service_urls(securesign_instance=securesign_data)
    os.environ["IDENTITY_TOKEN_PATH"] = generate_token(temp_base_folder=py_config["tmp_base_dir"])
    os.environ["SIGSTORE_TUF_URL"] = service_urls["tuf"]
    os.environ["SIGSTORE_FULCIO_URL"] = service_urls["fulcio"]
    os.environ["SIGSTORE_REKOR_URL"] = service_urls["rekor"]
    os.environ["SIGSTORE_TSA_URL"] = service_urls["tsa"]
    os.environ["ROOT_CHECKSUM"] = get_root_checksum(sigstore_tuf_url=service_urls["tuf"])
    os.environ["ROOT_URL"] = os.environ["SIGSTORE_TUF_URL"] + "/root.json"

    LOGGER.info("Environment variables set for signing tests")
    yield

    # Clean up environment variables
    for var_name in [
        "IDENTITY_TOKEN_PATH",
        "SIGSTORE_TUF_URL",
        "SIGSTORE_FULCIO_URL",
        "SIGSTORE_REKOR_URL",
        "SIGSTORE_TSA_URL",
        "ROOT_CHECKSUM",
        "ROOT_URL",
    ]:
        os.environ.pop(var_name, None)

    LOGGER.info("Environment variables cleaned up")


@pytest.fixture(scope="function")
def signer(set_environment_variables) -> Signer:
    """Create and initialize a Signer instance for model signing.

    Creates a Signer with identity token, root URL, and root checksum from environment
    variables set by the set_environment_variables fixture. Initializes the signer
    with force=True and debug logging.

    Args:
        set_environment_variables: Fixture that sets up required environment variables

    Returns:
        Signer: Initialized signer instance ready for model signing

    Raises:
        Exception: If signer initialization fails
    """
    LOGGER.info(f"Creating Signer with token path: {os.environ['IDENTITY_TOKEN_PATH']}")
    LOGGER.info(f"Root URL: {os.environ['ROOT_URL']}")

    signer = Signer(
        identity_token_path=os.environ["IDENTITY_TOKEN_PATH"],
        root_url=os.environ["ROOT_URL"],
        root_checksum=os.environ["ROOT_CHECKSUM"],
        log_level=logging.DEBUG,
    )

    LOGGER.info("Initializing signer...")
    signer.initialize(force=True)
    LOGGER.info("Signer initialized successfully")

    return signer


@pytest.fixture(scope="class")
def copied_model_to_oci_registry(
    oci_registry_pod: Pod,
    ai_hub_oci_registry_host: str,
) -> Generator[str, Any]:
    """Copy ModelCarImage.MNIST_8_1 from quay.io to the local OCI registry using skopeo.

    Sets COSIGN_ALLOW_INSECURE_REGISTRY so cosign can access the registry over
    the edge-terminated Route with a self-signed certificate.

    Args:
        oci_registry_pod: OCI registry pod fixture ensuring registry is running
        ai_hub_oci_registry_host: OCI registry hostname from route

    Yields:
        str: The destination image reference with digest (e.g. "{host}/{repo}@sha256:...")
    """
    # Wait for the OCI registry to be reachable via the Route
    registry_url = f"https://{ai_hub_oci_registry_host}"
    LOGGER.info(f"Waiting for OCI registry to be reachable at {registry_url}/v2/")
    for sample in TimeoutSampler(
        wait_timeout=120,
        sleep=5,
        func=requests.get,
        url=f"{registry_url}/v2/",
        timeout=5,
        verify=False,
    ):
        if sample.ok:
            LOGGER.info("OCI registry is reachable")
            break

    source_image = ModelCarImage.MNIST_8_1.removeprefix("oci://")
    dest_ref = f"{ai_hub_oci_registry_host}/{SIGNING_OCI_REPO_NAME}:{SIGNING_OCI_TAG}"

    LOGGER.info(f"Copying image from docker://{source_image} to docker://{dest_ref}")
    run_command(
        command=[
            "skopeo",
            "copy",
            "--dest-tls-verify=false",
            f"docker://{source_image}",
            f"docker://{dest_ref}",
        ],
        check=True,
    )
    LOGGER.info(f"Image copied successfully to {dest_ref}")

    # Get the digest of the pushed image
    _, inspect_out, _ = run_command(
        command=[
            "skopeo",
            "inspect",
            "--tls-verify=false",
            f"docker://{dest_ref}",
        ],
        check=True,
    )
    digest = json.loads(inspect_out).get("Digest", "")
    LOGGER.info(f"Pushed image {inspect_out} digest: {digest}")

    dest_with_digest = f"{ai_hub_oci_registry_host}/{SIGNING_OCI_REPO_NAME}@{digest}"
    LOGGER.info(f"Full image reference: {dest_with_digest}")

    # Set cosign env var to allow insecure registry access (self-signed cert from edge Route)
    os.environ["COSIGN_ALLOW_INSECURE_REGISTRY"] = "true"
    LOGGER.info("Set COSIGN_ALLOW_INSECURE_REGISTRY=true")

    yield dest_with_digest

    # Cleanup
    os.environ.pop("COSIGN_ALLOW_INSECURE_REGISTRY", None)
    LOGGER.info("Cleaned up COSIGN_ALLOW_INSECURE_REGISTRY")


@pytest.fixture(scope="function")
def signed_model(signer, downloaded_model_dir) -> Path:
    """
    Use an initialized signer to sign the downloaded model.
    """
    LOGGER.info(f"Signing model in directory: {downloaded_model_dir}")
    signer.sign_model(model_path=str(downloaded_model_dir))
    LOGGER.info("Model signed successfully")

    return downloaded_model_dir


@pytest.fixture(scope="class")
def signing_s3_secret(
    admin_client: DynamicClient,
    service_account: ServiceAccount,
    minio_service: Service,
) -> Generator[Secret, Any, Any]:
    """Create S3 data connection for signing async upload jobs."""
    minio_endpoint = (
        f"http://{minio_service.name}.{minio_service.namespace}.svc.cluster.local:{MinIo.Metadata.DEFAULT_PORT}"
    )

    with Secret(
        client=admin_client,
        name=f"signing-s3-{shortuuid.uuid().lower()}",
        namespace=service_account.namespace,
        data_dict=get_s3_secret_dict(
            aws_access_key=MinIo.Credentials.ACCESS_KEY_VALUE,
            aws_secret_access_key=MinIo.Credentials.SECRET_KEY_VALUE,
            aws_s3_bucket=MinIo.Buckets.MODELMESH_EXAMPLE_MODELS,
            aws_s3_endpoint=minio_endpoint,
            aws_default_region="us-east-1",
        ),
        label={
            Labels.OpenDataHub.DASHBOARD: "true",
            Labels.OpenDataHubIo.MANAGED: "true",
        },
        annotations={
            f"{ApiGroups.OPENDATAHUB_IO}/connection-type": "s3",
            "openshift.io/display-name": "Signing S3 Credentials",
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def signing_oci_secret(
    admin_client: DynamicClient,
    service_account: ServiceAccount,
    oci_registry_service: Service,
) -> Generator[Secret, Any, Any]:
    """Create OCI registry data connection for signing async upload jobs."""
    oci_internal_endpoint = get_oci_internal_endpoint(oci_registry_service=oci_registry_service)

    dockerconfig = {
        "auths": {
            oci_internal_endpoint: {
                "auth": "",
                "email": "user@example.com",
            }
        }
    }

    data_dict = {
        ".dockerconfigjson": b64_encoded_string(json.dumps(dockerconfig)),
        "ACCESS_TYPE": b64_encoded_string(json.dumps('["Push,Pull"]')),
        "OCI_HOST": b64_encoded_string(json.dumps(oci_internal_endpoint)),
    }

    with Secret(
        client=admin_client,
        name=f"signing-oci-{shortuuid.uuid().lower()}",
        namespace=service_account.namespace,
        data_dict=data_dict,
        label={
            Labels.OpenDataHub.DASHBOARD: "true",
            Labels.OpenDataHubIo.MANAGED: "true",
        },
        annotations={
            f"{ApiGroups.OPENDATAHUB_IO}/connection-type-ref": "oci-v1",
            "openshift.io/display-name": "Signing OCI Credentials",
        },
        type="kubernetes.io/dockerconfigjson",
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def signing_registered_model(
    model_registry_client: list[ModelRegistryClient],
) -> RegisteredModel:
    """Register a model in Model Registry for signing async tests."""
    return model_registry_client[0].register_model(
        name=SIGNING_MODEL_DATA["model_name"],
        uri=SIGNING_MODEL_DATA.get("model_uri"),
        version=SIGNING_MODEL_DATA.get("model_version"),
        version_description=SIGNING_MODEL_DATA.get("model_description"),
        model_format_name=SIGNING_MODEL_DATA.get("model_format"),
        model_format_version=SIGNING_MODEL_DATA.get("model_format_version"),
        storage_key=SIGNING_MODEL_DATA.get("model_storage_key"),
        storage_path=SIGNING_MODEL_DATA.get("model_storage_path"),
    )


@pytest.fixture(scope="class")
def upload_unsigned_model_to_minio(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    minio_service: Service,
) -> None:
    """Upload an unsigned model file to MinIO for native job signing test."""
    source_key = MODEL_SYNC_CONFIG["SOURCE_AWS_KEY"]
    bucket = MinIo.Buckets.MODELMESH_EXAMPLE_MODELS

    run_minio_uploader_pod(
        admin_client=admin_client,
        namespace=model_registry_namespace,
        minio_service=minio_service,
        pod_name="unsigned-model-uploader",
        mc_commands=(
            f"echo 'test model content for native signing' > /work/model.onnx && "
            f"mc cp /work/model.onnx testminio/{bucket}/{source_key}/model.onnx && "
            f"mc ls testminio/{bucket}/{source_key}/ && "
            f"echo 'Unsigned model upload completed'"
        ),
    )


@pytest.fixture(scope="class")
def identity_token_secret(
    admin_client: DynamicClient,
    service_account: ServiceAccount,
    securesign_instance: Securesign,
) -> Generator[Secret, Any, Any]:
    """Create a Secret containing a service account token for Sigstore keyless signing.

    The async upload job needs this token mounted at a known path so it can
    authenticate with Fulcio for keyless signing.
    """
    token = generate_token(temp_base_folder=py_config["tmp_base_dir"])
    with open(token) as f:
        token_content = f.read()

    with Secret(
        client=admin_client,
        name=f"signing-identity-token-{shortuuid.uuid().lower()}",
        namespace=service_account.namespace,
        data_dict={
            "token": b64_encoded_string(token_content),
        },
    ) as secret:
        LOGGER.info(f"Created identity token secret: {secret.name}")
        yield secret


@pytest.fixture(scope="class")
def native_signing_async_job(
    admin_client: DynamicClient,
    sa_token: str,
    service_account: ServiceAccount,
    model_registry_namespace: str,
    model_registry_instance: list[ModelRegistry],
    signing_s3_secret: Secret,
    signing_oci_secret: Secret,
    oci_registry_service: Service,
    mr_access_role_binding: RoleBinding,
    async_upload_image: str,
    signing_registered_model: RegisteredModel,
    upload_unsigned_model_to_minio: None,
    identity_token_secret: Secret,
    securesign_instance: Securesign,
    teardown_resources: bool,
) -> Generator[Job, Any, Any]:
    """Create and run the async upload job with native signing enabled.

    Configures the job with MODEL_SYNC_SIGN=true and Sigstore environment
    variables so the job signs both the model and the OCI image internally.
    """
    mr_host = get_model_registry_host(
        admin_client=admin_client,
        model_registry_namespace=model_registry_namespace,
        model_registry_instance=model_registry_instance,
    )
    oci_internal = get_oci_internal_endpoint(oci_registry_service=oci_registry_service)

    # Get Sigstore service URLs for the job
    securesign_data = securesign_instance.instance.to_dict()
    service_urls = get_tas_service_urls(securesign_instance=securesign_data)

    signing_env_vars = [
        {"name": "MODEL_SYNC_SIGN", "value": "true"},
        {"name": "MODEL_SYNC_SIGNING_IDENTITY_TOKEN_PATH", "value": f"{IDENTITY_TOKEN_MOUNT_PATH}/token"},
        {"name": "SIGSTORE_FULCIO_URL", "value": service_urls["fulcio"]},
        {"name": "SIGSTORE_REKOR_URL", "value": service_urls["rekor"]},
        {"name": "SIGSTORE_TSA_URL", "value": service_urls["tsa"]},
        {"name": "SIGSTORE_TUF_URL", "value": service_urls["tuf"]},
        {"name": "COSIGN_ALLOW_INSECURE_REGISTRY", "value": "true"},
    ]

    yield from create_async_upload_job(
        admin_client=admin_client,
        job_name=f"{ASYNC_UPLOAD_JOB_NAME}-native-signing",
        namespace=service_account.namespace,
        async_upload_image=async_upload_image,
        s3_secret=signing_s3_secret,
        oci_secret=signing_oci_secret,
        environment_variables=get_base_async_job_env_vars(
            mr_host=mr_host,
            sa_token=sa_token,
            oci_internal=oci_internal,
            oci_repo=NATIVE_SIGNING_REPO,
        )
        + signing_env_vars,
        extra_volume_mounts=[
            {"name": "identity-token", "readOnly": True, "mountPath": IDENTITY_TOKEN_MOUNT_PATH},
        ],
        extra_volumes=[
            {"name": "identity-token", "secret": {"secretName": identity_token_secret.name}},
        ],
        teardown=teardown_resources,
    )


@pytest.fixture(scope="class")
def native_signing_job_pod(
    admin_client: DynamicClient,
    native_signing_async_job: Job,
) -> Pod:
    """Get the pod created by the native signing async job."""
    return get_latest_job_pod(admin_client=admin_client, job=native_signing_async_job)


@pytest.fixture(scope="class")
def native_signing_oci_image_with_digest(
    ai_hub_oci_registry_host: str,
    native_signing_async_job: Job,
) -> Generator[str, Any, Any]:
    """Get the OCI image reference with digest after native signing job completes."""
    yield from get_oci_image_with_digest(
        oci_host=ai_hub_oci_registry_host,
        repo=NATIVE_SIGNING_REPO,
        tag=NATIVE_SIGNING_TAG,
    )
