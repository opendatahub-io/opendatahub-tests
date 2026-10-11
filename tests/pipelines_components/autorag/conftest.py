import base64
import uuid
from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.secret import Secret

from tests.pipelines_components.autorag.utils import submit_autorag_run, upload_autorag_data
from tests.pipelines_components.constants import (
    DSPA_MINIO_IMAGE,
    DSPA_NAME,
    DSPA_PIPELINE_DEPLOYMENT,
    DSPA_READY_BUFFER_SECONDS,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    MANAGED_PIPELINE_AUTORAG,
    MANAGED_PIPELINE_POLL_INTERVAL,
    MANAGED_PIPELINE_WAIT_TIMEOUT,
    MANAGED_PIPELINES_IMAGE,
)
from tests.pipelines_components.utils import wait_for_managed_pipeline
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import DscComponents, Timeout
from utilities.infra import create_ns, wait_for_dsc_status_ready


@pytest.fixture(scope="class")
def pipelines_namespace(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:  # noqa: UFN001
    """Namespace for the DSPA and its pipeline connection secrets."""
    with create_ns(admin_client=admin_client, name=f"autorag-aqa-{uuid.uuid4().hex[:8]}") as namespace:
        assert isinstance(namespace, Namespace)
        yield namespace


@pytest.fixture(scope="class")
def autorag_maas_secret(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Provide the required MaaS Secret shape with a non-routable endpoint."""
    with Secret(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name="autorag-maas-connection",
        string_data={
            "MAAS_BASE_URL": "http://maas.invalid/v1",
            "MAAS_API_KEY": "placeholder",  # pragma: allowlist secret
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def autorag_db_secret(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Provide the required Milvus Secret shape without a database service."""
    with Secret(
        client=admin_client,
        namespace=pipelines_namespace.name,
        name="autorag-vector-db-connection",
        string_data={
            "MILVUS_URI": "http://milvus.invalid:19530",
            "MILVUS_TOKEN": "placeholder",  # pragma: allowlist secret
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def autorag_managed_pipeline(
    dspa: DataSciencePipelinesApplication,
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
) -> dict[str, str]:
    """Discover the managed Documents RAG Optimization pipeline."""
    return wait_for_managed_pipeline(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        display_name=MANAGED_PIPELINE_AUTORAG,
        ca_bundle=dspa_ca_bundle_file,
        timeout=DSPA_READY_BUFFER_SECONDS + MANAGED_PIPELINE_WAIT_TIMEOUT,
        poll_interval=MANAGED_PIPELINE_POLL_INTERVAL,
    )


@pytest.fixture(scope="class")
def autorag_test_data(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa_s3_credentials: Secret,
) -> None:
    """Upload a local document and matching benchmark record to DSPA MinIO."""
    upload_autorag_data(client=admin_client, namespace=pipelines_namespace.name)


@pytest.fixture(scope="class")
def autorag_run_id(
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    autorag_managed_pipeline: dict[str, str],
    autorag_maas_secret: Secret,
    autorag_db_secret: Secret,
    dspa_s3_credentials: Secret,
    autorag_test_data: None,
) -> str:
    """Submit a managed AutoRAG run after its inputs and connections are ready."""
    return submit_autorag_run(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        ca_bundle=dspa_ca_bundle_file,
        pipeline=autorag_managed_pipeline,
        s3_secret_name=dspa_s3_credentials.name,
        maas_secret_name=autorag_maas_secret.name,
        db_secret_name=autorag_db_secret.name,
        run_name=f"autorag-smoke-{uuid.uuid4().hex[:8]}",
    )


@pytest.fixture(scope="session")
def autorag_upgrade_dsc(
    pytestconfig: pytest.Config,
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    """Keep AI Pipelines managed until the post-upgrade namespace is removed."""
    if pytestconfig.option.pre_upgrade:
        current_state = dsc_resource.instance.spec.components.get("aipipelines", {}).get("managementState")
        if current_state != DscComponents.ManagementState.MANAGED:
            assert current_state == DscComponents.ManagementState.REMOVED, (
                f"Unexpected AI Pipelines managementState: {current_state}"
            )
            ResourceEditor(
                patches={dsc_resource: {"spec": {"components": {"aipipelines": {"managementState": "Managed"}}}}}
            ).update()
            wait_for_dsc_status_ready(dsc_resource=dsc_resource)
    if pytestconfig.option.post_upgrade:
        current_state = dsc_resource.instance.spec.components.get("aipipelines", {}).get("managementState")
        assert current_state == DscComponents.ManagementState.MANAGED, (
            f"Expected AI Pipelines to remain Managed through upgrade, got {current_state}"
        )
    yield dsc_resource
    if pytestconfig.option.post_upgrade:
        current_state = dsc_resource.instance.spec.components.get("aipipelines", {}).get("managementState")
        if current_state == DscComponents.ManagementState.MANAGED:
            ResourceEditor(
                patches={dsc_resource: {"spec": {"components": {"aipipelines": {"managementState": "Removed"}}}}}
            ).update()


@pytest.fixture(scope="session")
def autorag_upgrade_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_dsc: DataScienceCluster,
) -> Generator[Namespace, Any, Any]:
    """Preserve the AutoRAG namespace until post-upgrade verification finishes."""
    name = "autorag-aqa-upgrade"
    if pytestconfig.option.pre_upgrade:
        assert not Namespace(client=admin_client, name=name).exists, f"Namespace {name} already exists"
        with create_ns(admin_client=admin_client, name=name, teardown=False) as namespace:
            assert isinstance(namespace, Namespace)
            yield namespace
    else:
        namespace = Namespace(client=admin_client, name=name, ensure_exists=True)
        yield namespace
        if pytestconfig.option.post_upgrade:
            namespace.clean_up()


@pytest.fixture(scope="session")
def autorag_upgrade_dspa(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
) -> Generator[DataSciencePipelinesApplication, Any, Any]:
    """Create the managed DSPA before upgrade and reuse it afterward."""
    if pytestconfig.option.pre_upgrade:
        managed_pipelines_spec: dict[str, Any] = {}
        if MANAGED_PIPELINES_IMAGE:
            managed_pipelines_spec["image"] = MANAGED_PIPELINES_IMAGE
        with DataSciencePipelinesApplication(
            client=admin_client,
            name=DSPA_NAME,
            namespace=autorag_upgrade_namespace.name,
            dsp_version="v2",
            api_server={"enableSamplePipeline": False, "managedPipelines": managed_pipelines_spec},
            object_storage={
                "disableHealthCheck": False,
                "enableExternalRoute": False,
                "minio": {"deploy": True, "image": DSPA_MINIO_IMAGE},
            },
            teardown=False,
        ) as dspa_resource:
            Deployment(
                client=admin_client, name=DSPA_PIPELINE_DEPLOYMENT, namespace=autorag_upgrade_namespace.name
            ).wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
            yield dspa_resource
    else:
        dspa_resource = DataSciencePipelinesApplication(
            client=admin_client, name=DSPA_NAME, namespace=autorag_upgrade_namespace.name, ensure_exists=True
        )
        Deployment(
            client=admin_client, name=DSPA_PIPELINE_DEPLOYMENT, namespace=autorag_upgrade_namespace.name
        ).wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
        yield dspa_resource
        if pytestconfig.option.post_upgrade:
            dspa_resource.clean_up()


@pytest.fixture(scope="session")
def autorag_upgrade_api_url(admin_client: DynamicClient, autorag_upgrade_dspa: DataSciencePipelinesApplication) -> str:
    """Return the KFP API URL for the persistent DSPA."""
    route = Route(
        client=admin_client, name=DSPA_PIPELINE_DEPLOYMENT, namespace=autorag_upgrade_dspa.namespace, ensure_exists=True
    )
    return f"https://{route.host}"


@pytest.fixture(scope="session")
def autorag_upgrade_auth_headers(current_client_token: str) -> dict[str, str]:
    """Return KFP API authentication headers."""
    return {"Authorization": f"Bearer {current_client_token}"}


@pytest.fixture(scope="session")
def autorag_upgrade_ca_bundle(admin_client: DynamicClient) -> str:
    """Return the cluster CA bundle for the KFP Route."""
    return create_ca_bundle_file(client=admin_client)


@pytest.fixture(scope="session")
def autorag_upgrade_pipeline(
    autorag_upgrade_api_url: str,
    autorag_upgrade_auth_headers: dict[str, str],
    autorag_upgrade_ca_bundle: str,
) -> dict[str, str]:
    """Discover the managed Documents RAG Optimization pipeline."""
    return wait_for_managed_pipeline(
        api_url=autorag_upgrade_api_url,
        headers=autorag_upgrade_auth_headers,
        display_name=MANAGED_PIPELINE_AUTORAG,
        ca_bundle=autorag_upgrade_ca_bundle,
        timeout=DSPA_READY_BUFFER_SECONDS + MANAGED_PIPELINE_WAIT_TIMEOUT,
        poll_interval=MANAGED_PIPELINE_POLL_INTERVAL,
    )


@pytest.fixture(scope="session")
def autorag_upgrade_s3_secret(
    admin_client: DynamicClient,
    autorag_upgrade_dspa: DataSciencePipelinesApplication,
) -> Secret:
    """Add AWS-style fields to the DSPA MinIO connection Secret."""
    secret = Secret(
        client=admin_client, name=DSPA_S3_SECRET, namespace=autorag_upgrade_dspa.namespace, ensure_exists=True
    )
    access_key = base64.b64decode(secret.instance.data.get("accesskey", "")).decode()
    secret_key = base64.b64decode(secret.instance.data.get("secretkey", "")).decode()
    endpoint = f"http://minio-{DSPA_NAME}.{autorag_upgrade_dspa.namespace}.svc.cluster.local:9000"
    secret.update(
        resource_dict={
            "metadata": {"name": secret.name, "namespace": autorag_upgrade_dspa.namespace},
            "stringData": {
                "AWS_ACCESS_KEY_ID": access_key,
                "AWS_SECRET_ACCESS_KEY": secret_key,
                "AWS_S3_ENDPOINT": endpoint,
                "AWS_S3_BUCKET": DSPA_S3_BUCKET,
                "AWS_DEFAULT_REGION": "us-east-1",
            },
        }
    )
    return secret


@pytest.fixture(scope="session")
def autorag_upgrade_maas_secret(
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Keep the placeholder MaaS connection Secret across the upgrade."""
    with Secret(
        client=admin_client,
        namespace=autorag_upgrade_namespace.name,
        name="autorag-maas-connection",
        string_data={
            "MAAS_BASE_URL": "http://maas.invalid/v1",
            "MAAS_API_KEY": "placeholder",  # pragma: allowlist secret
        },
        teardown=False,
    ) as secret:
        yield secret


@pytest.fixture(scope="session")
def autorag_upgrade_db_secret(
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Keep the placeholder Milvus connection Secret across the upgrade."""
    with Secret(
        client=admin_client,
        namespace=autorag_upgrade_namespace.name,
        name="autorag-vector-db-connection",
        string_data={
            "MILVUS_URI": "http://milvus.invalid:19530",
            "MILVUS_TOKEN": "placeholder",  # pragma: allowlist secret
        },
        teardown=False,
    ) as secret:
        yield secret


@pytest.fixture(scope="function")
def autorag_upgrade_test_data(
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_s3_secret: Secret,
) -> None:
    """Upload local documents and benchmark to the persistent DSPA."""
    upload_autorag_data(client=admin_client, namespace=autorag_upgrade_namespace.name)


@pytest.fixture(scope="function")
def autorag_upgrade_run_id(
    autorag_upgrade_api_url: str,
    autorag_upgrade_auth_headers: dict[str, str],
    autorag_upgrade_ca_bundle: str,
    autorag_upgrade_pipeline: dict[str, str],
    autorag_upgrade_s3_secret: Secret,
    autorag_upgrade_maas_secret: Secret,
    autorag_upgrade_db_secret: Secret,
    autorag_upgrade_test_data: None,
) -> str:
    """Submit a managed run with the same inputs as the AutoRAG smoke test."""
    return submit_autorag_run(
        api_url=autorag_upgrade_api_url,
        headers=autorag_upgrade_auth_headers,
        ca_bundle=autorag_upgrade_ca_bundle,
        pipeline=autorag_upgrade_pipeline,
        s3_secret_name=autorag_upgrade_s3_secret.name,
        maas_secret_name=autorag_upgrade_maas_secret.name,
        db_secret_name=autorag_upgrade_db_secret.name,
        run_name="autorag-upgrade",
    )
