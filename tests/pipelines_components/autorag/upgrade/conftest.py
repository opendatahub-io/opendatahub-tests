import base64
import os
import shlex
import uuid
from collections.abc import Generator
from typing import Any

import httpx
import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ogx_client import OgxClient
from timeout_sampler import TimeoutExpiredError

from tests.pipelines_components.autorag.upgrade.utils import (
    UPGRADE_BASELINE_CONFIGMAP,
    discover_vector_store_ids,
    load_baseline_from_configmap,
    save_baseline_to_configmap,
)
from tests.pipelines_components.autorag.utils import (
    AUTORAG_EMBEDDING_MODEL_NAME,
    AUTORAG_EMBEDDING_MODEL_URI,
    AUTORAG_INFERENCE_MODEL_NAME,
    AUTORAG_INFERENCE_MODEL_URI,
    AUTORAG_OGX_SECRET_DATA,
    OGX_CLIENT_VERIFY_SSL,
    create_ogx_server,
    get_etcd_template,
    get_milvus_template,
    get_postgres_template,
    log_registered_models,
    resolve_model_id,
    wait_for_ogx_client_ready,
    wait_for_vllm_model_ready,
)
from tests.pipelines_components.constants import (
    AUTORAG_EMBEDDING_MAX_MODEL_LEN,
    AUTORAG_INPUT_DATA_KEY,
    AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID,
    AUTORAG_MAX_RAG_PATTERNS,
    AUTORAG_OPTIMIZATION_METRIC,
    AUTORAG_PIPELINE_YAML,
    AUTORAG_S3_BUCKET,
    AUTORAG_TEST_DATA_KEY,
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
    MINIO_MC_IMAGE,
    MINIO_UPLOADER_SECURITY_CONTEXT,
)
from tests.pipelines_components.utils import (
    create_pipeline_run,
    create_pipeline_run_managed,
    resolve_pipeline_yaml,
    upload_pipeline,
    use_managed_pipelines,
    wait_for_managed_pipeline,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import (
    Annotations,
    DscComponents,
    KServeDeploymentType,
    RuntimeTemplates,
    Timeout,
)
from utilities.general import generate_random_name
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns, wait_for_dsc_status_ready
from utilities.resources.ogx_server import OgxServer
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_NAMESPACE = "autorag-aqa-upgrade"
UPGRADE_RUN_DISPLAY_NAME = "autorag-upgrade"
AUTORAG_RESOURCE_PREFIX = "autorag-upgrade"


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

_AUTORAG_REQUIRED_ENV = {
    "AUTORAG_INFERENCE_MODEL_URI": "Storage URI for inference model",
    "AUTORAG_INFERENCE_MODEL_NAME": "Inference model name",
    "AUTORAG_EMBEDDING_MODEL_URI": "Storage URI for embedding model",
    "AUTORAG_EMBEDDING_MODEL_NAME": "Embedding model name",
}


@pytest.fixture(scope="session", autouse=True)
def _validate_autorag_upgrade_env() -> None:
    missing = [f"  {var}: {desc}" for var, desc in _AUTORAG_REQUIRED_ENV.items() if not os.environ.get(var)]
    if missing:
        pytest.fail("AutoRAG upgrade test requires environment variables:\n" + "\n".join(missing))


# ---------------------------------------------------------------------------
# DSC patches — enable aipipelines + ogx before upgrade, restore after
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pre_upgrade_autorag_dsc_patch(
    pytestconfig: pytest.Config,
    dsc_resource: DataScienceCluster,
) -> DataScienceCluster:
    """Enable AI Pipelines and OGX in DSC before upgrade tests.

    Uses ResourceEditor.update() (non-reverting) so the components stay
    Managed through the upgrade boundary. No-op during post-upgrade.
    """
    if not pytestconfig.option.pre_upgrade:
        return dsc_resource

    components = dsc_resource.instance.spec.components
    patches: dict[str, Any] = {}

    for component_name in ("aipipelines", "ogx"):
        current_state = components.get(component_name, {}).get("managementState")
        if current_state != DscComponents.ManagementState.MANAGED:
            patches[component_name] = {"managementState": "Managed"}

    if patches:
        LOGGER.info("Setting DSC components to Managed", components=list(patches.keys()))
        editor = ResourceEditor(patches={dsc_resource: {"spec": {"components": patches}}})
        editor.update()
        wait_for_dsc_status_ready(dsc_resource=dsc_resource)

    return dsc_resource


@pytest.fixture(scope="session")
def post_upgrade_autorag_dsc_restore(
    pytestconfig: pytest.Config,
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    """Restore AI Pipelines and OGX to Removed state after all upgrade resources are cleaned up."""
    yield dsc_resource

    if not pytestconfig.option.post_upgrade:
        return

    components = dsc_resource.instance.spec.components
    patches: dict[str, Any] = {}

    for component_name in ("aipipelines", "ogx"):
        current_state = components.get(component_name, {}).get("managementState")
        if current_state == DscComponents.ManagementState.REMOVED:
            pytest.fail(
                f"{component_name} managementState is already 'Removed' during post-upgrade teardown. "
                "Expected 'Managed' — the pre-upgrade fixture should have set it. "
                "This may indicate the upgrade reverted the DSC configuration."
            )
        patches[component_name] = {"managementState": "Removed"}

    if patches:
        LOGGER.info("Restoring DSC components to Removed", components=list(patches.keys()))
        editor = ResourceEditor(patches={dsc_resource: {"spec": {"components": patches}}})
        editor.update()
        wait_for_dsc_status_ready(dsc_resource=dsc_resource)


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autorag_upgrade_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    pre_upgrade_autorag_dsc_patch: DataScienceCluster,
    post_upgrade_autorag_dsc_restore: DataScienceCluster,
) -> Generator[Namespace, Any, Any]:
    """Fixed-name namespace for AutoRAG upgrade tests."""
    pre = pytestconfig.option.pre_upgrade
    post = pytestconfig.option.post_upgrade
    should_cleanup = not pre or post

    if pre:
        ns = Namespace(client=admin_client, name=UPGRADE_NAMESPACE)
        if ns.exists:
            raise AssertionError(
                f"Namespace {UPGRADE_NAMESPACE} already exists. "
                "This indicates a previous test run did not clean up properly."
            )
        with create_ns(
            admin_client=admin_client,
            name=UPGRADE_NAMESPACE,
            teardown=should_cleanup,
        ) as ns:
            yield ns
    else:
        ns = Namespace(client=admin_client, name=UPGRADE_NAMESPACE)
        yield ns
        if should_cleanup:
            ns.clean_up()


# ---------------------------------------------------------------------------
# DSPA
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autorag_upgrade_dspa(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
) -> Generator[DataSciencePipelinesApplication, Any, Any]:
    """DataSciencePipelinesApplication — created pre-upgrade, referenced post-upgrade."""
    pre = pytestconfig.option.pre_upgrade
    post = pytestconfig.option.post_upgrade
    should_cleanup = not pre or post

    if pre:
        managed_pipelines_spec: dict[str, Any] = {}
        if MANAGED_PIPELINES_IMAGE:
            managed_pipelines_spec["image"] = MANAGED_PIPELINES_IMAGE

        with DataSciencePipelinesApplication(
            client=admin_client,
            name=DSPA_NAME,
            namespace=autorag_upgrade_namespace.name,
            dsp_version="v2",
            api_server={
                "enableSamplePipeline": False,
                "managedPipelines": managed_pipelines_spec,
            },
            object_storage={
                "disableHealthCheck": False,
                "enableExternalRoute": True,
                "minio": {
                    "deploy": True,
                    "image": DSPA_MINIO_IMAGE,
                },
            },
            teardown=should_cleanup,
        ) as dspa_resource:
            Deployment(
                client=admin_client,
                name=DSPA_PIPELINE_DEPLOYMENT,
                namespace=autorag_upgrade_namespace.name,
            ).wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
            yield dspa_resource
    else:
        dspa_resource = DataSciencePipelinesApplication(
            client=admin_client,
            name=DSPA_NAME,
            namespace=autorag_upgrade_namespace.name,
        )
        Deployment(
            client=admin_client,
            name=DSPA_PIPELINE_DEPLOYMENT,
            namespace=autorag_upgrade_namespace.name,
        ).wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
        yield dspa_resource
        if should_cleanup:
            dspa_resource.clean_up()


@pytest.fixture(scope="session")
def autorag_upgrade_dspa_route(
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_dspa: DataSciencePipelinesApplication,
) -> Route:
    return Route(
        client=admin_client,
        name=DSPA_PIPELINE_DEPLOYMENT,
        namespace=autorag_upgrade_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="session")
def autorag_upgrade_dspa_api_url(autorag_upgrade_dspa_route: Route) -> str:
    return f"https://{autorag_upgrade_dspa_route.host}"


@pytest.fixture(scope="session")
def autorag_upgrade_dspa_auth_headers(current_client_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {current_client_token}"}


@pytest.fixture(scope="session")
def autorag_upgrade_dspa_ca_bundle_file(admin_client: DynamicClient) -> str:
    return create_ca_bundle_file(client=admin_client)


@pytest.fixture(scope="session")
def autorag_upgrade_dspa_s3_credentials(
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_dspa: DataSciencePipelinesApplication,
) -> Secret:
    """Patch DSPA S3 secret with standard AWS credential fields."""
    secret = Secret(
        client=admin_client,
        name=DSPA_S3_SECRET,
        namespace=autorag_upgrade_namespace.name,
    )
    assert secret.exists, f"Secret '{DSPA_S3_SECRET}' not found in {autorag_upgrade_namespace.name}"

    access_key = base64.b64decode(secret.instance.data.get("accesskey", "")).decode()
    secret_key = base64.b64decode(secret.instance.data.get("secretkey", "")).decode()
    endpoint = f"http://minio-{DSPA_NAME}.{autorag_upgrade_namespace.name}.svc.cluster.local:9000"

    secret.update(
        resource_dict={
            "metadata": {"name": secret.name, "namespace": autorag_upgrade_namespace.name},
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


# ---------------------------------------------------------------------------
# OGX infrastructure: PostgreSQL, etcd, Milvus (persisted through upgrade)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autorag_upgrade_ogx_secret(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    pre = pytestconfig.option.pre_upgrade
    post = pytestconfig.option.post_upgrade
    should_cleanup = not pre or post
    secret_name = f"{AUTORAG_RESOURCE_PREFIX}-ogx-secret"

    if pre:
        with Secret(
            client=admin_client,
            namespace=autorag_upgrade_namespace.name,
            name=secret_name,
            type="Opaque",
            string_data=AUTORAG_OGX_SECRET_DATA,
            teardown=should_cleanup,
        ) as secret:
            yield secret
    else:
        secret = Secret(
            client=admin_client,
            name=secret_name,
            namespace=autorag_upgrade_namespace.name,
        )
        yield secret
        if should_cleanup:
            secret.clean_up()


@pytest.fixture(scope="session")
def autorag_upgrade_postgres_deployment(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_ogx_secret: Secret,
) -> Generator[Deployment, Any, Any]:
    pre = pytestconfig.option.pre_upgrade
    post = pytestconfig.option.post_upgrade
    should_cleanup = not pre or post
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-pg"
    deploy_name = f"{AUTORAG_RESOURCE_PREFIX}-pg"

    if pre:
        with Deployment(
            client=admin_client,
            namespace=autorag_upgrade_namespace.name,
            name=deploy_name,
            min_ready_seconds=5,
            replicas=1,
            selector={"matchLabels": {"app": app_label}},
            strategy={"type": "Recreate"},
            template=get_postgres_template(secret_name=autorag_upgrade_ogx_secret.name, app_label=app_label),
            teardown=should_cleanup,
        ) as deployment:
            deployment.wait_for_replicas(deployed=True, timeout=240)
            yield deployment
    else:
        deployment = Deployment(
            client=admin_client,
            name=deploy_name,
            namespace=autorag_upgrade_namespace.name,
        )
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment
        if should_cleanup:
            deployment.clean_up()


@pytest.fixture(scope="session")
def autorag_upgrade_postgres_service(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_postgres_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    pre = pytestconfig.option.pre_upgrade
    post = pytestconfig.option.post_upgrade
    should_cleanup = not pre or post
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-pg"
    svc_name = f"{AUTORAG_RESOURCE_PREFIX}-pg"

    if pre:
        with Service(
            client=admin_client,
            namespace=autorag_upgrade_namespace.name,
            name=svc_name,
            ports=[{"port": 5432, "targetPort": 5432}],
            selector={"app": app_label},
            wait_for_resource=True,
            teardown=should_cleanup,
        ) as service:
            yield service
    else:
        service = Service(
            client=admin_client,
            name=svc_name,
            namespace=autorag_upgrade_namespace.name,
        )
        yield service
        if should_cleanup:
            service.clean_up()


@pytest.fixture(scope="session")
def autorag_upgrade_etcd_deployment(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
) -> Generator[Deployment, Any, Any]:
    pre = pytestconfig.option.pre_upgrade
    post = pytestconfig.option.post_upgrade
    should_cleanup = not pre or post
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-etcd"
    deploy_name = f"{AUTORAG_RESOURCE_PREFIX}-etcd"

    if pre:
        template = get_etcd_template(etcd_service_name=deploy_name)
        template["metadata"]["labels"]["app"] = app_label
        with Deployment(
            client=admin_client,
            namespace=autorag_upgrade_namespace.name,
            name=deploy_name,
            replicas=1,
            selector={"matchLabels": {"app": app_label}},
            strategy={"type": "Recreate"},
            template=template,
            teardown=should_cleanup,
        ) as deployment:
            deployment.wait_for_replicas(deployed=True, timeout=120)
            yield deployment
    else:
        deployment = Deployment(
            client=admin_client,
            name=deploy_name,
            namespace=autorag_upgrade_namespace.name,
        )
        deployment.wait_for_replicas(deployed=True, timeout=120)
        yield deployment
        if should_cleanup:
            deployment.clean_up()


@pytest.fixture(scope="session")
def autorag_upgrade_etcd_service(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_etcd_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    pre = pytestconfig.option.pre_upgrade
    post = pytestconfig.option.post_upgrade
    should_cleanup = not pre or post
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-etcd"
    svc_name = f"{AUTORAG_RESOURCE_PREFIX}-etcd"

    if pre:
        with Service(
            client=admin_client,
            namespace=autorag_upgrade_namespace.name,
            name=svc_name,
            ports=[{"port": 2379, "targetPort": 2379}],
            selector={"app": app_label},
            wait_for_resource=True,
            teardown=should_cleanup,
        ) as service:
            yield service
    else:
        service = Service(
            client=admin_client,
            name=svc_name,
            namespace=autorag_upgrade_namespace.name,
        )
        yield service
        if should_cleanup:
            service.clean_up()


@pytest.fixture(scope="session")
def autorag_upgrade_milvus_deployment(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_etcd_deployment: Deployment,
    autorag_upgrade_etcd_service: Service,
) -> Generator[Deployment, Any, Any]:
    pre = pytestconfig.option.pre_upgrade
    post = pytestconfig.option.post_upgrade
    should_cleanup = not pre or post
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-milvus"
    deploy_name = f"{AUTORAG_RESOURCE_PREFIX}-milvus"
    etcd_service_name = f"{AUTORAG_RESOURCE_PREFIX}-etcd"

    if pre:
        template = get_milvus_template(etcd_service_name=etcd_service_name)
        template["metadata"]["labels"]["app"] = app_label
        with Deployment(
            client=admin_client,
            namespace=autorag_upgrade_namespace.name,
            name=deploy_name,
            min_ready_seconds=5,
            replicas=1,
            selector={"matchLabels": {"app": app_label}},
            strategy={"type": "Recreate"},
            template=template,
            teardown=should_cleanup,
        ) as deployment:
            deployment.wait_for_replicas(deployed=True, timeout=240)
            yield deployment
    else:
        deployment = Deployment(
            client=admin_client,
            name=deploy_name,
            namespace=autorag_upgrade_namespace.name,
        )
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment
        if should_cleanup:
            deployment.clean_up()


@pytest.fixture(scope="session")
def autorag_upgrade_milvus_service(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_milvus_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    pre = pytestconfig.option.pre_upgrade
    post = pytestconfig.option.post_upgrade
    should_cleanup = not pre or post
    app_label = f"{AUTORAG_RESOURCE_PREFIX}-milvus"
    svc_name = f"{AUTORAG_RESOURCE_PREFIX}-milvus"

    if pre:
        with Service(
            client=admin_client,
            namespace=autorag_upgrade_namespace.name,
            name=svc_name,
            ports=[{"name": "grpc", "port": 19530, "targetPort": 19530}],
            selector={"app": app_label},
            wait_for_resource=True,
            teardown=should_cleanup,
        ) as service:
            yield service
    else:
        service = Service(
            client=admin_client,
            name=svc_name,
            namespace=autorag_upgrade_namespace.name,
        )
        yield service
        if should_cleanup:
            service.clean_up()


# ---------------------------------------------------------------------------
# vLLM models — always created fresh (both pre and post upgrade)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autorag_upgrade_hf_token_secret(
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    existing = Secret(
        client=admin_client,
        namespace=autorag_upgrade_namespace.name,
        name="hf-token-secret",
    )
    if existing.exists:
        yield existing
    else:
        hf_token = os.environ.get("HF_TOKEN", "")
        with Secret(
            client=admin_client,
            namespace=autorag_upgrade_namespace.name,
            name="hf-token-secret",
            type="Opaque",
            string_data={"token": hf_token},
        ) as new_secret:
            yield new_secret


@pytest.fixture(scope="session")
def autorag_upgrade_model_service_account(
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_hf_token_secret: Secret,
) -> Generator[ServiceAccount, Any, Any]:
    with ServiceAccount(
        client=admin_client,
        namespace=autorag_upgrade_namespace.name,
        name=f"{AUTORAG_RESOURCE_PREFIX}-model-sa",
        secrets=[{"name": autorag_upgrade_hf_token_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="session")
def autorag_upgrade_inference_runtime(
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
) -> Generator[ServingRuntimeFromTemplate, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=f"{AUTORAG_RESOURCE_PREFIX}-vllm-inf",
        namespace=autorag_upgrade_namespace.name,
        template_name=RuntimeTemplates.VLLM_CPU_x86,
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as runtime:
        yield runtime


@pytest.fixture(scope="session")
def autorag_upgrade_inference_service(
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_inference_runtime: ServingRuntimeFromTemplate,
    autorag_upgrade_model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    served_model_name = AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID or AUTORAG_INFERENCE_MODEL_NAME
    with create_isvc(
        client=admin_client,
        name=f"{AUTORAG_RESOURCE_PREFIX}-inference",
        namespace=autorag_upgrade_namespace.name,
        model_format="vLLM",
        runtime=autorag_upgrade_inference_runtime.name,
        storage_uri=AUTORAG_INFERENCE_MODEL_URI,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        wait=True,
        timeout=1800,
        model_service_account=autorag_upgrade_model_service_account.name,
        resources={
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "4", "memory": "8Gi"},
        },
        model_env_variables=[{"name": "VLLM_CPU_KVCACHE_SPACE", "value": "2"}],
        argument=["--served-model-name", served_model_name, "--max-model-len", "4096"],
    ) as isvc:
        yield isvc


@pytest.fixture(scope="session")
def autorag_upgrade_inference_url(autorag_upgrade_inference_service: InferenceService) -> str:
    url = autorag_upgrade_inference_service.instance.status.address.url
    assert url, f"InferenceService {autorag_upgrade_inference_service.name} has no status.address.url"
    return f"{url}/v1"


@pytest.fixture(scope="session")
def autorag_upgrade_inference_route(
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_inference_service: InferenceService,
) -> Generator[Route, Any, Any]:
    route_name = generate_random_name(prefix=f"{AUTORAG_RESOURCE_PREFIX}-inf", length=12)
    with Route(
        client=admin_client,
        namespace=autorag_upgrade_namespace.name,
        name=route_name,
        service=f"{autorag_upgrade_inference_service.name}-predictor",
        wait_for_resource=True,
    ) as route:
        ResourceEditor(
            patches={
                route: {
                    "spec": {
                        "tls": {
                            "termination": "edge",
                            "insecureEdgeTerminationPolicy": "Redirect",
                        }
                    }
                }
            }
        ).update()
        route.wait(timeout=60)
        yield route


@pytest.fixture(scope="session")
def autorag_upgrade_embedding_runtime(
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
) -> Generator[ServingRuntimeFromTemplate, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=f"{AUTORAG_RESOURCE_PREFIX}-vllm-emb",
        namespace=autorag_upgrade_namespace.name,
        template_name=RuntimeTemplates.VLLM_CPU_x86,
        multi_model=False,
        enable_http=True,
        enable_grpc=False,
    ) as runtime:
        yield runtime


@pytest.fixture(scope="session")
def autorag_upgrade_embedding_service(
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_embedding_runtime: ServingRuntimeFromTemplate,
    autorag_upgrade_model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=f"{AUTORAG_RESOURCE_PREFIX}-embedding",
        namespace=autorag_upgrade_namespace.name,
        model_format="vLLM",
        runtime=autorag_upgrade_embedding_runtime.name,
        storage_uri=AUTORAG_EMBEDDING_MODEL_URI,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        wait=True,
        timeout=1800,
        model_service_account=autorag_upgrade_model_service_account.name,
        resources={
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "4", "memory": "8Gi"},
        },
        model_env_variables=[
            {"name": "VLLM_CPU_KVCACHE_SPACE", "value": "2"},
            {"name": "VLLM_ENGINE_ITERATION_TIMEOUT_S", "value": "600"},
            {"name": "VLLM_MAX_NUM_SEQS", "value": "2"},
            {"name": "VLLM_TARGET_DEVICE", "value": "cpu"},
        ],
        argument=[
            "--served-model-name",
            AUTORAG_EMBEDDING_MODEL_NAME,
            "--runner",
            "pooling",
            "--max-model-len",
            AUTORAG_EMBEDDING_MAX_MODEL_LEN,
        ],
    ) as isvc:
        yield isvc


@pytest.fixture(scope="session")
def autorag_upgrade_embedding_url(autorag_upgrade_embedding_service: InferenceService) -> str:
    url = autorag_upgrade_embedding_service.instance.status.address.url
    assert url, f"InferenceService {autorag_upgrade_embedding_service.name} has no status.address.url"
    return f"{url}/v1"


# ---------------------------------------------------------------------------
# OGX server (persisted through upgrade)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autorag_upgrade_ogx_server(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    pre_upgrade_autorag_dsc_patch: DataScienceCluster,
    autorag_upgrade_ogx_secret: Secret,
    autorag_upgrade_postgres_deployment: Deployment,
    autorag_upgrade_postgres_service: Service,
    autorag_upgrade_milvus_service: Service,
    autorag_upgrade_inference_url: str,
    autorag_upgrade_embedding_url: str,
    autorag_upgrade_inference_route: Route,
) -> Generator[OgxServer, Any, Any]:
    pre = pytestconfig.option.pre_upgrade
    post = pytestconfig.option.post_upgrade
    should_cleanup = not pre or post
    ogx_name = f"{AUTORAG_RESOURCE_PREFIX}-ogx"

    if pre:
        inference_catalog_model_id = AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID or AUTORAG_INFERENCE_MODEL_NAME

        wait_for_vllm_model_ready(
            vllm_base_url=f"https://{autorag_upgrade_inference_route.host}/v1",
            model_name=inference_catalog_model_id,
        )

        secret_name = autorag_upgrade_ogx_secret.name
        postgres_service_name = autorag_upgrade_postgres_service.name

        env_vars = [
            {"name": "INFERENCE_MODEL", "value": inference_catalog_model_id},
            {"name": "INFERENCE_PROVIDER_MODEL_ID", "value": inference_catalog_model_id},
            {
                "name": "VLLM_API_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "vllm-api-token"}},
            },
            {"name": "VLLM_URL", "value": autorag_upgrade_inference_url},
            {"name": "VLLM_TLS_VERIFY", "value": "false"},
            {"name": "VLLM_MAX_TOKENS", "value": "128"},
            {"name": "FMS_ORCHESTRATOR_URL", "value": "http://localhost"},
            {"name": "EMBEDDING_MODEL", "value": AUTORAG_EMBEDDING_MODEL_NAME},
            {"name": "EMBEDDING_PROVIDER_MODEL_ID", "value": AUTORAG_EMBEDDING_MODEL_NAME},
            {"name": "VLLM_EMBEDDING_URL", "value": autorag_upgrade_embedding_url},
            {
                "name": "VLLM_EMBEDDING_API_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "vllm-embedding-api-token"}},
            },
            {"name": "VLLM_EMBEDDING_MAX_TOKENS", "value": "768"},
            {"name": "VLLM_EMBEDDING_TLS_VERIFY", "value": "false"},
            {"name": "POSTGRES_HOST", "value": postgres_service_name},
            {"name": "POSTGRES_PORT", "value": "5432"},
            {
                "name": "POSTGRES_USER",
                "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "postgres-user"}},
            },
            {
                "name": "POSTGRES_PASSWORD",
                "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "postgres-password"}},
            },
            {"name": "POSTGRES_DB", "value": "ps_db"},
            {"name": "POSTGRES_TABLE_NAME", "value": "llamastack_kvstore"},
            {"name": "MILVUS_ENDPOINT", "value": f"http://{autorag_upgrade_milvus_service.name}:19530"},
            {
                "name": "MILVUS_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "milvus-token"}},
            },
            {"name": "MILVUS_CONSISTENCY_LEVEL", "value": "Bounded"},
        ]

        ogx_config: dict[str, Any] = {
            "distribution": {"name": "rh-dev"},
            "workload": {
                "resources": {
                    "requests": {"cpu": "1", "memory": "2Gi"},
                    "limits": {"cpu": "2", "memory": "4Gi"},
                },
                "overrides": {
                    "env": env_vars,
                },
            },
        }

        with create_ogx_server(
            client=admin_client,
            name=ogx_name,
            namespace=autorag_upgrade_namespace.name,
            config=ogx_config,
        ) as ogx_srv:
            ogx_srv.wait_for_status(status=OgxServer.Status.READY, timeout=900)
            yield ogx_srv
    else:
        ogx_srv = OgxServer(
            client=admin_client,
            name=ogx_name,
            namespace=autorag_upgrade_namespace.name,
        )
        ogx_srv.wait_for_status(status=OgxServer.Status.READY, timeout=900)
        yield ogx_srv
        if should_cleanup:
            ogx_srv.clean_up()


@pytest.fixture(scope="session")
def autorag_upgrade_ogx_deployment(
    admin_client: DynamicClient,
    autorag_upgrade_ogx_server: OgxServer,
) -> Deployment:
    deployment = Deployment(
        client=admin_client,
        namespace=autorag_upgrade_ogx_server.namespace,
        name=autorag_upgrade_ogx_server.name,
        min_ready_seconds=10,
    )
    deployment.timeout_seconds = 240
    deployment.wait(timeout=240)
    deployment.wait_for_replicas()
    return deployment


@pytest.fixture(scope="session")
def autorag_upgrade_ogx_route(
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_ogx_deployment: Deployment,
) -> Generator[Route, Any, Any]:
    route_name = generate_random_name(prefix=f"{AUTORAG_RESOURCE_PREFIX}-ogx", length=12)
    with Route(
        client=admin_client,
        namespace=autorag_upgrade_namespace.name,
        name=route_name,
        service=f"{autorag_upgrade_ogx_deployment.name}-service",
        wait_for_resource=True,
    ) as route:
        ResourceEditor(
            patches={
                route: {
                    "spec": {
                        "tls": {
                            "termination": "edge",
                            "insecureEdgeTerminationPolicy": "Redirect",
                        }
                    },
                    "metadata": {
                        "annotations": {Annotations.HaproxyRouterOpenshiftIo.TIMEOUT: "10m"},
                    },
                }
            }
        ).update()
        route.wait(timeout=60)
        yield route


@pytest.fixture(scope="session")
def autorag_upgrade_ogx_url(autorag_upgrade_ogx_route: Route) -> str:
    return f"https://{autorag_upgrade_ogx_route.host}"


@pytest.fixture(scope="session")
def autorag_upgrade_ogx_client(
    autorag_upgrade_ogx_route: Route,
) -> Generator[OgxClient, Any, Any]:
    http_client = httpx.Client(verify=OGX_CLIENT_VERIFY_SSL, timeout=300)
    try:
        client = OgxClient(
            base_url=f"https://{autorag_upgrade_ogx_route.host}",
            max_retries=3,
            http_client=http_client,
            timeout=300,
        )
        wait_for_ogx_client_ready(client=client)
        yield client
    finally:
        http_client.close()


@pytest.fixture(scope="session")
def autorag_upgrade_discovered_models(
    autorag_upgrade_ogx_client: OgxClient,
) -> tuple[str, str]:
    """Discover embedding and generation model IDs from OGX."""
    registered_ids = log_registered_models(client=autorag_upgrade_ogx_client)

    embedding_id = resolve_model_id(registered_ids=registered_ids, model_name=AUTORAG_EMBEDDING_MODEL_NAME)
    assert embedding_id is not None, (
        f"Embedding model '{AUTORAG_EMBEDDING_MODEL_NAME}' not registered in OGX server. "
        f"Available: {sorted(registered_ids)}"
    )

    generation_id = resolve_model_id(registered_ids=registered_ids, model_name=AUTORAG_INFERENCE_MODEL_NAME)
    if generation_id is None and AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID:
        generation_id = resolve_model_id(
            registered_ids=registered_ids, model_name=AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID
        )
    assert generation_id is not None, (
        f"Generation model not registered in OGX server. "
        f"Looked for '{AUTORAG_INFERENCE_MODEL_NAME}'"
        + (f" and '{AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID}'" if AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID else "")
        + f". Available: {sorted(registered_ids)}"
    )

    LOGGER.info("Using models", embedding=embedding_id, generation=generation_id)
    return embedding_id, generation_id


# ---------------------------------------------------------------------------
# Pipeline setup and execution
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autorag_upgrade_managed_pipeline(
    autorag_upgrade_dspa: DataSciencePipelinesApplication,
    autorag_upgrade_dspa_api_url: str,
    autorag_upgrade_dspa_auth_headers: dict[str, str],
    autorag_upgrade_dspa_ca_bundle_file: str,
) -> dict[str, str] | None:
    """Discover managed AutoRAG pipeline. None in legacy YAML mode."""
    if not use_managed_pipelines(yaml_env_value=AUTORAG_PIPELINE_YAML):
        return None
    return wait_for_managed_pipeline(
        api_url=autorag_upgrade_dspa_api_url,
        headers=autorag_upgrade_dspa_auth_headers,
        display_name=MANAGED_PIPELINE_AUTORAG,
        ca_bundle=autorag_upgrade_dspa_ca_bundle_file,
        timeout=DSPA_READY_BUFFER_SECONDS + MANAGED_PIPELINE_WAIT_TIMEOUT,
        poll_interval=MANAGED_PIPELINE_POLL_INTERVAL,
    )


@pytest.fixture(scope="session")
def autorag_upgrade_test_data(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_dspa_s3_credentials: Secret,
) -> None:
    """Upload AutoRAG test data to DSPA MinIO. No-op when pre-upgrade is not set."""
    if not pytestconfig.option.pre_upgrade:
        return

    src_bucket = shlex.quote(s=AUTORAG_S3_BUCKET)
    src_input_key = shlex.quote(s=AUTORAG_INPUT_DATA_KEY)
    src_test_key = shlex.quote(s=AUTORAG_TEST_DATA_KEY)
    dst_bucket = shlex.quote(s=DSPA_S3_BUCKET)

    minio_endpoint = f"http://minio-{DSPA_NAME}.{autorag_upgrade_namespace.name}.svc.cluster.local:9000"
    src_endpoint = os.environ.get("AWS_S3_ENDPOINT", "https://s3.amazonaws.com")
    src_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    src_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

    mc_setup = (
        "export MC_CONFIG_DIR=/work/.mc && "
        "mc alias set src $SRC_ENDPOINT $SRC_ACCESS_KEY $SRC_SECRET_KEY && "
        "mc alias set dst $DST_ENDPOINT $DST_ACCESS_KEY $DST_SECRET_KEY"
    )
    mc_copy = (
        f"mc cp --recursive src/{src_bucket}/{src_input_key} /work/input_data/ && "
        f"mc cp src/{src_bucket}/{src_test_key} /work/benchmark_data.json && "
        f"mc mb --ignore-existing dst/{dst_bucket} && "
        f"mc cp --recursive /work/input_data/ dst/{dst_bucket}/{src_input_key}/ && "
        f"mc cp /work/benchmark_data.json dst/{dst_bucket}/{src_test_key}"
    )

    pod_name = f"autorag-upgrade-uploader-{uuid.uuid4().hex[:8]}"
    with Pod(
        client=admin_client,
        name=pod_name,
        namespace=autorag_upgrade_namespace.name,
        restart_policy="Never",
        volumes=[{"name": "work", "emptyDir": {}}],
        containers=[
            {
                "name": "minio-uploader",
                "image": MINIO_MC_IMAGE,
                "command": ["/bin/sh", "-c"],
                "args": [f"{mc_setup} && {mc_copy}"],
                "volumeMounts": [{"name": "work", "mountPath": "/work"}],
                "securityContext": MINIO_UPLOADER_SECURITY_CONTEXT,
                "env": [
                    {"name": "SRC_ENDPOINT", "value": src_endpoint},
                    {"name": "SRC_ACCESS_KEY", "value": src_access_key},
                    {"name": "SRC_SECRET_KEY", "value": src_secret_key},
                    {"name": "DST_ENDPOINT", "value": minio_endpoint},
                    {
                        "name": "DST_ACCESS_KEY",
                        "valueFrom": {"secretKeyRef": {"name": DSPA_S3_SECRET, "key": "accesskey"}},
                    },
                    {
                        "name": "DST_SECRET_KEY",
                        "valueFrom": {"secretKeyRef": {"name": DSPA_S3_SECRET, "key": "secretkey"}},
                    },
                ],
            }
        ],
        wait_for_resource=True,
    ) as upload_pod:
        try:
            upload_pod.wait_for_status(status="Succeeded", timeout=300)
        except TimeoutExpiredError:
            try:
                LOGGER.error("Data upload pod logs", logs=upload_pod.log())
            except Exception:  # noqa: BLE001
                LOGGER.warning("Could not fetch upload pod logs")
            raise

    LOGGER.info("AutoRAG test data uploaded to MinIO")


@pytest.fixture(scope="session")
def autorag_upgrade_ogx_url_secret(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_ogx_url: str,
) -> Generator[Secret, Any, Any]:
    pre = pytestconfig.option.pre_upgrade
    post = pytestconfig.option.post_upgrade
    should_cleanup = not pre or post
    secret_name = f"{AUTORAG_RESOURCE_PREFIX}-ogx-url"  # pragma: allowlist secret

    if pre:
        with Secret(
            client=admin_client,
            name=secret_name,
            namespace=autorag_upgrade_namespace.name,
            string_data={
                "OGX_CLIENT_BASE_URL": autorag_upgrade_ogx_url,
                "OGX_CLIENT_API_KEY": "unused",  # pragma: allowlist secret
            },
            teardown=should_cleanup,
        ) as secret:
            yield secret
    else:
        secret = Secret(
            client=admin_client,
            name=secret_name,
            namespace=autorag_upgrade_namespace.name,
        )
        yield secret
        if should_cleanup:
            secret.clean_up()


@pytest.fixture(scope="session")
def autorag_upgrade_run_id(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_dspa_api_url: str,
    autorag_upgrade_dspa_auth_headers: dict[str, str],
    autorag_upgrade_dspa_ca_bundle_file: str,
    autorag_upgrade_managed_pipeline: dict[str, str] | None,
    autorag_upgrade_ogx_url_secret: Secret,
    autorag_upgrade_discovered_models: tuple[str, str],
    autorag_upgrade_inference_url: str,
    autorag_upgrade_embedding_url: str,
    autorag_upgrade_dspa_s3_credentials: Secret,
    autorag_upgrade_test_data: None,
) -> str:
    """Pipeline run ID — created when pre-upgrade is set, loaded from ConfigMap otherwise."""
    if not pytestconfig.option.pre_upgrade:
        baselines = load_baseline_from_configmap(
            client=admin_client,
            namespace=autorag_upgrade_namespace.name,
            configmap_name=UPGRADE_BASELINE_CONFIGMAP,
        )
        return baselines["run_id"]

    embedding_model, generation_model = autorag_upgrade_discovered_models

    parameters: dict[str, Any] = {
        "input_data_secret_name": autorag_upgrade_dspa_s3_credentials.name,
        "input_data_bucket_name": DSPA_S3_BUCKET,
        "input_data_key": AUTORAG_INPUT_DATA_KEY,
        "test_data_secret_name": autorag_upgrade_dspa_s3_credentials.name,
        "test_data_bucket_name": DSPA_S3_BUCKET,
        "test_data_key": AUTORAG_TEST_DATA_KEY,
        "ogx_secret_name": autorag_upgrade_ogx_url_secret.name,
        "optimization_max_rag_patterns": AUTORAG_MAX_RAG_PATTERNS,
        "optimization_metric": AUTORAG_OPTIMIZATION_METRIC,
        "embedding_models": [embedding_model],
        "generation_models": [generation_model],
        "vector_io_provider_id": "milvus-remote",
    }

    if autorag_upgrade_managed_pipeline is not None:
        run_id = create_pipeline_run_managed(
            api_url=autorag_upgrade_dspa_api_url,
            headers=autorag_upgrade_dspa_auth_headers,
            pipeline_id=autorag_upgrade_managed_pipeline["pipeline_id"],
            pipeline_version_id=autorag_upgrade_managed_pipeline["pipeline_version_id"],
            run_name=UPGRADE_RUN_DISPLAY_NAME,
            parameters=parameters,
            ca_bundle=autorag_upgrade_dspa_ca_bundle_file,
        )
    else:
        pipeline_yaml_path = resolve_pipeline_yaml(value=AUTORAG_PIPELINE_YAML)
        pipeline_id = upload_pipeline(
            api_url=autorag_upgrade_dspa_api_url,
            headers=autorag_upgrade_dspa_auth_headers,
            pipeline_yaml_path=pipeline_yaml_path,
            pipeline_name=f"autorag-upgrade-{autorag_upgrade_namespace.name}",
            ca_bundle=autorag_upgrade_dspa_ca_bundle_file,
        )
        run_id = create_pipeline_run(
            api_url=autorag_upgrade_dspa_api_url,
            headers=autorag_upgrade_dspa_auth_headers,
            pipeline_id=pipeline_id,
            run_name=UPGRADE_RUN_DISPLAY_NAME,
            parameters=parameters,
            ca_bundle=autorag_upgrade_dspa_ca_bundle_file,
        )

    return run_id


# ---------------------------------------------------------------------------
# Baseline capture / load
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def autorag_capture_upgrade_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
    autorag_upgrade_run_id: str,
    autorag_upgrade_managed_pipeline: dict[str, str] | None,
    autorag_upgrade_ogx_client: OgxClient,
) -> None:
    """Capture baseline after pre-upgrade experiment completes. No-op when pre-upgrade is not set."""
    if not pytestconfig.option.pre_upgrade:
        return

    vector_store_ids = discover_vector_store_ids(ogx_client=autorag_upgrade_ogx_client)

    baselines: dict[str, Any] = {
        "run_id": autorag_upgrade_run_id,
        "run_display_name": UPGRADE_RUN_DISPLAY_NAME,
        "vector_store_ids": vector_store_ids,
    }
    if autorag_upgrade_managed_pipeline is not None:
        baselines["pipeline_id"] = autorag_upgrade_managed_pipeline["pipeline_id"]
        baselines["pipeline_version_id"] = autorag_upgrade_managed_pipeline["pipeline_version_id"]

    save_baseline_to_configmap(
        client=admin_client,
        namespace=autorag_upgrade_namespace.name,
        baselines=baselines,
        configmap_name=UPGRADE_BASELINE_CONFIGMAP,
    )


@pytest.fixture(scope="session")
def autorag_upgrade_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    autorag_upgrade_namespace: Namespace,
) -> dict:
    """Load pre-upgrade baseline. Returns empty dict when post-upgrade is not set."""
    if not pytestconfig.option.post_upgrade:
        return {}

    return load_baseline_from_configmap(
        client=admin_client,
        namespace=autorag_upgrade_namespace.name,
        configmap_name=UPGRADE_BASELINE_CONFIGMAP,
    )
