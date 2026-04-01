import logging
from collections import namedtuple
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import Any

import pytest
import structlog
import yaml
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount

from tests.model_serving.model_server.llmd.llmd_configs import TinyLlamaOciConfig
from tests.model_serving.model_server.llmd.utils import wait_for_llmisvc, wait_for_llmisvc_pods_ready
from utilities.constants import Timeout
from utilities.infra import create_inference_token, s3_endpoint_secret, update_configmap_data
from utilities.llmd_utils import create_llmd_gateway
from utilities.logger import RedactedString

LOGGER = structlog.get_logger(name=__name__)
logging.getLogger("timeout_sampler").setLevel(logging.WARNING)

AuthEntry = namedtuple(typename="AuthEntry", field_names=["service", "token"])


# ===========================================
#  Gateway
# ===========================================
@pytest.fixture(scope="session", autouse=True)
def shared_llmd_gateway(admin_client: DynamicClient) -> Generator[Gateway]:
    """Shared LLMD gateway for all tests."""
    with create_llmd_gateway(
        client=admin_client,
        timeout=Timeout.TIMEOUT_1MIN,
    ) as gateway:
        yield gateway


# ===========================================
#  Storage — S3 secret + service account
# ===========================================
@pytest.fixture(scope="class")
def s3_service_account(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[str]:
    """Create S3 secret + service account. Resolved automatically for S3 configs."""
    with ExitStack() as stack:
        secret = stack.enter_context(
            cm=s3_endpoint_secret(
                client=admin_client,
                name="llmd-s3-secret",
                namespace=unprivileged_model_namespace.name,
                aws_access_key=request.getfixturevalue(argname="aws_access_key_id"),
                aws_secret_access_key=request.getfixturevalue(argname="aws_secret_access_key"),
                aws_s3_region=request.getfixturevalue(argname="models_s3_bucket_region"),
                aws_s3_bucket=request.getfixturevalue(argname="models_s3_bucket_name"),
                aws_s3_endpoint=request.getfixturevalue(argname="models_s3_bucket_endpoint"),
            )
        )
        sa = stack.enter_context(
            cm=ServiceAccount(
                client=admin_client,
                namespace=unprivileged_model_namespace.name,
                name="llmd-s3-service-account",
                secrets=[{"name": secret.name}],
            )
        )
        yield sa.name


# ===========================================
#  GPU guards
# ===========================================
@pytest.fixture(scope="session")
def skip_if_less_than_2_gpus(gpu_count_on_cluster: int) -> None:
    """Skip test if fewer than 2 GPUs are available on the cluster."""
    if gpu_count_on_cluster < 2:
        pytest.skip(f"Test requires at least 2 GPUs (found {gpu_count_on_cluster})")


# ===========================================
#  LLMInferenceService creation
# ===========================================
@pytest.fixture(scope="class")
def llmisvc(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[LLMInferenceService]:
    """LLMInferenceService fixture driven by a config class.

    Usage:
        NAMESPACE = ns_from_file(__file__)

        @pytest.mark.parametrize(
            "unprivileged_model_namespace, llmisvc",
            [({"name": NAMESPACE}, SomeConfig)],
            indirect=True,
        )
    """
    config_cls = request.param
    namespace = unprivileged_model_namespace.name

    service_account = None
    if config_cls.storage_uri.startswith("s3://"):
        service_account = request.getfixturevalue(argname="s3_service_account")

    with _create_llmisvc_from_config(
        config_cls=config_cls, namespace=namespace, client=admin_client, service_account=service_account
    ) as svc:
        yield svc


@pytest.fixture(scope="class")
def llmisvc_auth_pair(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[tuple[AuthEntry, AuthEntry]]:
    """Two auth-enabled LLMISVCs with independent tokens for cross-auth testing."""
    namespace = unprivileged_model_namespace.name
    with ExitStack() as stack:
        entries = []
        for i in range(2):
            cfg = TinyLlamaOciConfig.with_overrides(
                name=f"llmisvc-auth-{i}",
                enable_auth=True,
            )
            svc = stack.enter_context(
                cm=_create_llmisvc_from_config(
                    config_cls=cfg,
                    namespace=namespace,
                    client=admin_client,
                )
            )
            token = stack.enter_context(
                cm=_create_auth_resources(
                    client=admin_client,
                    namespace=namespace,
                    svc=svc,
                    sa_name=f"auth-sa-{i}",
                )
            )
            entries.append(AuthEntry(service=svc, token=token))
        yield tuple(entries)


# ===========================================
#  Auth — SA + RBAC + token
# ===========================================
@pytest.fixture(scope="class")
def llmisvc_token(
    admin_client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> Generator[str]:
    """Create a dedicated SA with RBAC and return an auth token for the llmisvc."""
    with _create_auth_resources(
        client=admin_client,
        namespace=llmisvc.namespace,
        svc=llmisvc,
        sa_name=f"{llmisvc.name}-auth-sa",
    ) as token:
        yield token


# ===========================================
#  Monitoring
# ===========================================
@pytest.fixture(scope="session", autouse=True)
def llmd_user_workload_monitoring_config_map(
    admin_client: DynamicClient, cluster_monitoring_config: ConfigMap
) -> Generator[ConfigMap]:
    """Ephemeral user workload monitoring for LLMD tests."""
    data = {
        "config.yaml": yaml.dump({
            "prometheus": {
                "logLevel": "debug",
                "retention": "15d",
            }
        })
    }

    with update_configmap_data(
        client=admin_client,
        name="user-workload-monitoring-config",
        namespace="openshift-user-workload-monitoring",
        data=data,
    ) as cm:
        yield cm


# ===========================================
#  Helpers (not fixtures)
# ===========================================
@contextmanager
def _create_auth_resources(
    client: DynamicClient,
    namespace: str,
    svc: LLMInferenceService,
    sa_name: str,
) -> Generator[RedactedString, Any]:
    """Create SA + Role + RoleBinding and yield an auth token."""
    with (
        ServiceAccount(client=client, namespace=namespace, name=sa_name) as sa,
        Role(
            client=client,
            name=f"{svc.name}-view",
            namespace=namespace,
            rules=[
                {
                    "apiGroups": [svc.api_group],
                    "resources": ["llminferenceservices"],
                    "verbs": ["get"],
                    "resourceNames": [svc.name],
                }
            ],
        ) as role,
        RoleBinding(
            client=client,
            namespace=namespace,
            name=f"{sa_name}-view",
            role_ref_name=role.name,
            role_ref_kind=role.kind,
            subjects_kind="ServiceAccount",
            subjects_name=sa_name,
        ),
    ):
        yield RedactedString(value=create_inference_token(model_service_account=sa))


@contextmanager
def _create_llmisvc_from_config(
    config_cls: type,
    namespace: str,
    client: DynamicClient,
    service_account: str | None = None,
    teardown: bool = True,
) -> Generator[LLMInferenceService, Any]:
    """Create an LLMInferenceService from a config class."""
    LOGGER.info(f"\n{config_cls.describe(namespace=namespace)}")

    model: dict[str, Any] = {"uri": config_cls.storage_uri}
    if config_cls.model_name:
        model["name"] = config_cls.model_name

    main_container: dict[str, Any] = {"name": "main"}
    main_container.update({
        k: v
        for k, v in {
            "image": config_cls.container_image,
            "resources": config_cls.container_resources(),
            "env": config_cls.container_env(),
            "livenessProbe": config_cls.liveness_probe(),
            "readinessProbe": config_cls.readiness_probe(),
        }.items()
        if v
    })

    template: dict[str, Any] = {
        "configRef": config_cls.template_config_ref,
        "containers": [main_container],
    }
    if service_account:
        template["serviceAccountName"] = service_account

    prefill = config_cls.prefill_config()

    svc_kwargs: dict[str, Any] = {
        "client": client,
        "name": config_cls.name,
        "namespace": namespace,
        "annotations": config_cls.annotations(),
        "label": config_cls.labels(),
        "teardown": teardown,
        "model": model,
        "replicas": config_cls.replicas,
        "router": config_cls.router_config(),
        "template": template,
    }
    if prefill is not None:
        if service_account and "template" in prefill:
            prefill["template"]["serviceAccountName"] = service_account
        svc_kwargs["prefill"] = prefill

    with LLMInferenceService(**svc_kwargs) as llm_service:
        wait_for_llmisvc(llmisvc=llm_service, timeout=config_cls.wait_timeout)
        wait_for_llmisvc_pods_ready(client=client, llmisvc=llm_service)
        yield llm_service
