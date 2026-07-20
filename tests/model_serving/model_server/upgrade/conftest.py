from typing import Any, Generator

import pytest
from _pytest.nodes import Item
from _pytest.runner import CallInfo
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from simple_logger.logger import get_logger

from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelFormat,
    ModelStoragePath,
    ModelVersion,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import create_isvc
from utilities.infra import create_inference_token, create_isvc_view_role, create_ns, s3_endpoint_secret
from utilities.serving_runtime import ServingRuntimeFromTemplate


from tests.model_serving.model_runtime.vllm.utils import skip_if_not_deployment_mode
from tests.model_serving.model_server.upgrade.kserve_kueue_upgrade_config import (
    KSERVE_KUEUE_CLUSTER_QUEUE,
    KSERVE_KUEUE_CPU_QUOTA,
    KSERVE_KUEUE_ISVC_LABELS,
    KSERVE_KUEUE_ISVC_RESOURCES,
    KSERVE_KUEUE_LOCAL_QUEUE,
    KSERVE_KUEUE_MAX_REPLICAS,
    KSERVE_KUEUE_MEMORY_QUOTA,
    KSERVE_KUEUE_MIN_REPLICAS,
    KSERVE_KUEUE_RESOURCE_FLAVOR,
    KSERVE_KUEUE_UPGRADE_ISVC_NAME,
    KSERVE_KUEUE_UPGRADE_NAMESPACE,
    KSERVE_KUEUE_UPGRADE_RUNTIME_NAME,
    KSERVE_KUEUE_UPGRADE_S3_SECRET,
)
from tests.model_serving.model_server.upgrade.utils import (
    _capture_and_save_isvc_kueue_baseline,
    _create_kueue_upgrade_resources,
    _kserve_kueue_upgrade_runtime_template_kwargs,
)
from utilities.kueue_utils import LocalQueue


LOGGER = get_logger(name=__name__)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Item, call: CallInfo[None]) -> Generator[None, Any, Any]:
    """Track pre-upgrade test failures to prevent baseline capture on failure."""
    outcome = yield
    report = outcome.get_result()

    # Only track failures during the actual test execution (not setup/teardown)
    if call.when == "call" and report.failed:
        if "pre_upgrade" in item.keywords:
            # Mark that a pre-upgrade test failed so baseline capture is skipped
            item.config._pre_upgrade_test_failed = True  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def model_namespace_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    name = "upgrade-model-server"
    ns = Namespace(client=admin_client, name=name)

    if pytestconfig.option.post_upgrade:
        yield ns
        ns.clean_up()

    else:
        with create_ns(
            admin_client=admin_client,
            name=name,
            model_mesh_enabled=True,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def models_endpoint_s3_secret_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace_scope_session: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    secret_kwargs = {
        "client": admin_client,
        "name": "models-bucket-secret",
        "namespace": model_namespace_scope_session.name,
    }

    secret = Secret(**secret_kwargs)

    if pytestconfig.option.post_upgrade:
        yield secret
        secret.clean_up()

    else:
        with s3_endpoint_secret(
            **secret_kwargs,
            aws_access_key=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_region=models_s3_bucket_region,
            aws_s3_bucket=models_s3_bucket_name,
            aws_s3_endpoint=models_s3_bucket_endpoint,
            teardown=teardown_resources,
        ) as secret:
            yield secret


@pytest.fixture(scope="session")
def ci_endpoint_s3_secret_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace_scope_session: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    ci_s3_bucket_name: str,
    ci_s3_bucket_region: str,
    ci_s3_bucket_endpoint: str,
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    secret_kwargs = {
        "client": admin_client,
        "name": "ci-bucket-secret",
        "namespace": model_namespace_scope_session.name,
    }

    secret = Secret(**secret_kwargs)

    if pytestconfig.option.post_upgrade:
        yield secret
        secret.clean_up()

    else:
        with s3_endpoint_secret(
            **secret_kwargs,
            aws_access_key=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_region=ci_s3_bucket_region,
            aws_s3_bucket=ci_s3_bucket_name,
            aws_s3_endpoint=ci_s3_bucket_endpoint,
            teardown=teardown_resources,
        ) as secret:
            yield secret


@pytest.fixture(scope="session")
def model_mesh_model_service_account_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    ci_endpoint_s3_secret_scope_session: Secret,
    teardown_resources: bool,
) -> Generator[ServiceAccount, Any, Any]:
    sa_kwargs = {
        "client": admin_client,
        "name": "models-bucket-sa",
        "namespace": ci_endpoint_s3_secret_scope_session.namespace,
    }

    sa = ServiceAccount(**sa_kwargs)

    if pytestconfig.option.post_upgrade:
        yield sa
        sa.clean_up()

    else:
        with ServiceAccount(
            **sa_kwargs,
            secrets=[{"name": ci_endpoint_s3_secret_scope_session.name}],
            teardown=teardown_resources,
        ) as sa:
            yield sa


@pytest.fixture(scope="session")
def openvino_serverless_serving_runtime_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace_scope_session: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    runtime_kwargs = {
        "client": admin_client,
        "name": "onnx-serverless",
        "namespace": model_namespace_scope_session.name,
    }

    model_runtime = ServingRuntime(**runtime_kwargs)

    if pytestconfig.option.post_upgrade:
        yield model_runtime
        model_runtime.clean_up()

    else:
        with ServingRuntimeFromTemplate(
            **runtime_kwargs,
            template_name=RuntimeTemplates.OVMS_KSERVE,
            multi_model=False,
            resources={
                ModelFormat.OVMS: {
                    "requests": {"cpu": "1", "memory": "4Gi"},
                    "limits": {"cpu": "2", "memory": "8Gi"},
                }
            },
            model_format_name={ModelFormat.ONNX: ModelVersion.OPSET13},
            teardown=teardown_resources,
        ) as model_runtime:
            yield model_runtime


@pytest.fixture(scope="session")
def ovms_serverless_inference_service_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    openvino_serverless_serving_runtime_scope_session: ServingRuntime,
    ci_endpoint_s3_secret_scope_session: Secret,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": openvino_serverless_serving_runtime_scope_session.name,
        "namespace": openvino_serverless_serving_runtime_scope_session.namespace,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc
        isvc.clean_up()

    else:
        with create_isvc(
            runtime=openvino_serverless_serving_runtime_scope_session.name,
            storage_path="test-dir",
            storage_key=ci_endpoint_s3_secret_scope_session.name,
            model_format=ModelAndFormat.OPENVINO_IR,
            deployment_mode=KServeDeploymentType.SERVERLESS,
            model_version=ModelVersion.OPSET13,
            teardown=teardown_resources,
            **isvc_kwargs,
        ) as isvc:
            yield isvc


@pytest.fixture(scope="session")
def caikit_raw_serving_runtime_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace_scope_session: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    runtime_kwargs = {
        "client": admin_client,
        "name": "caikit-raw",
        "namespace": model_namespace_scope_session.name,
    }

    model_runtime = ServingRuntime(**runtime_kwargs)

    if pytestconfig.option.post_upgrade:
        yield model_runtime
        model_runtime.clean_up()

    else:
        with ServingRuntimeFromTemplate(
            **runtime_kwargs,
            template_name=RuntimeTemplates.CAIKIT_STANDALONE_SERVING,
            multi_model=False,
            enable_http=True,
            teardown=teardown_resources,
        ) as model_runtime:
            yield model_runtime


@pytest.fixture(scope="session")
def caikit_raw_inference_service_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    caikit_raw_serving_runtime_scope_session: ServingRuntime,
    models_endpoint_s3_secret_scope_session: Secret,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": caikit_raw_serving_runtime_scope_session.name,
        "namespace": caikit_raw_serving_runtime_scope_session.namespace,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc

        isvc.clean_up()

    else:
        with create_isvc(
            runtime=caikit_raw_serving_runtime_scope_session.name,
            model_format=caikit_raw_serving_runtime_scope_session.instance.spec.supportedModelFormats[0].name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            storage_key=models_endpoint_s3_secret_scope_session.name,
            storage_path=ModelStoragePath.EMBEDDING_MODEL,
            external_route=False,
            teardown=teardown_resources,
            **isvc_kwargs,
        ) as isvc:
            yield isvc


@pytest.fixture(scope="session")
def s3_ovms_model_mesh_serving_runtime_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    model_namespace_scope_session: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    runtime_kwargs = {
        "client": admin_client,
        "name": "ovms-model-mesh",
        "namespace": model_namespace_scope_session.name,
    }

    model_runtime = ServingRuntime(**runtime_kwargs)

    if pytestconfig.option.post_upgrade:
        yield model_runtime
        model_runtime.clean_up()

    else:
        with ServingRuntimeFromTemplate(
            **runtime_kwargs,
            template_name=RuntimeTemplates.OVMS_MODEL_MESH,
            multi_model=True,
            protocol=Protocols.REST.upper(),
            resources={
                ModelFormat.OVMS: {
                    "requests": {"cpu": "1", "memory": "4Gi"},
                    "limits": {"cpu": "2", "memory": "8Gi"},
                }
            },
            teardown=teardown_resources,
        ) as model_runtime:
            yield model_runtime


@pytest.fixture(scope="session")
def openvino_model_mesh_inference_service_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    s3_ovms_model_mesh_serving_runtime_scope_session: ServingRuntime,
    ci_endpoint_s3_secret_scope_session: Secret,
    model_mesh_model_service_account_scope_session: ServiceAccount,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": s3_ovms_model_mesh_serving_runtime_scope_session.name,
        "namespace": s3_ovms_model_mesh_serving_runtime_scope_session.namespace,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc
        isvc.clean_up()

    else:
        with create_isvc(
            runtime=s3_ovms_model_mesh_serving_runtime_scope_session.name,
            model_service_account=model_mesh_model_service_account_scope_session.name,
            storage_key=ci_endpoint_s3_secret_scope_session.name,
            storage_path=ModelStoragePath.OPENVINO_EXAMPLE_MODEL,
            model_format=ModelAndFormat.OPENVINO_IR,
            deployment_mode=KServeDeploymentType.MODEL_MESH,
            model_version=ModelVersion.OPSET1,
            teardown=teardown_resources,
            **isvc_kwargs,
        ) as isvc:
            yield isvc


@pytest.fixture(scope="session")
def model_service_account_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    ci_endpoint_s3_secret_scope_session: Secret,
    teardown_resources: bool,
) -> Generator[ServiceAccount, Any, Any]:
    sa_kwargs = {
        "client": admin_client,
        "name": "upgrade-models-bucket-sa",
        "namespace": ci_endpoint_s3_secret_scope_session.namespace,
    }

    sa = ServiceAccount(**sa_kwargs)

    if pytestconfig.option.post_upgrade:
        yield sa
        sa.clean_up()

    else:
        with ServiceAccount(
            client=admin_client,
            namespace=ci_endpoint_s3_secret_scope_session.namespace,
            name="upgrade-models-bucket-sa",
            secrets=[{"name": ci_endpoint_s3_secret_scope_session.name}],
            teardown=teardown_resources,
        ) as sa:
            yield sa


@pytest.fixture(scope="session")
def http_view_role_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    ovms_authenticated_serverless_inference_service_scope_session: InferenceService,
    teardown_resources: bool,
) -> Generator[Role, Any, Any]:
    role_kwargs = {
        "client": admin_client,
        "name": f"{ovms_authenticated_serverless_inference_service_scope_session.name}-view",
    }

    role = Role(
        **role_kwargs,
        namespace=ovms_authenticated_serverless_inference_service_scope_session.namespace,
    )

    if pytestconfig.option.post_upgrade:
        yield role
        role.clean_up()

    else:
        with create_isvc_view_role(
            **role_kwargs,
            isvc=ovms_authenticated_serverless_inference_service_scope_session,
            resource_names=[ovms_authenticated_serverless_inference_service_scope_session.name],
            teardown=teardown_resources,
        ) as role:
            yield role


@pytest.fixture(scope="session")
def http_role_binding_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    http_view_role_scope_session: Role,
    model_service_account_scope_session: ServiceAccount,
    ovms_authenticated_serverless_inference_service_scope_session: InferenceService,
    teardown_resources: bool,
) -> Generator[RoleBinding, Any, Any]:
    rb_kwargs = {
        "client": admin_client,
        "name": f"{model_service_account_scope_session.name}-view",
        "namespace": ovms_authenticated_serverless_inference_service_scope_session.namespace,
    }

    rb = RoleBinding(**rb_kwargs)

    if pytestconfig.option.post_upgrade:
        yield rb
        rb.clean_up()

    else:
        with RoleBinding(
            **rb_kwargs,
            role_ref_name=http_view_role_scope_session.name,
            role_ref_kind=http_view_role_scope_session.kind,
            subjects_kind=model_service_account_scope_session.kind,
            subjects_name=model_service_account_scope_session.name,
            teardown=teardown_resources,
        ) as rb:
            yield rb


@pytest.fixture(scope="session")
def http_inference_token_scope_session(
    model_service_account_scope_session: ServiceAccount, http_role_binding_scope_session: RoleBinding
) -> str:
    return create_inference_token(model_service_account=model_service_account_scope_session)


@pytest.fixture(scope="session")
def ovms_authenticated_serverless_inference_service_scope_session(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    openvino_serverless_serving_runtime_scope_session: ServingRuntime,
    ci_endpoint_s3_secret_scope_session: Secret,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": f"{openvino_serverless_serving_runtime_scope_session.name}-auth",
        "namespace": openvino_serverless_serving_runtime_scope_session.namespace,
    }

    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        yield isvc
        isvc.clean_up()

    else:
        with create_isvc(
            runtime=openvino_serverless_serving_runtime_scope_session.name,
            storage_path="test-dir",
            storage_key=ci_endpoint_s3_secret_scope_session.name,
            model_format=ModelAndFormat.OPENVINO_IR,
            deployment_mode=KServeDeploymentType.SERVERLESS,
            model_version=ModelVersion.OPSET13,
            enable_auth=True,
            teardown=teardown_resources,
            **isvc_kwargs,
        ) as isvc:
            yield isvc


@pytest.fixture(scope="session")
def kserve_kueue_upgrade_s3_secret(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kserve_kueue_upgrade_namespace: Namespace,
    valid_aws_config: tuple[str, str],
    teardown_resources: bool,
) -> Generator[Secret, Any, Any]:
    """S3 connection secret for the KServe Kueue upgrade ISVC."""
    _ = valid_aws_config
    secret_kwargs = {
        "client": admin_client,
        "name": KSERVE_KUEUE_UPGRADE_S3_SECRET,
        "namespace": kserve_kueue_upgrade_namespace.name,
    }
    secret = Secret(**secret_kwargs)

    if pytestconfig.option.post_upgrade:
        if not secret.exists:
            pytest.fail(
                f"[POST-UPGRADE] S3 secret '{secret.name}' not found in namespace "
                f"'{secret.namespace}'. Ensure pre-upgrade tests completed successfully."
            )
        yield secret
        if teardown_resources:
            secret.clean_up()
    elif secret.exists:
        pytest.fail(
            f"Unexpected existing S3 secret '{secret.name}' in namespace '{secret.namespace}'. "
            "Clear stale upgrade resources before pre-upgrade."
        )
    else:
        with s3_endpoint_secret(
            **secret_kwargs,
            aws_access_key=pytestconfig.option.aws_access_key_id,
            aws_secret_access_key=pytestconfig.option.aws_secret_access_key,
            aws_s3_region=pytestconfig.option.ci_s3_bucket_region,
            aws_s3_bucket=pytestconfig.option.ci_s3_bucket_name,
            aws_s3_endpoint=pytestconfig.option.ci_s3_bucket_endpoint,
            teardown=teardown_resources,
        ) as configured_secret:
            yield configured_secret


@pytest.fixture(scope="session")
def kserve_kueue_upgrade_model_storage() -> dict[str, str]:
    """Return model storage path and version for external S3."""
    return {
        "storage_path": ModelStoragePath.OPENVINO_EXAMPLE_MODEL,
        "model_version": ModelVersion.OPSET13,
    }


@pytest.fixture(scope="session")
def kserve_kueue_upgrade_serving_runtime(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kserve_kueue_upgrade_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[ServingRuntime, Any, Any]:
    """OVMS ServingRuntime for KServe Kueue upgrade tests."""
    runtime_kwargs = {
        "client": admin_client,
        "name": KSERVE_KUEUE_UPGRADE_RUNTIME_NAME,
        "namespace": kserve_kueue_upgrade_namespace.name,
    }
    model_runtime = ServingRuntime(**runtime_kwargs)

    if pytestconfig.option.post_upgrade:
        if not model_runtime.exists:
            pytest.fail(
                f"[POST-UPGRADE] ServingRuntime '{model_runtime.name}' not found in namespace "
                f"'{model_runtime.namespace}'. Ensure pre-upgrade KServe+Kueue tests completed successfully."
            )
        yield model_runtime
        if teardown_resources:
            model_runtime.clean_up()
    elif model_runtime.exists:
        pytest.fail(
            f"Unexpected existing ServingRuntime '{model_runtime.name}' in namespace "
            f"'{model_runtime.namespace}'. Clear stale upgrade resources before pre-upgrade."
        )
    else:
        with ServingRuntimeFromTemplate(
            **_kserve_kueue_upgrade_runtime_template_kwargs(
                runtime_kwargs=runtime_kwargs,
                teardown_resources=teardown_resources,
            ),
        ) as model_runtime:
            yield model_runtime


@pytest.fixture(scope="session")
def kserve_kueue_upgrade_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace for KServe raw ISVC + Kueue upgrade tests."""
    ns = Namespace(client=admin_client, name=KSERVE_KUEUE_UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        if not ns.exists:
            pytest.fail(
                f"[POST-UPGRADE] Namespace '{ns.name}' not found. "
                "Ensure pre-upgrade KServe+Kueue tests completed successfully."
            )
        yield ns
        if teardown_resources:
            ns.clean_up()
    else:
        if ns.exists:
            pytest.fail(f"Unexpected existing namespace '{ns.name}'. Clear stale upgrade resources before pre-upgrade.")
        else:
            with create_ns(
                admin_client=admin_client,
                name=KSERVE_KUEUE_UPGRADE_NAMESPACE,
                model_mesh_enabled=False,
                add_dashboard_label=True,
                add_kueue_label=True,
                teardown=teardown_resources,
            ) as ns:
                yield ns


@pytest.fixture(scope="session")
def kserve_upgrade_kueue_resources(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kserve_kueue_upgrade_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[LocalQueue, Any, Any]:
    """Kueue ResourceFlavor, ClusterQueue, and LocalQueue for KServe upgrade tests."""
    yield from _create_kueue_upgrade_resources(
        pytestconfig=pytestconfig,
        admin_client=admin_client,
        namespace=kserve_kueue_upgrade_namespace.name,
        local_queue_name=KSERVE_KUEUE_LOCAL_QUEUE,
        cluster_queue_name=KSERVE_KUEUE_CLUSTER_QUEUE,
        resource_flavor_name=KSERVE_KUEUE_RESOURCE_FLAVOR,
        cpu_quota=KSERVE_KUEUE_CPU_QUOTA,
        memory_quota=KSERVE_KUEUE_MEMORY_QUOTA,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def kserve_kueue_upgrade_inference_service(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kserve_kueue_upgrade_namespace: Namespace,
    kserve_kueue_upgrade_serving_runtime: ServingRuntime,
    kserve_kueue_upgrade_s3_secret: Secret,
    kserve_kueue_upgrade_model_storage: dict[str, str],
    kserve_upgrade_kueue_resources: LocalQueue,
    teardown_resources: bool,
) -> Generator[InferenceService, Any, Any]:
    """Raw-deployment InferenceService with Kueue queue label for upgrade tests."""
    isvc_kwargs = {
        "client": admin_client,
        "name": KSERVE_KUEUE_UPGRADE_ISVC_NAME,
        "namespace": kserve_kueue_upgrade_namespace.name,
    }
    isvc = InferenceService(**isvc_kwargs)

    if pytestconfig.option.post_upgrade:
        if not isvc.exists:
            pytest.fail(
                f"[POST-UPGRADE] InferenceService '{isvc.name}' not found in namespace "
                f"'{isvc.namespace}'. Ensure pre-upgrade KServe+Kueue tests completed successfully."
            )
        yield isvc
        if teardown_resources:
            isvc.clean_up()
    elif isvc.exists:
        pytest.fail(
            f"Unexpected existing InferenceService '{isvc.name}' in namespace '{isvc.namespace}'. "
            "Clear stale upgrade resources before pre-upgrade."
        )
    else:
        with create_isvc(
            runtime=kserve_kueue_upgrade_serving_runtime.name,
            model_format=ModelAndFormat.OPENVINO_IR,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            storage_key=kserve_kueue_upgrade_s3_secret.name,
            storage_path=kserve_kueue_upgrade_model_storage["storage_path"],
            model_version=kserve_kueue_upgrade_model_storage["model_version"],
            external_route=True,
            min_replicas=KSERVE_KUEUE_MIN_REPLICAS,
            max_replicas=KSERVE_KUEUE_MAX_REPLICAS,
            resources=KSERVE_KUEUE_ISVC_RESOURCES,
            labels=KSERVE_KUEUE_ISVC_LABELS,
            teardown=teardown_resources,
            **isvc_kwargs,
        ) as isvc:
            yield isvc
            # Only capture baseline if no pre-upgrade tests failed.
            # The baseline represents the scaled and gated state after all validations pass.
            if not getattr(pytestconfig, "_pre_upgrade_test_failed", False):
                _capture_and_save_isvc_kueue_baseline(
                    pytestconfig=pytestconfig,
                    admin_client=admin_client,
                    isvc=isvc,
                )
            else:
                LOGGER.warning(
                    "Skipping baseline capture: pre-upgrade test(s) failed. "
                    "Post-upgrade tests will not have a valid baseline for comparison."
                )


@pytest.fixture
def skip_if_not_raw_deployment(
    kserve_kueue_upgrade_inference_service: InferenceService,
) -> None:
    """Skip tests when the Kueue upgrade ISVC is not deployed in RawDeployment mode."""
    skip_if_not_deployment_mode(
        isvc=kserve_kueue_upgrade_inference_service,
        deployment_type=KServeDeploymentType.RAW_DEPLOYMENT,
    )
