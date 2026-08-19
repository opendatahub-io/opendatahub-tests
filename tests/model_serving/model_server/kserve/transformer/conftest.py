import os
from collections.abc import Generator
from typing import Any
from urllib.parse import urlparse

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount

from tests.model_serving.image_constants import ModelServingImages
from utilities.constants import (
    Annotations,
    KServeDeploymentType,
    Labels,
)
from utilities.infra import (
    create_inference_token,
    create_isvc_view_role,
    wait_for_inference_deployment_replicas,
)
from utilities.logger import RedactedString
from utilities.serving_runtime import ServingRuntimeFromTemplate

DEFAULT_TRANSFORMER_IMAGE: str = ModelServingImages.TRANSFORMER_IMAGE


@pytest.fixture(scope="class")
def ci_service_account(
    unprivileged_client: DynamicClient, ci_endpoint_s3_secret: Secret
) -> Generator[ServiceAccount, Any, Any]:
    """ServiceAccount referencing the CI S3 bucket secret.

    Pairs with ``ci_endpoint_s3_secret`` so the storage-initializer can
    find AWS credentials via the KServe-annotated secret.
    """
    with ServiceAccount(
        client=unprivileged_client,
        namespace=ci_endpoint_s3_secret.namespace,
        name="ci-bucket-sa",
        secrets=[{"name": ci_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def transformer_auth_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ci_s3_bucket_name: str,
    ci_endpoint_s3_secret: Secret,
    ci_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    """InferenceService with a custom transformer and auth enabled.

    Deploys a KServe InferenceService in raw-deployment mode with
    ``security: true`` and a custom sentiment transformer container.
    The predictor runtime is created from an ODH template specified
    via ``request.param``.

    Expected ``request.param`` keys:
        template-name: ODH runtime template name (e.g. ``RuntimeTemplates.MLSERVER``).
        multi-model: Whether the runtime supports multi-model serving.
        model-dir: S3 key prefix inside the CI bucket for the model artifacts.
        name (optional): Model format name override; defaults to the first
            format advertised by the template.

    Args:
        request: Pytest request providing indirect parametrisation.
        unprivileged_client: OpenShift client scoped to an unprivileged user.
        unprivileged_model_namespace: Namespace where resources are created.
        ci_s3_bucket_name: CI S3 bucket name from env/CLI.
        ci_endpoint_s3_secret: Secret with S3 credentials and KServe annotations.
        ci_service_account: ServiceAccount referencing the CI S3 secret.
    Yields:
        InferenceService: The ready ISVC; torn down after the test class.
    """
    transformer_image = os.environ.get("IMAGE_TRANSFORMER_IMG_TAG", DEFAULT_TRANSFORMER_IMAGE)

    isvc_name = "sentiment-analysis"
    template_name = request.param["template-name"]
    multi_model = request.param.get("multi-model", False)
    storage_uri = f"s3://{ci_s3_bucket_name}/{request.param['model-dir']}/"

    with ServingRuntimeFromTemplate(
        client=unprivileged_client,
        name="transformer-runtime",
        namespace=unprivileged_model_namespace.name,
        template_name=template_name,
        multi_model=multi_model,
        containers={
            "kserve-container": {
                "image": "quay.io/opendatahub/mlserver:odh-v3.4",
                "env": [
                    {"name": "MLSERVER_MODEL_NAME", "value": isvc_name},
                    {"name": "MLSERVER_MODEL_IMPLEMENTATION", "value": "mlserver_onnx.OnnxModel"},
                    {"name": "MLSERVER_MODEL_URI", "value": "/mnt/models"},
                    {"name": "MLSERVER_HTTP_PORT", "value": "8080"},
                    {"name": "MLSERVER_MODELS_DIR", "value": "/mnt/models"},
                ],
                "startupProbe": {
                    "initialDelaySeconds": 1,
                    "periodSeconds": 1,
                    "failureThreshold": 1,
                    "exec": {"command": ["/bin/sh", "-c", "[ -n \"$(ls -A /mnt/models 2>/dev/null)\" ]"]},
                },
                "readinessProbe": {
                    "initialDelaySeconds": 5,
                    "periodSeconds": 5,
                    "failureThreshold": 3,
                    "timeoutSeconds": 5,
                    "httpGet": {"path": f"/v2/models/{isvc_name}/ready", "port": 8080},
                },
                "livenessProbe": {
                    "initialDelaySeconds": 20,
                    "periodSeconds": 10,
                    "failureThreshold": 6,
                    "timeoutSeconds": 5,
                    "httpGet": {"path": f"/v2/models/{isvc_name}/ready", "port": 8080},
                },
                "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "privileged": False,
                    "runAsNonRoot": True,
                },
            }
        },
        supported_model_formats=[
            {"name": "onnx", "version": "1"},
            {"name": "sklearn", "version": "0"},
            {"name": "sklearn", "version": "1"},
            {"name": "xgboost", "version": "1"},
            {"name": "xgboost", "version": "2"},
            {"name": "lightgbm", "version": "3"},
            {"name": "lightgbm", "version": "4"},
        ],
        enable_http=True,
        enable_grpc=False,
    ) as runtime:
        model_format = request.param.get("name", runtime.instance.spec.supportedModelFormats[0].name)

        with InferenceService(
            client=unprivileged_client,
            name=isvc_name,
            namespace=unprivileged_model_namespace.name,
            annotations={
                Annotations.KserveAuth.SECURITY: "true",
                Annotations.KserveIo.DEPLOYMENT_MODE: KServeDeploymentType.RAW_DEPLOYMENT,
            },
            label={
                Labels.Kserve.NETWORKING_KSERVE_IO: Labels.Kserve.EXPOSED,
            },
            predictor={
                "serviceAccountName": ci_service_account.name,
                "minReplicas": 1,
                "model": {
                    "modelFormat": {"name": model_format},
                    "runtime": runtime.name,
                    "storage": {
                        "key": ci_endpoint_s3_secret.name,
                        "path": urlparse(storage_uri).path,
                    },
                    "resources": {
                        "requests": {"cpu": "10m", "memory": "256Mi"},
                        "limits": {"cpu": "1", "memory": "2Gi"},
                    },
                },
            },
            transformer={
                "minReplicas": 1,
                "containers": [
                    {
                        "name": "kserve-container",
                        "image": transformer_image,
                        "imagePullPolicy": "IfNotPresent",
                        "args": [
                            f"--model-name={isvc_name}",
                            "--tokenizer_name=optimum/distilbert-base-uncased-finetuned-sst-2-english",
                            "--sentiment_labels=negative,positive",
                            "--max_length=128",
                            "--input_names=input_ids,attention_mask",
                            "--output_name=predict",
                            "--include_star_rating",
                        ],
                        "resources": {
                            "requests": {"cpu": "500m", "memory": "512Mi"},
                            "limits": {"cpu": "1000m", "memory": "2Gi"},
                        },
                    }
                ],
            },
        ) as isvc:
            isvc.wait_for_condition(
                condition=isvc.Condition.READY,
                status=isvc.Condition.Status.TRUE,
                timeout=600,
            )
            wait_for_inference_deployment_replicas(
                client=unprivileged_client,
                isvc=isvc,
                expected_num_deployments=2,
            )
            yield isvc


@pytest.fixture(scope="class")
def transformer_auth_service_account(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[ServiceAccount, Any, Any]:
    """ServiceAccount used to mint auth token for inference requests."""
    with ServiceAccount(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name="transformer-auth-sa",
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def transformer_view_role(
    unprivileged_client: DynamicClient,
    transformer_auth_inference_service: InferenceService,
) -> Generator[Role, Any, Any]:
    """RBAC Role granting view access to the transformer ISVC.

    Args:
        unprivileged_client: OpenShift client scoped to an unprivileged user.
        transformer_auth_inference_service: The auth-enabled ISVC whose name
            is added to the role's ``resourceNames``.

    Yields:
        Role: The created view role; torn down after the test class.
    """
    with create_isvc_view_role(
        client=unprivileged_client,
        isvc=transformer_auth_inference_service,
        name=f"{transformer_auth_inference_service.name}-view",
        resource_names=[transformer_auth_inference_service.name],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def transformer_role_binding(
    unprivileged_client: DynamicClient,
    transformer_view_role: Role,
    transformer_auth_service_account: ServiceAccount,
) -> Generator[RoleBinding, Any, Any]:
    """RoleBinding that binds the view role to the model ServiceAccount.

    Args:
        unprivileged_client: OpenShift client scoped to an unprivileged user.
        transformer_view_role: The ISVC view role to bind.
        transformer_auth_service_account: ServiceAccount that receives the role binding.

    Yields:
        RoleBinding: The created binding; torn down after the test class.
    """
    with RoleBinding(
        client=unprivileged_client,
        namespace=transformer_auth_service_account.namespace,
        name=f"transformer-{transformer_auth_service_account.name}-view",
        role_ref_name=transformer_view_role.name,
        role_ref_kind=transformer_view_role.kind,
        subjects_kind=transformer_auth_service_account.kind,
        subjects_name=transformer_auth_service_account.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def transformer_inference_token(
    transformer_auth_service_account: ServiceAccount,
    transformer_role_binding: RoleBinding,
) -> str:
    """Bearer token for authenticating inference requests to the transformer ISVC.

    Depends on ``transformer_role_binding`` to ensure the ServiceAccount has
    view access before a token is minted.

    Args:
        transformer_auth_service_account: ServiceAccount from which the token is created.
        transformer_role_binding: Ensures the RBAC binding exists before
            token creation (not used directly).

    Returns:
        str: A ``RedactedString`` wrapping the bearer token so it is masked
            in logs.
    """
    return RedactedString(
        value=create_inference_token(model_service_account=transformer_auth_service_account)
    )
