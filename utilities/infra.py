import json
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.secret import Secret

from tests.model_serving.model_server.utils import b64_encoded_string
from utilities.general import get_s3_secret_dict


@contextmanager
def create_ns(
    name: str,
    admin_client: DynamicClient,
    teardown: bool = True,
    delete_timeout: int = 4 * 60,
    labels: Optional[Dict[str, str]] = None,
) -> Generator[Namespace, None, None]:
    with Namespace(
        client=admin_client,
        name=name,
        label=labels,
        teardown=teardown,
        delete_timeout=delete_timeout,
    ) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=2 * 10)
        yield ns


def wait_for_kserve_predictor_deployment_replicas(client: DynamicClient, isvc: InferenceService) -> Deployment:
    ns = isvc.namespace

    deployments = list(
        Deployment.get(
            label_selector=f"{isvc.ApiGroup.SERVING_KSERVE_IO}/inferenceservice={isvc.name}",
            client=client,
            namespace=isvc.namespace,
        )
    )

    if len(deployments) == 1:
        deployment = deployments[0]
        if deployment.exists:
            deployment.wait_for_replicas()
            return deployment

    elif len(deployments) > 1:
        raise ResourceNotUniqueError(f"Multiple predictor deployments found in namespace {ns}")

    else:
        raise ResourceNotFoundError(f"Predictor deployment not found in namespace {ns}")


@contextmanager
def s3_endpoint_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
) -> Generator[Secret, None, None]:
    with Secret(
        client=admin_client,
        name=name,
        namespace=namespace,
        annotations={"opendatahub.io/connection-type": "s3"},
        data_dict=get_s3_secret_dict(
            aws_access_key=aws_access_key,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_bucket=aws_s3_bucket,
            aws_s3_endpoint=aws_s3_endpoint,
            aws_s3_region=aws_s3_region,
        ),
        wait_for_resource=True,
    ) as secret:
        yield secret


@contextmanager
def create_storage_config_secret(
    admin_client: DynamicClient,
    endpoint_secret_name: str,
    namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
    aws_s3_region: str,
    aws_s3_endpoint: str,
) -> Generator[Secret, None, None]:
    secret = {
        "access_key_id": aws_access_key,
        "bucket": aws_s3_bucket,
        "default_bucket": aws_s3_bucket,
        "endpoint_url": aws_s3_endpoint,
        "region": aws_s3_region,
        "secret_access_key": aws_secret_access_key,
        "type": "s3",
    }
    data = {endpoint_secret_name: b64_encoded_string(string_to_encode=json.dumps(secret))}
    with Secret(
        client=admin_client,
        namespace=namespace,
        data_dict=data,
        wait_for_resource=True,
        name="storage-config",
    ) as storage_config:
        yield storage_config


@contextmanager
def create_isvc_view_role(
    client: DynamicClient,
    isvc: InferenceService,
    name: str,
    resource_names: Optional[List[str]] = None,
) -> Role:
    rules = [
        {
            "apiGroups": [isvc.api_group],
            "resources": ["inferenceservices"],
            "verbs": ["get"],
        },
    ]

    if resource_names:
        rules[0].update({"resourceNames": resource_names})

    with Role(
        client=client,
        name=name,
        namespace=isvc.namespace,
        rules=rules,
    ) as role:
        yield role
