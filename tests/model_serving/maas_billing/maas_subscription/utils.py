from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from timeout_sampler import TimeoutSampler

from utilities.constants import (
    MAAS_GATEWAY_NAME,
    MAAS_GATEWAY_NAMESPACE,
    ApiGroups,
)
from utilities.resources.auth import Auth

LOGGER = structlog.get_logger(name=__name__)
MAAS_SUBSCRIPTION_NAMESPACE = "models-as-a-service"
MAAS_DB_NAMESPACE = "redhat-ods-applications"
POSTGRES_DEPLOYMENT_NAME = "postgres"
POSTGRES_SERVICE_NAME = "postgres"
POSTGRES_CREDS_SECRET_NAME = "postgres-creds"  # pragma: allowlist secret
MAAS_DB_CONFIG_SECRET_NAME = "maas-db-config"  # pragma: allowlist secret
POSTGRES_IMAGE = "registry.redhat.io/rhel9/postgresql-15:latest"
POSTGRES_READY_LOG_TEXT = "accepting connections"


@contextmanager
def patch_llmisvc_with_maas_router_and_tiers(
    llm_service: LLMInferenceService,
    tiers: Sequence[str],
    enable_auth: bool = True,
) -> Generator[None]:
    """
    Patch an LLMInferenceService to use MaaS router (gateway refs + route {})
    and set MaaS tier annotation.

    This is intended for MaaS subscription tests where you want distinct
    tiered models (e.g. free vs premium)

    Examples:
      - tiers=[]              -> open model
      - tiers=["premium"]     -> premium-only
    """
    router_spec = {
        "gateway": {"refs": [{"name": MAAS_GATEWAY_NAME, "namespace": MAAS_GATEWAY_NAMESPACE}]},
        "route": {},
    }

    tiers_val = list(tiers)
    patch_body = {
        "metadata": {
            "annotations": {
                f"alpha.{ApiGroups.MAAS_IO}/tiers": json.dumps(tiers_val),
                "security.opendatahub.io/enable-auth": "true" if enable_auth else "false",
            }
        },
        "spec": {"router": router_spec},
    }

    with ResourceEditor(patches={llm_service: patch_body}):
        yield


def model_id_from_chat_completions_url(model_url: str) -> str:
    path = urlparse(model_url).path.strip("/")
    parts = path.split("/")

    if len(parts) >= 2 and parts[0] == "llm":
        model_id = parts[1]
        if model_id:
            return model_id

    raise AssertionError(f"Cannot extract model id from url: {model_url!r} (path={path!r})")


def chat_payload_for_url(model_url: str, *, prompt: str = "Hello", max_tokens: int = 8) -> dict:
    model_id = model_id_from_chat_completions_url(model_url=model_url)
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }


def poll_expected_status(
    request_session_http: requests.Session,
    model_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    expected_statuses: set[int],
    wait_timeout: int = 240,
    sleep: int = 5,
    request_timeout: int = 60,
) -> requests.Response:
    """
    Poll model endpoint until we see one of `expected_statuses` or timeout.

    Returns the response that matched expected status.
    """
    last_response: requests.Response | None = None
    observed_responses: list[tuple[int | None, str]] = []

    for response in TimeoutSampler(
        wait_timeout=wait_timeout,
        sleep=sleep,
        func=request_session_http.post,
        url=model_url,
        headers=headers,
        json=payload,
        timeout=request_timeout,
    ):
        last_response = response
        status_code = getattr(response, "status_code", None)
        response_text = (getattr(response, "text", "") or "")[:200]

        observed_responses.append((status_code, response_text))

        LOGGER.info(f"Polling model_url={model_url} status={status_code} expected={sorted(expected_statuses)}")

        if status_code in expected_statuses:
            return response

    pytest.fail(
        "Timed out waiting for expected HTTP status. "
        f"model_url={model_url}, "
        f"expected={sorted(expected_statuses)}, "
        f"last_status={getattr(last_response, 'status_code', None)}, "
        f"last_body={(getattr(last_response, 'text', '') or '')[:200]}, "
        f"seen_count={len(observed_responses)}"
    )


def create_maas_subscription(
    admin_client: DynamicClient,
    subscription_namespace: str,
    subscription_name: str,
    owner_group_name: str,
    model_name: str,
    model_namespace: str,
    tokens_per_minute: int,
    window: str = "1m",
    priority: int = 0,
    teardown: bool = True,
    wait_for_resource: bool = True,
) -> MaaSSubscription:

    return MaaSSubscription(
        client=admin_client,
        name=subscription_name,
        namespace=subscription_namespace,
        owner={
            "groups": [{"name": owner_group_name}],
        },
        model_refs=[
            {
                "name": model_name,
                "namespace": model_namespace,
                "tokenRateLimits": [{"limit": tokens_per_minute, "window": window}],
            }
        ],
        priority=priority,
        teardown=teardown,
        wait_for_resource=wait_for_resource,
    )


def wait_for_auth_ready(auth: Auth, baseline_time: str, timeout: int = 60) -> None:
    """Wait for Auth CR to reconcile after a patch."""
    for instance in TimeoutSampler(wait_timeout=timeout, sleep=2, func=lambda: auth.instance):
        auth_conditions = (instance.status or {}).get("conditions") or []
        ready_condition = next(
            (condition for condition in auth_conditions if condition.get("type") == "Ready"),
            None,
        )
        if (
            ready_condition
            and ready_condition.get("lastTransitionTime") != baseline_time
            and ready_condition.get("status") == "True"
        ):
            return


def get_maas_postgres_labels() -> dict[str, str]:
    return {
        "app": "postgres",
        "purpose": "poc",
    }


def get_maas_api_labels() -> dict[str, str]:
    return {
        "app": "maas-api",
        "purpose": "poc",
    }


def get_maas_postgres_secret_objects(
    client: DynamicClient,
    namespace: str,
    teardown_resources: bool,
    postgres_user: str,
    postgres_password: str,
    postgres_db: str,
) -> list[Secret]:
    return [
        Secret(
            client=client,
            name=POSTGRES_CREDS_SECRET_NAME,
            namespace=namespace,
            string_data={
                "POSTGRES_USER": postgres_user,
                "POSTGRES_PASSWORD": postgres_password,
                "POSTGRES_DB": postgres_db,
            },
            label=get_maas_postgres_labels(),
            type="Opaque",
            teardown=teardown_resources,
        )
    ]


def get_maas_db_config_secret_objects(
    client: DynamicClient,
    namespace: str,
    teardown_resources: bool,
    postgres_user: str,
    postgres_password: str,
    postgres_db: str,
) -> list[Secret]:
    db_connection_url = (
        f"postgresql://{postgres_user}:{postgres_password}@{POSTGRES_SERVICE_NAME}:5432/{postgres_db}?sslmode=disable"
    )

    return [
        Secret(
            client=client,
            name=MAAS_DB_CONFIG_SECRET_NAME,
            namespace=namespace,
            string_data={"DB_CONNECTION_URL": db_connection_url},
            label=get_maas_api_labels(),
            type="Opaque",
            teardown=teardown_resources,
        )
    ]


def get_maas_postgres_service_objects(
    client: DynamicClient,
    namespace: str,
    teardown_resources: bool,
) -> list[Service]:
    return [
        Service(
            client=client,
            name=POSTGRES_SERVICE_NAME,
            namespace=namespace,
            selector={"app": "postgres"},
            ports=[
                {
                    "name": "postgres",
                    "port": 5432,
                    "protocol": "TCP",
                    "targetPort": 5432,
                }
            ],
            label=get_maas_postgres_labels(),
            teardown=teardown_resources,
        )
    ]


def get_maas_postgres_deployment_template_dict() -> dict[str, Any]:
    return {
        "metadata": {
            "labels": get_maas_postgres_labels(),
        },
        "spec": {
            "containers": [
                {
                    "name": "postgres",
                    "image": POSTGRES_IMAGE,
                    "env": [
                        {
                            "name": "POSTGRESQL_USER",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": POSTGRES_CREDS_SECRET_NAME,
                                    "key": "POSTGRES_USER",
                                }
                            },
                        },
                        {
                            "name": "POSTGRESQL_PASSWORD",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": POSTGRES_CREDS_SECRET_NAME,
                                    "key": "POSTGRES_PASSWORD",
                                }
                            },
                        },
                        {
                            "name": "POSTGRESQL_DATABASE",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": POSTGRES_CREDS_SECRET_NAME,
                                    "key": "POSTGRES_DB",
                                }
                            },
                        },
                    ],
                    "ports": [{"containerPort": 5432}],
                    "volumeMounts": [{"name": "data", "mountPath": "/var/lib/pgsql/data"}],
                    "resources": {
                        "requests": {"memory": "256Mi", "cpu": "100m"},
                        "limits": {"memory": "512Mi", "cpu": "500m"},
                    },
                    "readinessProbe": {
                        "exec": {"command": ["/usr/libexec/check-container"]},
                        "initialDelaySeconds": 5,
                        "periodSeconds": 5,
                    },
                }
            ],
            "volumes": [{"name": "data", "emptyDir": {}}],
        },
    }


def get_maas_postgres_deployment_objects(
    client: DynamicClient,
    namespace: str,
    teardown_resources: bool,
) -> list[Deployment]:
    return [
        Deployment(
            client=client,
            name=POSTGRES_DEPLOYMENT_NAME,
            namespace=namespace,
            label=get_maas_postgres_labels(),
            replicas=1,
            selector={"matchLabels": {"app": "postgres"}},
            template=get_maas_postgres_deployment_template_dict(),
            wait_for_resource=True,
            teardown=teardown_resources,
        )
    ]


def get_maas_postgres_resources(
    client: DynamicClient,
    namespace: str,
    teardown_resources: bool,
    postgres_user: str,
    postgres_password: str,
    postgres_db: str,
) -> dict[Any, Any]:
    return {
        Secret: get_maas_postgres_secret_objects(
            client=client,
            namespace=namespace,
            teardown_resources=teardown_resources,
            postgres_user=postgres_user,
            postgres_password=postgres_password,
            postgres_db=postgres_db,
        )
        + get_maas_db_config_secret_objects(
            client=client,
            namespace=namespace,
            teardown_resources=teardown_resources,
            postgres_user=postgres_user,
            postgres_password=postgres_password,
            postgres_db=postgres_db,
        ),
        Service: get_maas_postgres_service_objects(
            client=client,
            namespace=namespace,
            teardown_resources=teardown_resources,
        ),
        Deployment: get_maas_postgres_deployment_objects(
            client=client,
            namespace=namespace,
            teardown_resources=teardown_resources,
        ),
    }


def wait_for_postgres_deployment_ready(
    admin_client: DynamicClient,
    namespace: str = MAAS_DB_NAMESPACE,
    timeout: int = 180,
) -> None:
    deployment = Deployment(
        client=admin_client,
        name=POSTGRES_DEPLOYMENT_NAME,
        namespace=namespace,
    )
    deployment.wait_for_condition(condition="Available", status="True", timeout=timeout)


def get_postgres_pod_in_namespace(
    admin_client: DynamicClient,
    namespace: str = MAAS_DB_NAMESPACE,
) -> Pod:
    postgres_pods = list(Pod.get(client=admin_client, namespace=namespace, label_selector="app=postgres"))
    assert postgres_pods, f"No PostgreSQL pod found in namespace {namespace}"
    return postgres_pods[0]


def wait_for_postgres_connection_log(
    admin_client: DynamicClient,
    namespace: str = MAAS_DB_NAMESPACE,
    timeout: int = 180,
    sleep: int = 5,
) -> None:
    for _ in TimeoutSampler(wait_timeout=timeout, sleep=sleep, func=lambda: True):
        postgres_pod = get_postgres_pod_in_namespace(admin_client=admin_client, namespace=namespace)
        pod_log = postgres_pod.log(container="postgres")
        if POSTGRES_READY_LOG_TEXT in pod_log:
            LOGGER.info(f"PostgreSQL pod is accepting connections in namespace {namespace}")
            return

    raise TimeoutError(f"PostgreSQL pod in namespace {namespace} did not report accepting connections")
