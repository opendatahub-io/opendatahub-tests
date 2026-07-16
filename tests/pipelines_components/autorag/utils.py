import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import httpx
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.pod import Pod
from ogx_client import APIConnectionError, InternalServerError, OgxClient
from timeout_sampler import TimeoutSampler, retry

from tests.fixtures.vector_io import (  # noqa: NIT001
    MILVUS_TOKEN,
    get_etcd_deployment_template,
    get_milvus_deployment_template,
)
from utilities.exceptions import UnexpectedResourceCountError
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)

OGX_CLIENT_VERIFY_SSL: bool = os.getenv("OGX_CLIENT_VERIFY_SSL", "false").lower() == "true"
OGX_CORE_POD_FILTER: str = "app=ogx"
POSTGRES_IMAGE: str = os.getenv(
    "OGX_VECTOR_IO_POSTGRES_IMAGE",
    (
        "registry.redhat.io/rhel9/postgresql-15@sha256:"
        "90ec347a35ab8a5d530c8d09f5347b13cc71df04f3b994bfa8b1a409b1171d59"  # pragma: allowlist secret
    ),
)

AUTORAG_INFERENCE_MODEL_URI: str = os.environ.get("AUTORAG_INFERENCE_MODEL_URI", "")
AUTORAG_INFERENCE_MODEL_NAME: str = os.environ.get("AUTORAG_INFERENCE_MODEL_NAME", "")
AUTORAG_EMBEDDING_MODEL_URI: str = os.environ.get("AUTORAG_EMBEDDING_MODEL_URI", "")
AUTORAG_EMBEDDING_MODEL_NAME: str = os.environ.get("AUTORAG_EMBEDDING_MODEL_NAME", "")

AUTORAG_OGX_SECRET_DATA: dict[str, str] = {
    "postgres-user": os.getenv("OGX_VECTOR_IO_POSTGRESQL_USER", "ps_user"),
    "postgres-password": os.getenv("OGX_VECTOR_IO_POSTGRESQL_PASSWORD", "ps_password"),
    "vllm-api-token": os.getenv("OGX_CORE_VLLM_API_TOKEN", ""),
    "vllm-embedding-api-token": os.getenv("OGX_CORE_VLLM_EMBEDDING_API_TOKEN", "fake"),
    "milvus-token": MILVUS_TOKEN,
    "aws-access-key-id": os.getenv("AWS_ACCESS_KEY_ID", ""),
    "aws-secret-access-key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
}


@contextmanager
def create_ogx_server(
    client: DynamicClient,
    name: str,
    namespace: str,
    config: dict[str, Any],
) -> Generator[OgxServer, Any, Any]:
    network: dict[str, Any] = {
        "policy": {
            "ingress": [
                {
                    "from": [
                        {
                            "namespaceSelector": {
                                "matchLabels": {
                                    "kubernetes.io/metadata.name": "openshift-ingress",
                                },
                            },
                        },
                    ],
                    "ports": [{"protocol": "TCP", "port": 8321}],
                },
            ],
        },
    }
    with OgxServer(
        client=client,
        name=name,
        namespace=namespace,
        distribution=config["distribution"],
        workload=config.get("workload"),
        network=network,
        tls=config.get("tls"),
        wait_for_resource=True,
    ) as ogx_srv:
        yield ogx_srv


@retry(
    wait_timeout=240,
    sleep=5,
    exceptions_dict={ResourceNotFoundError: [], UnexpectedResourceCountError: []},
)
def wait_for_unique_ogx_pod(client: DynamicClient, namespace: str) -> Pod:
    pods = list(Pod.get(client=client, namespace=namespace, label_selector=OGX_CORE_POD_FILTER))
    if not pods:
        raise ResourceNotFoundError(f"No pods found with label selector {OGX_CORE_POD_FILTER} in namespace {namespace}")
    if len(pods) != 1:
        raise UnexpectedResourceCountError(
            f"Expected exactly 1 pod with label selector {OGX_CORE_POD_FILTER} "
            f"in namespace {namespace}, found {len(pods)}"
        )
    return pods[0]


@retry(wait_timeout=90, sleep=5)
def wait_for_ogx_client_ready(client: OgxClient) -> bool:
    try:
        client.inspect.health()
        version = client.inspect.version()
        models = client.models.list()
        vector_stores = client.vector_stores.list()
        files = client.files.list()
        LOGGER.info(
            f"OGX server is available! "
            f"(version:{version.version} "
            f"models:{len(models.data)} "
            f"vector_stores:{len(vector_stores.data)} "
            f"files:{len(files.data)})"
        )
    except (APIConnectionError, InternalServerError) as error:
        LOGGER.debug(f"OGX server not ready yet: {error}")
        return False
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"Unexpected error checking OGX readiness: {e}")
        return False
    else:
        return True


def get_postgres_template(secret_name: str, app_label: str) -> dict[str, Any]:
    return {
        "metadata": {"labels": {"app": app_label, "autorag-component": "postgres"}},
        "spec": {
            "containers": [
                {
                    "name": "postgres",
                    "image": POSTGRES_IMAGE,
                    "ports": [{"containerPort": 5432}],
                    "env": [
                        {"name": "POSTGRESQL_DATABASE", "value": "ps_db"},
                        {
                            "name": "POSTGRESQL_USER",
                            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "postgres-user"}},
                        },
                        {
                            "name": "POSTGRESQL_PASSWORD",
                            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "postgres-password"}},
                        },
                    ],
                    "volumeMounts": [{"name": "postgresdata", "mountPath": "/var/lib/pgsql/data"}],
                },
            ],
            "volumes": [{"name": "postgresdata", "emptyDir": {}}],
        },
    }


def get_etcd_template(etcd_service_name: str) -> dict[str, Any]:
    template = get_etcd_deployment_template()
    container = template["spec"]["containers"][0]
    container["command"] = [
        "etcd",
        f"--advertise-client-urls=http://{etcd_service_name}:2379",
        "--listen-client-urls=http://0.0.0.0:2379",
        "--data-dir=/etcd",
    ]
    return template


def get_milvus_template(etcd_service_name: str) -> dict[str, Any]:
    template = get_milvus_deployment_template()
    container = template["spec"]["containers"][0]
    for env in container["env"]:
        if env["name"] == "ETCD_ENDPOINTS":
            env["value"] = f"{etcd_service_name}:2379"
    return template


def resolve_model_id(registered_ids: set[str], model_name: str) -> str | None:
    if model_name in registered_ids:
        return model_name
    matches = [mid for mid in registered_ids if mid.endswith(f"/{model_name}")]
    return matches[0] if len(matches) == 1 else None


def log_registered_models(client: OgxClient) -> set[str]:
    models = client.models.list()
    registered_ids = {model.id for model in models.data}
    LOGGER.info(
        "OGX registered models",
        models=[
            {
                "id": model.id,
                "model_type": str(getattr(model, "model_type", "?")),
                "custom_metadata": getattr(model, "custom_metadata", {}),
            }
            for model in models.data
        ],
    )
    return registered_ids


def wait_for_vllm_model_ready(vllm_base_url: str, model_name: str, timeout: int = 300) -> None:
    LOGGER.info("Probing vLLM reachability from test runner", url=vllm_base_url, model=model_name)
    try:
        with httpx.Client(verify=False, timeout=5) as probe:
            probe.get(f"{vllm_base_url}/models")
    except Exception as exc:  # noqa: BLE001
        LOGGER.info(
            "vLLM URL not reachable from test runner (cluster-internal); skipping readiness wait",
            url=vllm_base_url,
            model=model_name,
            reason=str(exc),
        )
        return

    def _check_model() -> bool:
        with httpx.Client(verify=False, timeout=30) as http_client:
            resp = http_client.get(f"{vllm_base_url}/models")
            if resp.status_code == 200:
                model_ids = {model.get("id", "") for model in resp.json().get("data", [])}
                LOGGER.info("vLLM models", url=vllm_base_url, models=sorted(model_ids))
                if resolve_model_id(model_ids, model_name) is not None:
                    LOGGER.info("vLLM model is ready", model=model_name)
                    return True
            else:
                LOGGER.debug("vLLM /v1/models returned non-200", status=resp.status_code)
        return False

    LOGGER.info("vLLM URL is reachable; waiting for model", url=vllm_base_url, model=model_name)
    for ready in TimeoutSampler(
        wait_timeout=timeout,
        sleep=15,
        func=_check_model,
        exceptions_dict={httpx.HTTPError: [], ConnectionError: [], OSError: []},
    ):
        if ready:
            return

    raise TimeoutError(
        f"vLLM did not serve model '{model_name}' at '{vllm_base_url}' within {timeout}s. "
        f"Verify --served-model-name is set correctly in the ISVC spec."
    )
