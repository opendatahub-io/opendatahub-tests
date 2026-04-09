import json
from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.serving_runtime import ServingRuntime
from pytest import Config, FixtureRequest

from tests.model_explainability.lm_eval.constants import (
    ACCELERATOR_IDENTIFIER,
    ARC_EASY_DATASET_IMAGE,
    FLAN_T5_IMAGE,
    LMEVAL_OCI_REPO,
    LMEVAL_OCI_TAG,
)
from tests.model_explainability.lm_eval.utils import get_lmevaljob_pod
from utilities.constants import ApiGroups, KServeDeploymentType, Labels, MinIo, Protocols, RuntimeTemplates, Timeout
from utilities.exceptions import MissingParameter
from utilities.general import b64_encoded_string
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate

VLLM_EMULATOR: str = "vllm-emulator"
VLLM_EMULATOR_PORT: int = 8000
LMEVALJOB_NAME: str = "lmeval-test-job"


@pytest.fixture(scope="function")
def lmevaljob_hf(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    patched_dsc_lmeval_allow_all: DataScienceCluster,
    lmeval_hf_access_token: Secret,
) -> Generator[LMEvalJob]:
    with LMEvalJob(
        client=admin_client,
        name=LMEVALJOB_NAME,
        namespace=model_namespace.name,
        model="hf",
        model_args=[{"name": "pretrained", "value": "rgeada/tiny-untrained-granite"}],
        task_list=request.param.get("task_list"),
        log_samples=True,
        allow_online=True,
        allow_code_execution=True,
        system_instruction="Be concise. At every point give the shortest acceptable answer.",
        chat_template={
            "enabled": True,
        },
        limit="0.01",
        pod={
            "container": {
                "resources": {
                    "limits": {"cpu": "1", "memory": "8Gi"},
                    "requests": {"cpu": "1", "memory": "8Gi"},
                },
                "env": [
                    {
                        "name": "HF_TOKEN",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "hf-secret",
                                "key": "HF_ACCESS_TOKEN",
                            },
                        },
                    },
                    {"name": "HF_ALLOW_CODE_EVAL", "value": "1"},
                ],
            },
        },
    ) as job:
        yield job


@pytest.fixture(scope="function")
def lmevaljob_local_offline(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    patched_dsc_lmeval_allow_all: DataScienceCluster,
    lmeval_data_downloader_pod: Pod,
) -> Generator[LMEvalJob, Any, Any]:
    with LMEvalJob(
        client=admin_client,
        name=LMEVALJOB_NAME,
        namespace=model_namespace.name,
        model="hf",
        model_args=[{"name": "pretrained", "value": "/opt/app-root/src/hf_home/flan"}],
        task_list=request.param.get("task_list"),
        limit="0.01",
        log_samples=True,
        offline={"storage": {"pvcName": "lmeval-data"}},
        pod={
            "container": {
                "env": [
                    {"name": "HF_HUB_VERBOSITY", "value": "debug"},
                    {"name": "UNITXT_DEFAULT_VERBOSITY", "value": "debug"},
                ]
            }
        },
        label={Labels.OpenDataHub.DASHBOARD: "true", "lmevaltests": "vllm"},
    ) as job:
        yield job


@pytest.fixture(scope="class")
def oci_credentials_secret(
    admin_client: DynamicClient,
    oci_registry_host: str,
    model_namespace: Namespace,
) -> Generator[Secret, Any, Any]:
    """Create OCI registry data connection for async upload job"""

    # Create anonymous dockerconfig for OCI registry (no authentication)
    dockerconfig = {
        "auths": {
            f"{oci_registry_host}": {
                "auth": "",
                "email": "user@example.com",
            }
        }
    }

    data_dict = {
        ".dockerconfigjson": b64_encoded_string(json.dumps(dockerconfig)),
        "ACCESS_TYPE": b64_encoded_string(json.dumps(["Push", "Pull"])),
        "OCI_HOST": b64_encoded_string(oci_registry_host),
    }

    with Secret(
        client=admin_client,
        name="my-oci-credentials",
        namespace=model_namespace.name,
        data_dict=data_dict,
        label={
            Labels.OpenDataHub.DASHBOARD: "true",
            Labels.OpenDataHubIo.MANAGED: "true",
        },
        annotations={
            f"{ApiGroups.OPENDATAHUB_IO}/connection-type-ref": "oci-v1",
            "openshift.io/display-name": "My OCI Credentials",
        },
        type="kubernetes.io/dockerconfigjson",
    ) as secret:
        yield secret


@pytest.fixture(scope="function")
def lmevaljob_local_offline_oci(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    patched_dsc_lmeval_allow_all: DataScienceCluster,
    oci_credentials_secret: Secret,
    oci_registry_pod_with_minio: Pod,
    lmeval_data_downloader_pod: Pod,
) -> Generator[LMEvalJob, Any, Any]:
    with LMEvalJob(
        client=admin_client,
        name=LMEVALJOB_NAME,
        namespace=model_namespace.name,
        model="hf",
        model_args=[{"name": "pretrained", "value": "/opt/app-root/src/hf_home/flan"}],
        task_list=request.param.get("task_list"),
        limit="0.01",
        log_samples=True,
        offline={"storage": {"pvcName": "lmeval-data"}},
        pod={
            "container": {
                "env": [
                    {"name": "HF_HUB_VERBOSITY", "value": "debug"},
                    {"name": "UNITXT_DEFAULT_VERBOSITY", "value": "debug"},
                ]
            }
        },
        label={Labels.OpenDataHub.DASHBOARD: "true", "lmevaltests": "vllm"},
        outputs={
            "pvcManaged": {"size": "5Gi"},
            "oci": {
                "registry": {"name": oci_credentials_secret.name, "key": "OCI_HOST"},
                "repository": LMEVAL_OCI_REPO,
                "tag": LMEVAL_OCI_TAG,
                "dockerConfigJson": {"name": oci_credentials_secret.name, "key": ".dockerconfigjson"},
                "verifySSL": False,
            },
        },
    ) as job:
        yield job


@pytest.fixture(scope="function")
def lmevaljob_vllm_emulator(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    patched_dsc_lmeval_allow_all: DataScienceCluster,
    vllm_emulator_deployment: Deployment,
    vllm_emulator_service: Service,
    vllm_emulator_route: Route,
) -> Generator[LMEvalJob, Any, Any]:
    with LMEvalJob(
        client=admin_client,
        namespace=model_namespace.name,
        name=LMEVALJOB_NAME,
        model="local-completions",
        task_list={"taskNames": ["arc_easy"]},
        log_samples=True,
        batch_size="1",
        allow_online=True,
        allow_code_execution=False,
        outputs={"pvcManaged": {"size": "5Gi"}},
        model_args=[
            {"name": "model", "value": "emulatedModel"},
            {
                "name": "base_url",
                "value": f"http://{vllm_emulator_service.name}:{VLLM_EMULATOR_PORT!s}/v1/completions",
            },
            {"name": "num_concurrent", "value": "1"},
            {"name": "max_retries", "value": "3"},
            {"name": "tokenized_requests", "value": "False"},
            {"name": "tokenizer", "value": "ibm-granite/granite-guardian-3.1-8b"},
        ],
    ) as job:
        yield job


@pytest.fixture(scope="function")
def lmeval_data_pvc(
    admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[PersistentVolumeClaim, Any, Any]:
    with PersistentVolumeClaim(
        client=admin_client,
        name="lmeval-data",
        namespace=model_namespace.name,
        label={"lmevaltests": "vllm"},
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        size="20Gi",
    ) as pvc:
        yield pvc


@pytest.fixture(scope="function")
def lmeval_data_downloader_pod(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lmeval_data_pvc: PersistentVolumeClaim,
) -> Generator[Pod, Any, Any]:
    with Pod(
        client=admin_client,
        namespace=model_namespace.name,
        name="lmeval-downloader",
        label={"lmevaltests": "vllm"},
        security_context={"fsGroup": 1000, "seccompProfile": {"type": "RuntimeDefault"}},
        init_containers=[
            {
                "name": "flan-data-copy-to-pvc",
                "image": FLAN_T5_IMAGE,
                "command": ["/bin/sh", "-c", "cp --verbose -r /mnt/data/flan /mnt/pvc/flan"],
                "securityContext": {
                    "runAsUser": 1000,
                    "runAsNonRoot": True,
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": [{"mountPath": "/mnt/pvc", "name": "pvc-volume"}],
            }
        ],
        containers=[
            {
                "name": "dataset-copy-to-pvc",
                "image": request.param.get("dataset_image"),
                "command": [
                    "/bin/sh",
                    "-c",
                    "cp --verbose -r /mnt/data/datasets /mnt/pvc/datasets && chmod -R g+w /mnt/pvc/datasets",
                ],
                "securityContext": {
                    "runAsUser": 1000,
                    "runAsNonRoot": True,
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": [{"mountPath": "/mnt/pvc", "name": "pvc-volume"}],
            },
        ],
        restart_policy="Never",
        volumes=[{"name": "pvc-volume", "persistentVolumeClaim": {"claimName": "lmeval-data"}}],
    ) as pod:
        pod.wait_for_status(status=Pod.Status.SUCCEEDED, timeout=Timeout.TIMEOUT_20MIN)
        yield pod


@pytest.fixture(scope="function")
def vllm_emulator_deployment(
    admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[Deployment, Any, Any]:
    label = {Labels.Openshift.APP: VLLM_EMULATOR}
    with Deployment(
        client=admin_client,
        namespace=model_namespace.name,
        name=VLLM_EMULATOR,
        label=label,
        selector={"matchLabels": label},
        template={
            "metadata": {
                "labels": {
                    Labels.Openshift.APP: VLLM_EMULATOR,
                    "maistra.io/expose-route": "true",
                },
                "name": VLLM_EMULATOR,
            },
            "spec": {
                "containers": [
                    {
                        "image": "quay.io/trustyai_testing/vllm_emulator"
                        "@sha256:c4bdd5bb93171dee5b4c8454f36d7c42b58b2a4ceb74f29dba5760ac53b5c12d",
                        "name": "vllm-emulator",
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "capabilities": {"drop": ["ALL"]},
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                    }
                ]
            },
        },
        replicas=1,
    ) as deployment:
        yield deployment


@pytest.fixture(scope="function")
def vllm_emulator_service(
    admin_client: DynamicClient, model_namespace: Namespace, vllm_emulator_deployment: Deployment
) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        namespace=vllm_emulator_deployment.namespace,
        name=f"{VLLM_EMULATOR}-service",
        ports=[
            {
                "name": f"{VLLM_EMULATOR}-endpoint",
                "port": VLLM_EMULATOR_PORT,
                "protocol": Protocols.TCP,
                "targetPort": VLLM_EMULATOR_PORT,
            }
        ],
        selector={Labels.Openshift.APP: VLLM_EMULATOR},
    ) as service:
        yield service


@pytest.fixture(scope="function")
def vllm_emulator_route(
    admin_client: DynamicClient, model_namespace: Namespace, vllm_emulator_service: Service
) -> Generator[Route, Any, Any]:
    with Route(
        client=admin_client,
        namespace=vllm_emulator_service.namespace,
        name=VLLM_EMULATOR,
        service=vllm_emulator_service.name,
    ) as route:
        yield route


@pytest.fixture(scope="function")
def lmeval_minio_deployment(
    admin_client: DynamicClient, minio_namespace: Namespace, pvc_minio_namespace: PersistentVolumeClaim
) -> Generator[Deployment, Any, Any]:
    minio_app_label = {"app": MinIo.Metadata.NAME}
    # TODO: Unify with minio_llm_deployment fixture once datasets and models are in new model image
    with Deployment(
        client=admin_client,
        name=MinIo.Metadata.NAME,
        namespace=minio_namespace.name,
        replicas=1,
        selector={"matchLabels": minio_app_label},
        template={
            "metadata": {"labels": minio_app_label},
            "spec": {
                "volumes": [
                    {"name": "minio-storage", "persistentVolumeClaim": {"claimName": pvc_minio_namespace.name}}
                ],
                "containers": [
                    {
                        "name": MinIo.Metadata.NAME,
                        "image": "quay.io/minio/minio"
                        "@sha256:46b3009bf7041eefbd90bd0d2b38c6ddc24d20a35d609551a1802c558c1c958f",
                        "args": ["server", "/data", "--console-address", ":9001"],
                        "env": [
                            {"name": "MINIO_ROOT_USER", "value": MinIo.Credentials.ACCESS_KEY_VALUE},
                            {"name": "MINIO_ROOT_PASSWORD", "value": MinIo.Credentials.SECRET_KEY_VALUE},
                        ],
                        "ports": [{"containerPort": MinIo.Metadata.DEFAULT_PORT}, {"containerPort": 9001}],
                        "volumeMounts": [{"name": "minio-storage", "mountPath": "/data"}],
                    }
                ],
            },
        },
        label=minio_app_label,
        wait_for_resource=True,
    ) as deployment:
        deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_20MIN)
        yield deployment


@pytest.fixture(scope="function")
def lmeval_minio_copy_pod(
    admin_client: DynamicClient, minio_namespace: Namespace, lmeval_minio_deployment: Deployment, minio_service: Service
) -> Generator[Pod, Any, Any]:
    with Pod(
        client=admin_client,
        name="copy-to-minio",
        namespace=minio_namespace.name,
        restart_policy="Never",
        volumes=[{"name": "shared-data", "emptyDir": {}}],
        init_containers=[
            {
                "name": "copy-dataset-data",
                "image": ARC_EASY_DATASET_IMAGE,
                "command": ["/bin/sh", "-c"],
                "args": ["cp --verbose -r /mnt/data/datasets /shared/datasets"],
                "volumeMounts": [{"name": "shared-data", "mountPath": "/shared"}],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
            },
            {
                "name": "copy-flan-model-data",
                "image": FLAN_T5_IMAGE,
                "command": ["/bin/sh", "-c"],
                "args": ["cp --verbose -r /mnt/data/flan /shared/flan"],
                "volumeMounts": [{"name": "shared-data", "mountPath": "/shared"}],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
            },
        ],
        containers=[
            {
                "name": "minio-uploader",
                "image": "quay.io/minio/mc@sha256:470f5546b596e16c7816b9c3fa7a78ce4076bb73c2c73f7faeec0c8043923123",
                "command": ["/bin/sh", "-c"],
                "args": [
                    f"export MC_CONFIG_DIR=/shared/.mc && "
                    f"mc alias set myminio http://{minio_service.name}:{MinIo.Metadata.DEFAULT_PORT} "
                    f"{MinIo.Credentials.ACCESS_KEY_VALUE} {MinIo.Credentials.SECRET_KEY_VALUE} && "
                    "mc mb --ignore-existing myminio/models && "
                    "mc cp --recursive /shared/datasets/ myminio/models/datasets/ && "
                    "mc cp --recursive /shared/flan/ myminio/models/flan/"
                ],
                "volumeMounts": [{"name": "shared-data", "mountPath": "/shared"}],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
            }
        ],
        wait_for_resource=True,
    ) as pod:
        pod.wait_for_status(status=Pod.Status.SUCCEEDED, timeout=600)
        yield pod


@pytest.fixture(scope="function")
def lmevaljob_s3_offline(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lmeval_minio_deployment: Deployment,
    minio_service: Service,
    lmeval_minio_copy_pod: Pod,
    minio_data_connection: Secret,
) -> Generator[LMEvalJob, Any, Any]:
    with LMEvalJob(
        client=admin_client,
        name="evaljob-sample",
        namespace=model_namespace.name,
        model="hf",
        model_args=[{"name": "pretrained", "value": "/opt/app-root/src/hf_home/flan"}],
        task_list={"taskNames": ["arc_easy"]},
        log_samples=True,
        allow_online=False,
        offline={
            "storage": {
                "s3": {
                    "accessKeyId": {"name": minio_data_connection.name, "key": "AWS_ACCESS_KEY_ID"},
                    "secretAccessKey": {"name": minio_data_connection.name, "key": "AWS_SECRET_ACCESS_KEY"},
                    "bucket": {"name": minio_data_connection.name, "key": "AWS_S3_BUCKET"},
                    "endpoint": {"name": minio_data_connection.name, "key": "AWS_S3_ENDPOINT"},
                    "region": {"name": minio_data_connection.name, "key": "AWS_DEFAULT_REGION"},
                    "path": "",
                    "verifySSL": False,
                }
            }
        },
    ) as job:
        yield job


@pytest.fixture(scope="function")
def lmevaljob_hf_pod(admin_client: DynamicClient, lmevaljob_hf: LMEvalJob) -> Generator[Pod, Any, Any]:
    yield get_lmevaljob_pod(client=admin_client, lmevaljob=lmevaljob_hf)


@pytest.fixture(scope="function")
def lmevaljob_local_offline_pod(
    admin_client: DynamicClient, lmevaljob_local_offline: LMEvalJob
) -> Generator[Pod, Any, Any]:
    yield get_lmevaljob_pod(client=admin_client, lmevaljob=lmevaljob_local_offline)


@pytest.fixture(scope="function")
def lmevaljob_local_offline_pod_oci(
    admin_client: DynamicClient, lmevaljob_local_offline_oci: LMEvalJob
) -> Generator[Pod, Any, Any]:
    yield get_lmevaljob_pod(client=admin_client, lmevaljob=lmevaljob_local_offline_oci)


@pytest.fixture(scope="function")
def lmevaljob_vllm_emulator_pod(
    admin_client: DynamicClient, lmevaljob_vllm_emulator: LMEvalJob
) -> Generator[Pod, Any, Any]:
    yield get_lmevaljob_pod(client=admin_client, lmevaljob=lmevaljob_vllm_emulator)


@pytest.fixture(scope="function")
def lmevaljob_s3_offline_pod(admin_client: DynamicClient, lmevaljob_s3_offline: LMEvalJob) -> Generator[Pod, Any, Any]:
    yield get_lmevaljob_pod(client=admin_client, lmevaljob=lmevaljob_s3_offline)


@pytest.fixture(scope="function")
def lmevaljob_gpu_pod(admin_client: DynamicClient, lmevaljob_gpu: LMEvalJob) -> Generator[Pod, Any, Any]:
    yield get_lmevaljob_pod(client=admin_client, lmevaljob=lmevaljob_gpu)


@pytest.fixture(scope="function")
def lmeval_hf_access_token(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    pytestconfig: Config,
) -> Secret:
    hf_access_token = pytestconfig.option.hf_access_token
    if not hf_access_token:
        raise MissingParameter(
            "HF access token is not set. "
            "Either pass with `--hf-access-token` or set `HF_ACCESS_TOKEN` environment variable"
        )
    with Secret(
        client=admin_client,
        name="hf-secret",
        namespace=model_namespace.name,
        string_data={
            "HF_ACCESS_TOKEN": hf_access_token,
        },
        wait_for_resource=True,
    ) as secret:
        yield secret


# GPU-based vLLM fixtures for SmolLM-1.7B
@pytest.fixture(scope="function")
def lmeval_vllm_serving_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    vllm_runtime_image: str,
    supported_accelerator_type: str | None,
) -> Generator[ServingRuntime]:
    """vLLM ServingRuntime for GPU-based model deployment in LMEval tests."""
    # Map accelerator type to runtime template
    accelerator_to_template = {
        "nvidia": RuntimeTemplates.VLLM_CUDA,
        "amd": RuntimeTemplates.VLLM_ROCM,
        "gaudi": RuntimeTemplates.VLLM_GAUDI,
    }

    if not supported_accelerator_type:
        pytest.skip("supported_accelerator_type is required for GPU-backed vLLM tests")

    accelerator_type = supported_accelerator_type.lower()
    template_name = accelerator_to_template.get(accelerator_type)

    if not template_name:
        pytest.skip(f"Unsupported accelerator type for vLLM: {accelerator_type}")

    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="lmeval-vllm-runtime",
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=KServeDeploymentType.RAW_DEPLOYMENT,
        runtime_image=vllm_runtime_image,
        support_tgis_open_ai_endpoints=True,
    ) as serving_runtime:
        yield serving_runtime


@pytest.fixture(scope="function")
def lmeval_vllm_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lmeval_vllm_serving_runtime: ServingRuntime,
    supported_accelerator_type: str | None,
) -> Generator[InferenceService]:
    """InferenceService for GPU-based model deployment in LMEval tests."""
    model_path = "HuggingFaceTB/SmolLM-1.7B"
    model_name = "lmeval-model"

    # Get the correct GPU identifier based on accelerator type
    accelerator_type = supported_accelerator_type.lower() if supported_accelerator_type else "nvidia"
    gpu_identifier = ACCELERATOR_IDENTIFIER.get(accelerator_type, Labels.Nvidia.NVIDIA_COM_GPU)

    resources = {
        "requests": {
            "cpu": "2",
            "memory": "8Gi",
            gpu_identifier: "1",
        },
        "limits": {
            "cpu": "3",
            "memory": "8Gi",
            gpu_identifier: "1",
        },
    }

    runtime_args = [
        f"--model={model_path}",
        "--dtype=float16",
        "--max-model-len=2048",
    ]

    env_vars = [
        {"name": "HF_HUB_OFFLINE", "value": "0"},
        {"name": "HF_HUB_ENABLE_HF_TRANSFER", "value": "0"},
    ]

    with create_isvc(
        client=admin_client,
        name=model_name,
        namespace=model_namespace.name,
        runtime=lmeval_vllm_serving_runtime.name,
        model_format=lmeval_vllm_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        resources=resources,
        argument=runtime_args,
        model_env_variables=env_vars,
        min_replicas=1,
    ) as inference_service:
        yield inference_service


@pytest.fixture(scope="function")
def lmevaljob_gpu(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lmeval_vllm_inference_service: InferenceService,
) -> Generator[LMEvalJob]:
    """LMEvalJob for evaluating a GPU-deployed model via vLLM."""
    model_path = "HuggingFaceTB/SmolLM-1.7B"
    model_service = Service(
        name=f"{lmeval_vllm_inference_service.name}-predictor",
        namespace=lmeval_vllm_inference_service.namespace,
    )

    with LMEvalJob(
        client=admin_client,
        namespace=model_namespace.name,
        name=LMEVALJOB_NAME,
        model="local-completions",
        task_list={"taskNames": ["arc_easy"]},
        log_samples=True,
        batch_size="1",
        allow_online=True,
        allow_code_execution=False,
        outputs={"pvcManaged": {"size": "5Gi"}},
        limit="0.01",
        model_args=[
            {"name": "model", "value": lmeval_vllm_inference_service.name},
            {
                "name": "base_url",
                "value": f"http://{model_service.name}.{model_namespace.name}.svc.cluster.local:80/v1/completions",
            },
            {"name": "num_concurrent", "value": "1"},
            {"name": "max_retries", "value": "3"},
            {"name": "tokenized_requests", "value": "False"},
            {"name": "tokenizer", "value": model_path},
        ],
    ) as lmevaljob:
        yield lmevaljob
