from typing import Generator, Any

import pytest
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from pytest import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from pytest_testconfig import py_config

from tests.model_explainability.lm_eval.utils import get_lmevaljob_pod
from utilities.constants import Labels, Timeout, Annotations, Protocols, MinIo

VLLM_EMULATOR: str = "vllm-emulator"
VLLM_EMULATOR_PORT: int = 8000
LMEVALJOB_NAME: str = "lmeval-test-job"


@pytest.fixture(scope="function")
def lmevaljob_hf(
    admin_client: DynamicClient, model_namespace: Namespace, patched_trustyai_operator_configmap_allow_online: ConfigMap
) -> Generator[LMEvalJob, None, None]:
    with LMEvalJob(
        client=admin_client,
        name="test-job",
        namespace=model_namespace.name,
        model="hf",
        model_args=[{"name": "pretrained", "value": "google/flan-t5-base"}],
        task_list={
            "custom": {
                "systemPrompts": [
                    {"name": "sp_0", "value": "Be concise. At every point give the shortest acceptable answer."}
                ],
                "templates": [
                    {
                        "name": "tp_0",
                        "value": '{ "__type__": "input_output_template", '
                        '"input_format": "{text_a_type}: {text_a}\\n'
                        '{text_b_type}: {text_b}", '
                        '"output_format": "{label}", '
                        '"target_prefix": '
                        '"The {type_of_relation} class is ", '
                        '"instruction": "Given a {text_a_type} and {text_b_type} '
                        'classify the {type_of_relation} of the {text_b_type} to one of {classes}.",'
                        ' "postprocessors": [ "processors.take_first_non_empty_line",'
                        ' "processors.lower_case_till_punc" ] }',
                    }
                ],
            },
            "taskRecipes": [
                {"card": {"name": "cards.wnli"}, "systemPrompt": {"ref": "sp_0"}, "template": {"ref": "tp_0"}}
            ],
        },
        log_samples=True,
        allow_online=True,
        allow_code_execution=True,
    ) as job:
        yield job


@pytest.fixture(scope="function")
def lmevaljob_local_offline(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    patched_trustyai_operator_configmap_allow_online: ConfigMap,
    lmeval_data_downloader_pod: Pod,
) -> Generator[LMEvalJob, Any, Any]:
    with LMEvalJob(
        client=admin_client,
        name=LMEVALJOB_NAME,
        namespace=model_namespace.name,
        model="hf",
        model_args=[{"name": "pretrained", "value": "/opt/app-root/src/hf_home/flan"}],
        task_list=request.param.get("task_list"),
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


@pytest.fixture(scope="function")
def lmevaljob_vllm_emulator(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    patched_trustyai_operator_configmap_allow_online: ConfigMap,
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
                "value": f"http://{vllm_emulator_service.name}:{str(VLLM_EMULATOR_PORT)}/v1/completions",
            },
            {"name": "num_concurrent", "value": "1"},
            {"name": "max_retries", "value": "3"},
            {"name": "tokenized_requests", "value": "False"},
            {"name": "tokenizer", "value": "ibm-granite/granite-guardian-3.1-8b"},
        ],
    ) as job:
        yield job


@pytest.fixture(scope="function")
def patched_trustyai_operator_configmap_allow_online(admin_client: DynamicClient) -> Generator[ConfigMap, Any, Any]:
    namespace: str = py_config["applications_namespace"]
    trustyai_service_operator: str = "trustyai-service-operator"

    configmap: ConfigMap = ConfigMap(
        client=admin_client, name=f"{trustyai_service_operator}-config", namespace=namespace, ensure_exists=True
    )
    with ResourceEditor(
        patches={
            configmap: {
                "metadata": {"annotations": {Annotations.OpenDataHubIo.MANAGED: "false"}},
                "data": {"lmes-allow-online": "true", "lmes-allow-code-execution": "true"},
            }
        }
    ):
        deployment: Deployment = Deployment(
            client=admin_client,
            name=f"{trustyai_service_operator}-controller-manager",
            namespace=namespace,
            ensure_exists=True,
        )
        num_replicas: int = deployment.replicas
        deployment.scale_replicas(replica_count=0)
        deployment.scale_replicas(replica_count=num_replicas)
        deployment.wait_for_replicas()
        yield configmap


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
        containers=[
            {
                "name": "data",
                "image": request.param.get("image"),
                "command": ["/bin/sh", "-c", "cp -r /mnt/data/. /mnt/pvc/ && chmod -R g+w /mnt/pvc/datasets"],
                "securityContext": {
                    "runAsUser": 1000,
                    "runAsNonRoot": True,
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": [{"mountPath": "/mnt/pvc", "name": "pvc-volume"}],
            }
        ],
        restart_policy="Never",
        volumes=[{"name": "pvc-volume", "persistentVolumeClaim": {"claimName": "lmeval-data"}}],
    ) as pod:
        pod.wait_for_status(status=Pod.Status.SUCCEEDED, timeout=Timeout.TIMEOUT_10MIN)
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
                        "@sha256:4214f31bff9de6cc723da23324fb8974cea8abadcab621d85a97a3503cabbdc6",
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
        deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_10MIN)
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
                "name": "copy-data",
                "image": "quay.io/trustyai_testing/lmeval-assets-flan-arceasy"
                "@sha256:11cc9c2f38ac9cc26c4fab1a01a8c02db81c8f4801b5d2b2b90f90f91b97ac98",
                "command": ["/bin/sh", "-c"],
                "args": ["cp -r /mnt/data /shared"],
                "volumeMounts": [{"name": "shared-data", "mountPath": "/shared"}],
            }
        ],
        containers=[
            {
                "name": "minio-uploader",
                "image": "quay.io/minio/mc@sha256:470f5546b596e16c7816b9c3fa7a78ce4076bb73c2c73f7faeec0c8043923123",
                "command": ["/bin/sh", "-c"],
                "args": [
                    f"mc alias set myminio http://{minio_service.name}:{MinIo.Metadata.DEFAULT_PORT} "
                    f"{MinIo.Credentials.ACCESS_KEY_VALUE} {MinIo.Credentials.SECRET_KEY_VALUE} &&\n"
                    "mc mb --ignore-existing myminio/models &&\n"
                    "mc cp --recursive /shared/data/ myminio/models"
                ],
                "volumeMounts": [{"name": "shared-data", "mountPath": "/shared"}],
            }
        ],
        wait_for_resource=True,
    ) as pod:
        pod.wait_for_status(status=Pod.Status.SUCCEEDED)
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
def lmevaljob_vllm_emulator_pod(
    admin_client: DynamicClient, lmevaljob_vllm_emulator: LMEvalJob
) -> Generator[Pod, Any, Any]:
    yield get_lmevaljob_pod(client=admin_client, lmevaljob=lmevaljob_vllm_emulator)


@pytest.fixture(scope="function")
def lmevaljob_s3_offline_pod(admin_client: DynamicClient, lmevaljob_s3_offline: LMEvalJob) -> Generator[Pod, Any, Any]:
    yield get_lmevaljob_pod(client=admin_client, lmevaljob=lmevaljob_s3_offline)
