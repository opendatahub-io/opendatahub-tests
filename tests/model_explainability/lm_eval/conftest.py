from typing import Generator, Any

import pytest
from ocp_resources.route import Route
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

from utilities.constants import Labels, Timeout, Annotations, Protocols

VLLM_EMULATOR: str = "vllm-emulator"
VLLM_EMULATOR_PORT: int = 8000
LMEVALJOB_NAME: str = "lmeval-test-job"


@pytest.fixture(scope="function")
def lmevaljob_hf_pod(admin_client: DynamicClient, lmevaljob_hf: LMEvalJob) -> Generator[Pod, Any, Any]:
    lmeval_pod = Pod(
        client=admin_client,
        namespace=lmevaljob_hf.namespace,
        name=lmevaljob_hf.name,
    )

    # TODO: Check if we can rely on LMEvalJob instead of pod
    lmeval_pod.wait(timeout=Timeout.TIMEOUT_2MIN)

    yield lmeval_pod


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
def lmevaljob_vllm_emulator_pod(
    admin_client: DynamicClient, lmevaljob_vllm_emulator: LMEvalJob
) -> Generator[Pod, Any, Any]:
    lmeval_pod = Pod(
        client=admin_client,
        namespace=lmevaljob_vllm_emulator.namespace,
        name=lmevaljob_vllm_emulator.name,
    )

    # TODO: Check if we can rely on LMEvalJob instead of pod
    lmeval_pod.wait(timeout=Timeout.TIMEOUT_2MIN)

    yield lmeval_pod


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
                "command": ["/bin/sh", "-c", "cp -r /mnt/data/. /mnt/pvc/"],
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
