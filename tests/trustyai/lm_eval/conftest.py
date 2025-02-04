from typing import Generator

import pytest
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

from tests.trustyai.constants import TIMEOUT_10MIN
from utilities.constants import Labels


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
            "taskRecipes": [
                {"card": {"name": "cards.wnli"}, "template": "templates.classification.multi_class.relation.default"}
            ]
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
) -> Generator[LMEvalJob, None, None]:
    with LMEvalJob(
        client=admin_client,
        name="lmeval-test",
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
def patched_trustyai_operator_configmap_allow_online(admin_client: DynamicClient) -> Generator[ConfigMap, None, None]:
    namespace: str = py_config["applications_namespace"]
    trustyai_service_operator: str = "trustyai-service-operator"

    configmap: ConfigMap = ConfigMap(
        client=admin_client, name=f"{trustyai_service_operator}-config", namespace=namespace, ensure_exists=True
    )
    with ResourceEditor(
        patches={configmap: {"data": {"lmes-allow-online": "true", "lmes-allow-code-execution": "true"}}}
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
) -> Generator[PersistentVolumeClaim, None, None]:
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
) -> Generator[Pod, None, None]:
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
        pod.wait_for_status(status=Pod.Status.SUCCEEDED, timeout=TIMEOUT_10MIN)
        yield pod
