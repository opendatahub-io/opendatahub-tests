from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from pytest import Config, FixtureRequest

from tests.ai_safety.lm_eval.utils import get_lmevaljob_pod
from utilities.constants import ApiGroups, Labels, Protocols, SeaweedFs, Timeout
from utilities.exceptions import MissingParameter
from utilities.general import get_s3_secret_dict

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
) -> Generator[LMEvalJob, None, None]:
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
def lmeval_seaweedfs_deployment(
    admin_client: DynamicClient, seaweedfs_namespace: Namespace, pvc_seaweedfs_namespace: PersistentVolumeClaim
) -> Generator[Deployment, Any, Any]:
    seaweedfs_app_label = {"app": SeaweedFs.Metadata.NAME}
    initialization_command = (
        "for attempt in $(seq 1 60); do "
        f"wget -q --spider http://127.0.0.1:{SeaweedFs.Metadata.DEFAULT_PORT}/status && break; "
        '[ "$attempt" -eq 60 ] && exit 1; sleep 2; '
        "done; "
        'echo "s3.configure -user admin -access_key $accesskey -secret_key $secretkey -actions Admin -apply" '
        "| /usr/bin/weed shell && "
        "echo 's3.bucket.create -name models' | /usr/bin/weed shell"
    )
    with Deployment(
        client=admin_client,
        name=SeaweedFs.Metadata.NAME,
        namespace=seaweedfs_namespace.name,
        replicas=1,
        selector={"matchLabels": seaweedfs_app_label},
        template={
            "metadata": {"labels": seaweedfs_app_label},
            "spec": {
                "volumes": [
                    {"name": "seaweedfs-storage", "persistentVolumeClaim": {"claimName": pvc_seaweedfs_namespace.name}}
                ],
                "containers": [
                    {
                        "name": SeaweedFs.Metadata.NAME,
                        "image": SeaweedFs.PodConfig.IMAGE,
                        "args": ["server", "-dir=/data", "-s3", "-iam", "-master.volumePreallocate=false"],
                        "env": [
                            {"name": "accesskey", "value": SeaweedFs.Credentials.ACCESS_KEY_VALUE},
                            {"name": "secretkey", "value": SeaweedFs.Credentials.SECRET_KEY_VALUE},
                        ],
                        "ports": [{"containerPort": SeaweedFs.Metadata.DEFAULT_PORT}],
                        "volumeMounts": [{"name": "seaweedfs-storage", "mountPath": "/data"}],
                        "lifecycle": {"postStart": {"exec": {"command": ["/bin/sh", "-c", initialization_command]}}},
                    }
                ],
            },
        },
        label=seaweedfs_app_label,
        wait_for_resource=True,
    ) as deployment:
        deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_20MIN)
        yield deployment


@pytest.fixture(scope="function")
def lmeval_seaweedfs_copy_pod(
    admin_client: DynamicClient,
    seaweedfs_namespace: Namespace,
    lmeval_seaweedfs_deployment: Deployment,
    seaweedfs_service: Service,
) -> Generator[Pod, Any, Any]:
    with Pod(
        client=admin_client,
        name="copy-to-seaweedfs",
        namespace=seaweedfs_namespace.name,
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
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
            }
        ],
        containers=[
            {
                "name": "seaweedfs-uploader",
                "image": SeaweedFs.PodConfig.IMAGE,
                "command": [
                    "/usr/bin/weed",
                    "filer.copy",
                    "/shared/data/",
                    (
                        f"http://{seaweedfs_service.name}.{seaweedfs_service.namespace}.svc.cluster.local:"
                        f"{SeaweedFs.Metadata.FILER_PORT}/buckets/models/"
                    ),
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
def lmeval_seaweedfs_data_connection(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    seaweedfs_service: Service,
) -> Generator[Secret, Any, Any]:
    """Create a data connection Secret pointing at the SeaweedFS-backed 'models' bucket."""
    data_dict = get_s3_secret_dict(
        aws_access_key=SeaweedFs.Credentials.ACCESS_KEY_VALUE,
        aws_secret_access_key=SeaweedFs.Credentials.SECRET_KEY_VALUE,
        aws_s3_bucket="models",
        aws_s3_endpoint=(
            f"{Protocols.HTTP}://{seaweedfs_service.instance.spec.clusterIP}:{SeaweedFs.Metadata.DEFAULT_PORT}"
        ),
        aws_s3_region="us-south",
    )
    with Secret(
        client=admin_client,
        name="aws-connection-seaweedfs-data-connection",
        namespace=model_namespace.name,
        data_dict=data_dict,
        label={
            Labels.OpenDataHub.DASHBOARD: "true",
            Labels.OpenDataHubIo.MANAGED: "true",
        },
        annotations={
            f"{ApiGroups.OPENDATAHUB_IO}/connection-type": "s3",
            "openshift.io/display-name": "SeaweedFS Data Connection",
        },
    ) as seaweedfs_secret:
        yield seaweedfs_secret


@pytest.fixture(scope="function")
def lmevaljob_s3_offline(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    lmeval_seaweedfs_deployment: Deployment,
    seaweedfs_service: Service,
    lmeval_seaweedfs_copy_pod: Pod,
    lmeval_seaweedfs_data_connection: Secret,
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
                    "accessKeyId": {"name": lmeval_seaweedfs_data_connection.name, "key": "AWS_ACCESS_KEY_ID"},
                    "secretAccessKey": {
                        "name": lmeval_seaweedfs_data_connection.name,
                        "key": "AWS_SECRET_ACCESS_KEY",
                    },
                    "bucket": {"name": lmeval_seaweedfs_data_connection.name, "key": "AWS_S3_BUCKET"},
                    "endpoint": {"name": lmeval_seaweedfs_data_connection.name, "key": "AWS_S3_ENDPOINT"},
                    "region": {"name": lmeval_seaweedfs_data_connection.name, "key": "AWS_DEFAULT_REGION"},
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
def lmevaljob_vllm_emulator_pod(
    admin_client: DynamicClient, lmevaljob_vllm_emulator: LMEvalJob
) -> Generator[Pod, Any, Any]:
    yield get_lmevaljob_pod(client=admin_client, lmevaljob=lmevaljob_vllm_emulator)


@pytest.fixture(scope="function")
def lmevaljob_s3_offline_pod(admin_client: DynamicClient, lmevaljob_s3_offline: LMEvalJob) -> Generator[Pod, Any, Any]:
    yield get_lmevaljob_pod(client=admin_client, lmevaljob=lmevaljob_s3_offline)


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
