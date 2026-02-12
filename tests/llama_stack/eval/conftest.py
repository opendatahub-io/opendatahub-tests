from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from timeout_sampler import TimeoutSampler

from tests.llama_stack.eval.constants import DK_CUSTOM_DATASET_IMAGE
from tests.llama_stack.eval.utils import wait_for_dspa_pods
from utilities.constants import MinIo


@pytest.fixture(scope="class")
def dataset_pvc(admin_client, model_namespace) -> Generator[PersistentVolumeClaim, Any, Any]:
    """
    Creates a PVC to store the custom dataset.
    """
    with PersistentVolumeClaim(
        client=admin_client,
        namespace=model_namespace.name,
        name="dataset-pvc",
        size="1Gi",
        accessmodes="ReadWriteOnce",
        label={"app.kubernetes.io/name": "dataset-storage"},
    ) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def dataset_upload(
    admin_client: DynamicClient, model_namespace: Namespace, dataset_pvc: PersistentVolumeClaim
) -> Generator[dict[str, Any], None, None]:
    """
    Copies dataset files from an image into the PVC at the location expected by LM-Eval
    """

    dataset_target_path = "/opt/app-root/src/hf_home"

    with Pod(
        client=admin_client,
        namespace=model_namespace.name,
        name="dataset-copy-to-pvc",
        label={"trustyai-tests": "dataset-upload"},
        security_context={"fsGroup": 1001, "seccompProfile": {"type": "RuntimeDefault"}},
        containers=[
            {
                "name": "dataset-copy-to-pvc",
                "image": DK_CUSTOM_DATASET_IMAGE,
                "command": ["/bin/sh", "-c", "cp --verbose -r /models/* /mnt/pvc"],
                "securityContext": {
                    "runAsUser": 1001,
                    "runAsNonRoot": True,
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": [{"mountPath": "/mnt/pvc", "name": "pvc-volume"}],
            }
        ],
        restart_policy="Never",
        volumes=[{"name": "pvc-volume", "persistentVolumeClaim": {"claimName": dataset_pvc.name}}],
    ) as pod:
        pod.wait_for_status(status=Pod.Status.SUCCEEDED)
        yield {
            "pod": pod,
            "dataset_path": f"{dataset_target_path}/example-dk-bench-input-bmo.jsonl",
        }


@pytest.fixture(scope="function")
def teardown_lmeval_job_pod(admin_client, model_namespace) -> None:
    """
    Cleans up the evaluation Pods created by the LMEval job during the test run.

    This teardown logic is **CRITICAL** for ensuring dependent resources (like the PVC)
    can be deleted. If the Pod is not completely deleted and confirmed gone
    before the PVC teardown begins, the PVC will get stuck in a **'Terminating'** state.
    This happens because Kubernetes prevents the volume from being deleted
    while a resource (the Pod) is still actively using or referencing it,
    causing the entire test run to time out waiting for resource cleanup.
    """
    yield

    if pods := [
        pod
        for pod in Pod.get(
            client=admin_client, namespace=model_namespace.name, label_selector="app.kubernetes.io/name=ta-lmes"
        )
    ]:
        for pod in pods:
            pod.delete()


@pytest.fixture(scope="class")
def dspa(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_pod: Pod,
    minio_service: Service,
    dspa_s3_secret: Secret,
) -> Generator[DataSciencePipelinesApplication, Any, Any]:
    """
    Creates a DataSciencePipelinesApplication with MinIO object storage.
    """

    with DataSciencePipelinesApplication(
        client=admin_client,
        name="dspa",
        namespace=model_namespace.name,
        dsp_version="v2",
        pod_to_pod_tls=True,
        api_server={
            "deploy": True,
            "enableOauth": True,
            "enableSamplePipeline": False,
            "cacheEnabled": True,
            "artifactSignedURLExpirySeconds": 60,
            "pipelineStore": "kubernetes",
        },
        database={
            "disableHealthCheck": False,
            "mariaDB": {
                "deploy": True,
                "pipelineDBName": "mlpipeline",
                "pvcSize": "10Gi",
                "username": "mlpipeline",
            },
        },
        object_storage={
            "disableHealthCheck": False,
            "enableExternalRoute": False,
            "externalStorage": {
                "bucket": "ods-ci-ds-pipelines",
                "host": f"{minio_service.instance.spec.clusterIP}:{MinIo.Metadata.DEFAULT_PORT}",
                "region": "us-east-1",
                "scheme": "http",
                "s3CredentialsSecret": {
                    "accessKey": "AWS_ACCESS_KEY_ID",  # pragma: allowlist secret
                    "secretKey": "AWS_SECRET_ACCESS_KEY",  # pragma: allowlist secret
                    "secretName": dspa_s3_secret.name,
                },
            },
        },
        persistence_agent={
            "deploy": True,
            "numWorkers": 2,
        },
        scheduled_workflow={
            "deploy": True,
            "cronScheduleTimezone": "UTC",
        },
    ) as dspa_resource:
        wait_for_dspa_pods(
            admin_client=admin_client,
            namespace=model_namespace.name,
            dspa_name=dspa_resource.name,
        )
        yield dspa_resource


@pytest.fixture(scope="class")
def dspa_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_service: Service,
) -> Generator[Secret, Any, Any]:
    """
    Creates a secret for DSPA S3 credentials using MinIO.
    """
    with Secret(
        client=admin_client,
        name="dashboard-dspa-secret",
        namespace=model_namespace.name,
        string_data={
            "AWS_ACCESS_KEY_ID": MinIo.Credentials.ACCESS_KEY_VALUE,
            "AWS_SECRET_ACCESS_KEY": MinIo.Credentials.SECRET_KEY_VALUE,
            "AWS_DEFAULT_REGION": "us-east-1",
        },
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def dspa_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    dspa: DataSciencePipelinesApplication,
) -> Generator[Route, Any, Any]:
    """
    Retrieves the Route for the DSPA API server.
    """

    def _get_dspa_route() -> Route | None:
        routes = list(
            Route.get(
                client=admin_client,
                namespace=model_namespace.name,
                name="ds-pipeline-dspa",
            )
        )
        return routes[0] if routes else None

    for route in TimeoutSampler(wait_timeout=240, sleep=5, func=_get_dspa_route):
        if route:
            yield route
            return
