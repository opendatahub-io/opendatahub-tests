import pytest
from typing import Any, Generator
import shortuuid
from pytest import FixtureRequest

from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.service import Service

from utilities.infra import create_ns
from utilities.constants import OCIRegistry, MinIo, Protocols, Labels

from kubernetes.dynamic import DynamicClient


# OCI Registry
@pytest.fixture(scope="class")
def oci_namespace(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    with create_ns(
        name=f"{OCIRegistry.Metadata.NAME}-{shortuuid.uuid().lower()}",
        admin_client=admin_client,
    ) as ns:
        yield ns


@pytest.fixture(scope="class")
def oci_registry_pod_with_minio(
    request: FixtureRequest,
    admin_client: DynamicClient,
    oci_namespace: Namespace,
    minio_service: Service,
) -> Generator[Pod, Any, Any]:
    pod_labels = {Labels.Openshift.APP: OCIRegistry.Metadata.NAME}

    if labels := request.param.get("labels"):
        pod_labels.update(labels)

    minio_fqdn = f"{minio_service.name}.{minio_service.namespace}.svc.cluster.local"
    minio_endpoint = f"{minio_fqdn}:{MinIo.Metadata.DEFAULT_PORT}"

    with Pod(
        client=admin_client,
        name=OCIRegistry.Metadata.NAME,
        namespace=oci_namespace.name,
        containers=[
            {
                "args": request.param.get("args"),
                "env": [
                    {"name": "ZOT_STORAGE_STORAGEDRIVER_NAME", "value": OCIRegistry.Storage.STORAGE_DRIVER},
                    {
                        "name": "ZOT_STORAGE_STORAGEDRIVER_ROOTDIRECTORY",
                        "value": OCIRegistry.Storage.STORAGE_DRIVER_ROOT_DIRECTORY,
                    },
                    {"name": "ZOT_STORAGE_STORAGEDRIVER_BUCKET", "value": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS},
                    {"name": "ZOT_STORAGE_STORAGEDRIVER_REGION", "value": OCIRegistry.Storage.STORAGE_DRIVER_REGION},
                    {"name": "ZOT_STORAGE_STORAGEDRIVER_REGIONENDPOINT", "value": f"http://{minio_endpoint}"},
                    {"name": "ZOT_STORAGE_STORAGEDRIVER_ACCESSKEY", "value": MinIo.Credentials.ACCESS_KEY_VALUE},
                    {"name": "ZOT_STORAGE_STORAGEDRIVER_SECRETKEY", "value": MinIo.Credentials.SECRET_KEY_VALUE},
                    {
                        "name": "ZOT_STORAGE_STORAGEDRIVER_SECURE",
                        "value": OCIRegistry.Storage.STORAGE_STORAGEDRIVER_SECURE,
                    },
                    {
                        "name": "ZOT_STORAGE_STORAGEDRIVER_FORCEPATHSTYLE",
                        "value": OCIRegistry.Storage.STORAGE_STORAGEDRIVER_FORCEPATHSTYLE,
                    },
                    {"name": "ZOT_HTTP_ADDRESS", "value": OCIRegistry.Metadata.DEFAULT_HTTP_ADDRESS},
                    {"name": "ZOT_HTTP_PORT", "value": str(OCIRegistry.Metadata.DEFAULT_PORT)},
                    {"name": "ZOT_LOG_LEVEL", "value": "info"},
                ],
                "image": request.param.get("image", OCIRegistry.PodConfig.REGISTRY_IMAGE),
                "name": OCIRegistry.Metadata.NAME,
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "volumeMounts": [
                    {
                        "name": "zot-data",
                        "mountPath": "/var/lib/registry",
                    }
                ],
            }
        ],
        volumes=[
            {
                "name": "zot-data",
                "emptyDir": {},
            }
        ],
        label=pod_labels,
        annotations=request.param.get("annotations"),
    ) as oci_pod:
        oci_pod.wait_for_condition(condition="Ready", status="True")
        yield oci_pod


@pytest.fixture(scope="class")
def oci_registry_service(admin_client: DynamicClient, oci_namespace: Namespace) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        name=OCIRegistry.Metadata.NAME,
        namespace=oci_namespace.name,
        ports=[
            {
                "name": f"{OCIRegistry.Metadata.NAME}-port",
                "port": OCIRegistry.Metadata.DEFAULT_PORT,
                "protocol": Protocols.TCP,
                "targetPort": OCIRegistry.Metadata.DEFAULT_PORT,
            }
        ],
        selector={
            Labels.Openshift.APP: OCIRegistry.Metadata.NAME,
        },
        session_affinity="ClientIP",
    ) as oci_service:
        yield oci_service


@pytest.fixture(scope="class")
def oci_registry_route(admin_client: DynamicClient, oci_registry_service: Service) -> Generator[Route, Any, Any]:
    with Route(
        client=admin_client,
        name=OCIRegistry.Metadata.NAME,
        namespace=oci_registry_service.namespace,
        service=oci_registry_service.name,
    ) as oci_route:
        yield oci_route
