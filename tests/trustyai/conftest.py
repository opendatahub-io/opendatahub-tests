import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.trustyai_service import TrustyAIService

from tests.trustyai.constants import TRUSTYAI_SERVICE
from utilities.constants import MODELMESH_SERVING
from tests.trustyai.utils import update_configmap_data

MINIO: str = "minio"
OPENDATAHUB_IO: str = "opendatahub.io"


@pytest.fixture(scope="class")
def trustyai_service_with_pvc_storage(
    admin_client: DynamicClient,
    ns_with_modelmesh_enabled: Namespace,
    modelmesh_serviceaccount: ServiceAccount,
    cluster_monitoring_config: ConfigMap,
    user_workload_monitoring_config: ConfigMap,
) -> TrustyAIService:
    with TrustyAIService(
        client=admin_client,
        name=TRUSTYAI_SERVICE,
        namespace=ns_with_modelmesh_enabled.name,
        storage={"format": "PVC", "folder": "/inputs", "size": "1Gi"},
        data={"filename": "data.csv", "format": "CSV"},
        metrics={"schedule": "5s"},
    ) as trustyai_service:
        trustyai_deployment = Deployment(
            namespace=ns_with_modelmesh_enabled.name, name=TRUSTYAI_SERVICE, wait_for_resource=True
        )
        trustyai_deployment.wait_for_replicas()
        yield trustyai_service


@pytest.fixture(scope="class")
def modelmesh_serviceaccount(admin_client: DynamicClient, ns_with_modelmesh_enabled: Namespace) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client, name=f"{MODELMESH_SERVING}-sa", namespace=ns_with_modelmesh_enabled.name
    ) as sa:
        yield sa


@pytest.fixture(scope="session")
def cluster_monitoring_config(admin_client: DynamicClient) -> ConfigMap:
    name = "cluster-monitoring-config"
    namespace = "openshift-monitoring"
    data = {"config.yaml": yaml.dump({"enableUserWorkload": "true"})}
    cm = ConfigMap(client=admin_client, name=name, namespace=namespace)
    if cm.exists:  # This resource is usually created when doing exploratory testing, add this exception for convenience
        with update_configmap_data(configmap=cm, data=data) as cm:
            yield cm

    else:
        with ConfigMap(
            client=admin_client,
            name=name,
            namespace=namespace,
            data=data,
        ) as cm:
            yield cm


@pytest.fixture(scope="session")
def user_workload_monitoring_config(admin_client: DynamicClient) -> ConfigMap:
    name = "user-workload-monitoring-config"
    namespace = "openshift-user-workload-monitoring"
    data = {"config.yaml": yaml.dump({"prometheus": {"logLevel": "debug", "retention": "15d"}})}
    cm = ConfigMap(client=admin_client, name=name, namespace=namespace)
    if cm.exists:  # This resource is usually created when doing exploratory testing, add this exception for convenience
        with update_configmap_data(configmap=cm, data=data) as cm:
            yield cm

    else:
        with ConfigMap(
            client=admin_client,
            name=name,
            namespace=namespace,
            data=data,
        ) as cm:
            yield cm


@pytest.fixture(scope="class")
def minio_pod(admin_client: DynamicClient, ns_with_modelmesh_enabled: Namespace) -> Pod:
    with Pod(
        client=admin_client,
        name=MINIO,
        namespace=ns_with_modelmesh_enabled.name,
        containers=[
            {
                "args": [
                    "server",
                    "/data1",
                ],
                "env": [
                    {
                        "name": "MINIO_ACCESS_KEY",
                        "value": "THEACCESSKEY",
                    },
                    {
                        "name": "MINIO_SECRET_KEY",
                        "value": "THESECRETKEY",
                    },
                ],
                "image": "quay.io/trustyai/modelmesh-minio-examples@"
                "sha256:e8360ec33837b347c76d2ea45cd4fea0b40209f77520181b15e534b101b1f323",
                "name": MINIO,
            }
        ],
        label={"app": "minio", "maistra.io/expose-route": "true"},
        annotations={"sidecar.istio.io/inject": "true"},
    ) as minio_pod:
        yield minio_pod


@pytest.fixture(scope="class")
def minio_service(admin_client: DynamicClient, ns_with_modelmesh_enabled: Namespace) -> Service:
    with Service(
        client=admin_client,
        name=MINIO,
        namespace=ns_with_modelmesh_enabled.name,
        ports=[
            {
                "name": "minio-client-port",
                "port": 9000,
                "protocol": "TCP",
                "targetPort": 9000,
            }
        ],
        selector={
            "app": MINIO,
        },
    ) as minio_service:
        yield minio_service


@pytest.fixture(scope="class")
def minio_data_connection(
    admin_client: DynamicClient, ns_with_modelmesh_enabled: Namespace, minio_pod: Pod, minio_service: Service
) -> Secret:
    with Secret(
        client=admin_client,
        name="aws-connection-minio-data-connection",
        namespace=ns_with_modelmesh_enabled.name,
        data_dict={
            "AWS_ACCESS_KEY_ID": "VEhFQUNDRVNTS0VZ",
            "AWS_DEFAULT_REGION": "dXMtc291dGg=",
            "AWS_S3_BUCKET": "bW9kZWxtZXNoLWV4YW1wbGUtbW9kZWxz",
            "AWS_S3_ENDPOINT": "aHR0cDovL21pbmlvOjkwMDA=",
            "AWS_SECRET_ACCESS_KEY": "VEhFU0VDUkVUS0VZ",  # pragma: allowlist secret
        },
        label={
            f"{OPENDATAHUB_IO}/dashboard": "true",
            f"{OPENDATAHUB_IO}/managed": "true",
        },
        annotations={
            f"{OPENDATAHUB_IO}/connection-type": "s3",
            "openshift.io/display-name": "Minio Data Connection",
        },
    ) as minio_secret:
        yield minio_secret
