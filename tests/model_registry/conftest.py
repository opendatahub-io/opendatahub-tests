import pytest
from typing import Generator, Any
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.deployment import Deployment
from ocp_resources.model_registry import ModelRegistry
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient

from utilities.infra import create_ns


LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def model_registry_namespace(admin_client: DynamicClient) -> Generator[Namespace,Any,Any]:
    ns = Namespace(name="rhoai-model-registries", client=admin_client)
    if ns.exists:
        yield ns
    else:
        with create_ns(
            name="rhoai-model-registries",
            admin_client=admin_client,
            teardown=False,
            ensure_exists=True,
        ) as ns:
            yield ns


#ModelRegistryDBService
@pytest.fixture(scope="class")
def model_registry_db_service(admin_client: DynamicClient, model_registry_namespace: Namespace) -> Generator[Service,Any,Any]:
    with Service(
        client=admin_client,
        name="model-registry-db",
        namespace=model_registry_namespace.name,
        ports=[{
            "name": "mysql",
            "nodePort": 0,
            "port": 3306,
            "protocol": "TCP",
            "appProtocol": "tcp",
            "targetPort": 3306,
        }],
        selector={
            "name": "model-registry-db",
        },
        label={
            "app.kubernetes.io/name": "model-registry-db",
            "app.kubernetes.io/instance": "model-registry-db",
            "app.kubernetes.io/part-of": "model-registry-db",
            "app.kubernetes.io/managed-by": "kustomize",
        },
        annotations={
            "template.openshift.io/expose-uri": "mysql://{.spec.clusterIP}:{.spec.ports[?(.name==\mysql\)].port}",
        }
    ) as mr_db_service:
        yield mr_db_service


#ModelRegistryDBPVC
@pytest.fixture(scope="class")
def model_registry_db_pvc(
    admin_client: DynamicClient,
    model_registry_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    pvc_kwargs = {
        "access_mode": "ReadWriteOnce",
        "name": "model-registry-db",
        "namespace": model_registry_namespace.name,
        "client": admin_client,
        "size": "5Gi",
        "label": {
            "app.kubernetes.io/name": "model-registry-db",
            "app.kubernetes.io/instance": "model-registry-db",
            "app.kubernetes.io/part-of": "model-registry-db",
            "app.kubernetes.io/managed-by": "kustomize",
        }
    }

    with PersistentVolumeClaim(**pvc_kwargs) as pvc:
        yield pvc


#ModelRegistryDBSecret
@pytest.fixture(scope="class")
def model_registry_db_secret(
    admin_client: DynamicClient,
    model_registry_namespace: Namespace,
) -> Generator[Secret,Any,Any]:
    with Secret(
        client=admin_client,
        name="model-registry-db",
        namespace=model_registry_namespace.name,
        string_data={
            "database-name": "model_registry",
            "database-password": "TheBlurstOfTimes",  #not secret
            "database-user": "mlmduser",  #not secret
        },
        label= {
            "app.kubernetes.io/name": "model-registry-db",
            "app.kubernetes.io/instance": "model-registry-db",
            "app.kubernetes.io/part-of": "model-registry-db",
            "app.kubernetes.io/managed-by": "kustomize",
        },
        annotations= {
            "template.openshift.io/expose-database_name": "'{.data[''database-name'']}'",
            "template.openshift.io/expose-password": "'{.data[''database-password'']}'",
            "template.openshift.io/expose-username": "'{.data[''database-user'']}'",
        }
    ) as mr_db_secret:
        yield mr_db_secret


#ModelRegistryDBDeployment
@pytest.fixture(scope="class")
def model_registry_db_deployment(
    admin_client: DynamicClient,
    model_registry_namespace: Namespace,
) -> Generator[Deployment,Any,Any]:
    with Deployment(
        # name= "model-registry-db",
        # namespace=model_registry_namespace.name,
        # annotations={
        #     "template.alpha.openshift.io/wait-for-ready": "true",
        # },
        # label= {
        #     "app.kubernetes.io/name": "model-registry-db",
        #     "app.kubernetes.io/instance": "model-registry-db",
        #     "app.kubernetes.io/part-of": "model-registry-db",
        #     "app.kubernetes.io/managed-by": "kustomize",
        # },
        # replicas= 1,
        # revision_history_limit= 0,
        # selector={
        #     "matchLabels":{
        #         "name": "model-registry-db"
        #     }
        # },
        # strategy= {"type": "Recreate"},
        # template={
        #     "metadata":{
        #         "labels":{
        #             "name": "model-registry-db",
        #             "sidecar.istio.io/inject": "false"
        #         }
        #     },
        #     "spec":{
        #         "containers":[{
        #             "env":[
        #                 {"name": "MYSQL_USER","valueFrom":{"secretKeyRef":{"key": "database-user","name": "model-registry-db"}}},
        #                 {"name": "MYSQL_PASSWORD", "valueFrom":{ "secretKeyRef":{"key": "database-password","name": "model-registry-db"}}},
        #                 {"name": "MYSQL_ROOT_PASSWORD","valueFrom":{"secretKeyRef":{"key": "database-password","name": "model-registry-db"}}},
        #                 {"name": "MYSQL_DATABASE","valueFrom":{"secretKeyRef":{"key": "database-name","name": "model-registry-db"}}},
        #             ],
        #             "args":[
        #                 "--datadir",
        #                 "/var/lib/mysql/datadir",
        #                 "--default-authentication-plugin=mysql_native_password"
        #             ],
        #             "image": "mysql:8.3.0",
        #             "imagePullPolicy": "IfNotPresent",
        #             "livenessProbe":{
        #                 "exec":{
        #                     "command": ["/bin/bash", "-c", "mysqladmin -u\${MYSQL_USER} -p\${MYSQL_ROOT_PASSWORD} ping"]
        #                 },
        #                     "initialDelaySeconds": 15, 
        #                     "periodSeconds": 10, 
        #                     "timeoutSeconds": 5
        #             },
        #             "name": "mysql",
        #             "ports":[{"containerPort": 3306, "protocol": "TCP"}],
        #             "readinessProbe":{
        #                 "exec":{
        #                     "command": ["/bin/bash", "-c", "mysql -D \${MYSQL_DATABASE} -u\${MYSQL_USER} -p\${MYSQL_ROOT_PASSWORD} -e 'SELECT 1'"]
        #                 },
        #                     "initialDelaySeconds": 10,
        #                     "timeoutSeconds": 5
        #             },
        #             "securityContext":{
        #                 "capabilities" : {},
        #                 "priviledged": "false"
        #             },
        #             "terminationMessagePath": "/dev/termination-log",
        #             "volumeMounts":[{"mountPath": "/var/lib/mysql", "name": "model-registry-db-data"}]
        #             },
        #             {
        #             "dnsPolicy": "ClusterFirst",
        #             "restartPolicy": "Always",
        #             "volumes":[{
        #                 "name": "model-registry-db-data",
        #                 "persistentVolumeClaim":{
        #                     "claimName": "model-registry-db"
        #                 }
        #             }],
        #             }]
        #         }
        # }
        yaml_file="utilities/manifests/db.yaml",
        namespace=model_registry_namespace.name
    ) as mr_db_deployment:
        yield mr_db_deployment


@pytest.fixture(scope="class")
def model_registry_instance(
    admin_client: DynamicClient,
    model_registry_namespace: Namespace,
    model_registry_db_deployment: Deployment,
    model_registry_db_secret: Secret
) -> Generator[ModelRegistry,Any,Any]:
    with ModelRegistry(
        name="model-registry",
        namespace=model_registry_namespace.name,
        label= {
            "app.kubernetes.io/name": "model-registry",
            "app.kubernetes.io/instance": "model-registry",
            "app.kubernetes.io/part-of": "model-registry-operator",
            "app.kubernetes.io/managed-by": "kustomize",
            "app.kubernetes.io/created-by": "model-registry-operator"
        },
        grpc={},
        rest={},
        istio={
            "authProvider": "redhat-ods-applications-auth-provider",
            "gateway":{
                "grpc":{
                    "tls": {}
                },
                "rest":{
                    "tls": {}
                }
            }
        },
        mysql={
            "host": f"{model_registry_db_deployment.name}.svc.cluster.local",
            "database": model_registry_db_secret.string_data["database-name"],
            "passwordSecret":{
                "key": "database-password",
                "name": "model-registry-db"
            },
            "port": 3306,
            "skipDBCreation": False,
            "username": model_registry_db_secret.string_data["database-user"]
        }
    ) as mr:
        yield mr
