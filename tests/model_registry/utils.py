from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.model_registry import ModelRegistry
from kubernetes.dynamic.exceptions import ResourceNotFoundError

from tests.model_registry.constants import MR_DB_IMAGE_DIGEST
from utilities.exceptions import ProtocolNotSupportedError, TooManyServicesError
from utilities.constants import Protocols, Annotations

ADDRESS_ANNOTATION_PREFIX: str = "routing.opendatahub.io/external-address-"


def get_mr_service_by_label(client: DynamicClient, ns: Namespace, mr_instance: ModelRegistry) -> Service:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        ns (Namespace): Namespace object where to find the Service
        mr_instance (ModelRegistry): Model Registry instance

    Returns:
        Service: The matching Service

    Raises:
        ResourceNotFoundError: if no service is found.
    """
    if svc := [
        svcs
        for svcs in Service.get(
            dyn_client=client,
            namespace=ns.name,
            label_selector=f"app={mr_instance.name},component=model-registry",
        )
    ]:
        if len(svc) == 1:
            return svc[0]
        raise TooManyServicesError(svc)
    raise ResourceNotFoundError(f"{mr_instance.name} has no Service")


def get_endpoint_from_mr_service(client: DynamicClient, svc: Service, protocol: str) -> str:
    if protocol in (Protocols.REST, Protocols.GRPC):
        return svc.instance.metadata.annotations[f"{ADDRESS_ANNOTATION_PREFIX}{protocol}"]
    else:
        raise ProtocolNotSupportedError(protocol)


def get_model_registry_deployment_template_dict(secret_name: str, resource_name: str) -> dict[str, Any]:
    return {
        "metadata": {
            "labels": {
                "name": resource_name,
                "sidecar.istio.io/inject": "false",
            }
        },
        "spec": {
            "containers": [
                {
                    "env": [
                        {
                            "name": "MYSQL_USER",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "key": "database-user",
                                    "name": secret_name,
                                }
                            },
                        },
                        {
                            "name": "MYSQL_PASSWORD",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "key": "database-password",
                                    "name": secret_name,
                                }
                            },
                        },
                        {
                            "name": "MYSQL_ROOT_PASSWORD",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "key": "database-password",
                                    "name": secret_name,
                                }
                            },
                        },
                        {
                            "name": "MYSQL_DATABASE",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "key": "database-name",
                                    "name": secret_name,
                                }
                            },
                        },
                    ],
                    "args": [
                        "--datadir",
                        "/var/lib/mysql/datadir",
                        "--default-authentication-plugin=mysql_native_password",
                    ],
                    "image": MR_DB_IMAGE_DIGEST,
                    "imagePullPolicy": "IfNotPresent",
                    "livenessProbe": {
                        "exec": {
                            "command": [
                                "/bin/bash",
                                "-c",
                                "mysqladmin -u${MYSQL_USER} -p${MYSQL_ROOT_PASSWORD} ping",
                            ]
                        },
                        "initialDelaySeconds": 15,
                        "periodSeconds": 10,
                        "timeoutSeconds": 5,
                    },
                    "name": "mysql",
                    "ports": [{"containerPort": 3306, "protocol": "TCP"}],
                    "readinessProbe": {
                        "exec": {
                            "command": [
                                "/bin/bash",
                                "-c",
                                'mysql -D ${MYSQL_DATABASE} -u${MYSQL_USER} -p${MYSQL_ROOT_PASSWORD} -e "SELECT 1"',
                            ]
                        },
                        "initialDelaySeconds": 10,
                        "timeoutSeconds": 5,
                    },
                    "securityContext": {"capabilities": {}, "privileged": False},
                    "terminationMessagePath": "/dev/termination-log",
                    "volumeMounts": [
                        {
                            "mountPath": "/var/lib/mysql",
                            "name": f"{resource_name}-data",
                        }
                    ],
                }
            ],
            "dnsPolicy": "ClusterFirst",
            "restartPolicy": "Always",
            "volumes": [
                {
                    "name": f"{resource_name}-data",
                    "persistentVolumeClaim": {"claimName": resource_name},
                }
            ],
        },
    }


def get_model_registry_db_label_dict(db_resource_name: str) -> dict[str, str]:
    return {
        Annotations.KubernetesIo.NAME: db_resource_name,
        Annotations.KubernetesIo.INSTANCE: db_resource_name,
        Annotations.KubernetesIo.PART_OF: db_resource_name,
    }
