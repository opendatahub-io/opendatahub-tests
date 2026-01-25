import pytest
from pytest_testconfig import config as py_config

from tests.model_registry.constants import (
    MODEL_REGISTRY_DB_SECRET_STR_DATA,
    DB_BASE_RESOURCES_NAME,
    NUM_MR_INSTANCES,
    MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
    MR_INSTANCE_BASE_NAME,
)
from tests.model_registry.utils import (
    get_model_registry_db_label_dict,
    get_model_registry_deployment_template_dict,
    get_mr_standard_labels,
    get_mysql_config,
)

ns_name = py_config["model_registry_namespace"]

resource_names = [f"{DB_BASE_RESOURCES_NAME}{index}" for index in range(0, NUM_MR_INSTANCES)]

db_secret_params = [
    {
        "name": resource_name,
        "namespace": ns_name,
        "string_data": MODEL_REGISTRY_DB_SECRET_STR_DATA,
        "label": get_model_registry_db_label_dict(db_resource_name=resource_name),
        "annotations": MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
    }
    for resource_name in resource_names
]

db_pvc_params = [
    {
        "name": resource_name,
        "namespace": ns_name,
        "accessmodes": "ReadWriteOnce",
        "size": "4Gi",
        "label": get_model_registry_db_label_dict(db_resource_name=resource_name),
    }
    for resource_name in resource_names
]
annotation = {"template.openshift.io/expose-uri": r"mysql://{.spec.clusterIP}:{.spec.ports[?(.name==\mysql\)].port}"}
db_service_params = [
    {
        "name": resource_name,
        "ports": [{"name": "mysql", "port": 3306, "protocol": "TCP", "targetPort": 3306}],
        "namespace": ns_name,
        "selector": {"name": resource_name},
        "label": get_model_registry_db_label_dict(db_resource_name=resource_name),
        "annotations": annotation,
    }
    for resource_name in resource_names
]

db_deployment_params = [
    {
        "name": resource_name,
        "namespace": ns_name,
        "replicas": 1,
        "revision_history_limit": 0,
        "annotations": {
            "template.alpha.openshift.io/wait-for-ready": "true",
        },
        "label": get_model_registry_db_label_dict(db_resource_name=resource_name),
        "selector": {"matchLabels": {"name": resource_name}},
        "strategy": {"type": "Recreate"},
        "template": get_model_registry_deployment_template_dict(
            secret_name=resource_name, resource_name=resource_name, db_backend="mysql"
        ),
        "wait_for_resource": True,
    }
    for resource_name in resource_names
]

model_registry_instance_params = [
    {
        "name": f"{MR_INSTANCE_BASE_NAME}{index}",
        "namespace": ns_name,
        "label": get_mr_standard_labels(resource_name=f"{MR_INSTANCE_BASE_NAME}{index}"),
        "rest": {},
        "mysql": get_mysql_config(base_name=f"{DB_BASE_RESOURCES_NAME}{index}", namespace=ns_name, db_backend="mysql"),
        "wait_for_resource": True,
        "kube_rbac_proxy": {},
    }
    for index in range(0, NUM_MR_INSTANCES)
]

# Add this complete set of parameters as a pytest.param tuple to the list.
MR_MULTIPROJECT_TEST_SCENARIO_PARAMS = [
    pytest.param(
        db_secret_params,
        db_pvc_params,
        db_service_params,
        db_deployment_params,
        model_registry_instance_params,
        id=f"mr-scenario-{len(resource_names)}-instances",  # Unique ID for pytest output
    )
]
