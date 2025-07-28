import pytest
import uuid
from tests.model_registry.constants import (
    MODEL_REGISTRY_DB_SECRET_STR_DATA,
    MR_INSTANCE_NAME,
    DB_RESOURCES_NAME,
    NUM_MR_INSTANCES,
)


ns_name = f"{MR_INSTANCE_NAME}-ns-{str(uuid.uuid4())[:8]}"
ns_params = {"ns_name": ns_name}

db_names = [f"{DB_RESOURCES_NAME}-{i + 1}-{str(uuid.uuid4())[:8]}" for i in range(NUM_MR_INSTANCES)]

db_secret_params = [{"db_name": db_name, "ns_name": ns_name} for db_name in db_names]

db_pvc_params = [
    {
        "db_name": db_name,
        "ns_name": ns_name,
        "accessmodes": "ReadWriteOnce",
        "size": "4Gi",
    }
    for db_name in db_names
]

db_service_params = [
    {
        "db_name": db_name,
        "ports": [{"name": "mysql", "port": 3306, "protocol": "TCP", "targetPort": 3306}],
        "ns_name": ns_name,
    }
    for db_name in db_names
]

db_deployment_params = [
    {
        "db_name": db_name,
        "ns_name": ns_name,
        "replicas": 1,
        "revision_history_limit": 0,
    }
    for db_name in db_names
]

model_registry_instance_params = [
    {
        "is_model_registry_oauth": True,
        "mr_name": f"{MR_INSTANCE_NAME}-{i + 1}",
        "db_name": db_name,
        "ns_name": ns_name,
        "mysql_config": {
            "host": f"{db_name}.{ns_name}.svc.cluster.local",
            "database": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
            "passwordSecret": {"key": "database-password", "name": db_name},
            "port": 3306,
            "skipDBCreation": False,
            "username": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
        },
    }
    for i, db_name in enumerate(db_names)
]

# Add this complete set of parameters as a pytest.param tuple to the list.
MR_MULTIPROJECT_TEST_SCENARIO_PARAMS = [
    pytest.param(
        ns_params,  # updated_dsc_component_state_parametrized (expects dict)
        db_secret_params,
        db_pvc_params,
        db_service_params,
        db_deployment_params,
        model_registry_instance_params,
        id=f"mr-scenario-{len(db_names)}-instances",  # Unique ID for pytest output
    )
]
