import pytest
from pytest_testconfig import config as py_config
import uuid
from tests.model_registry.constants import MODEL_REGISTRY_DB_SECRET_STR_DATA, MR_INSTANCE_NAME, DB_RESOURCES_NAME

# Configuration: Get from pytest config or use default
NUM_MR_INSTANCES = int(py_config.get("num_mr_instances", 2))

db_names = [f"{DB_RESOURCES_NAME}-{i + 1}-{str(uuid.uuid4())[:8]}" for i in range(NUM_MR_INSTANCES)]

db_secret_params = [{"db_name": db_name} for db_name in db_names]
db_pvc_params = [{"db_name": db_name} for db_name in db_names]
db_service_params = [
    {"db_name": db_name, "ports": [{"name": "mysql", "port": 3306, "protocol": "TCP", "targetPort": 3306}]}
    for db_name in db_names
]
db_deployment_params = [{"db_name": db_name} for db_name in db_names]
model_registry_instance_params = [
    {
        "mr_name": f"{MR_INSTANCE_NAME}-{i + 1}",
        "db_name": db_name,
        "mysql_config": {
            "host": f"{db_name}.{py_config['model_registry_namespace']}.svc.cluster.local",
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
ALL_MR_TEST_SCENARIOS = [
    pytest.param(
        db_secret_params,
        db_pvc_params,
        db_service_params,
        db_deployment_params,
        model_registry_instance_params,
        id=f"mr-scenario-{len(db_names)}-instances",  # Unique ID for pytest output
    )
]
