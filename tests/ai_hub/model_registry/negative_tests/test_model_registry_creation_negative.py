from typing import Self

import pytest
import structlog
from pytest_testconfig import config as py_config

from tests.ai_hub.constants import (
    DB_RESOURCE_NAME,
    MR_INSTANCE_NAME,
    MR_OPERATOR_NAME,
)
from utilities.constants import Annotations
from utilities.openshift_resources.data_science_cluster import DataScienceCluster
from utilities.openshift_resources.deployment import Deployment
from utilities.openshift_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from utilities.openshift_resources.namespace import Namespace
from utilities.openshift_resources.oc import ForbiddenError
from utilities.openshift_resources.secret import Secret

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "model_registry_namespace_for_negative_tests",
    "updated_dsc_component_state_scope_session",
    "model_registry_db_secret_negative_test",
    "model_registry_db_deployment_negative_test",
)
@pytest.mark.custom_namespace
class TestModelRegistryCreationNegative:
    @pytest.mark.tier3
    def test_registering_model_negative(
        self: Self,
        current_client_token: str,
        model_registry_namespace_for_negative_tests: Namespace,
        updated_dsc_component_state_scope_session: DataScienceCluster,
        model_registry_db_secret_negative_test: Secret,
        model_registry_db_deployment_negative_test: Deployment,
    ):
        my_sql_dict: dict[str, str] = {
            "host": f"{model_registry_db_deployment_negative_test.name}."
            f"{model_registry_db_deployment_negative_test.namespace}.svc.cluster.local",
            "database": model_registry_db_secret_negative_test.string_data["database-name"],
            "passwordSecret": {"key": "database-password", "name": DB_RESOURCE_NAME},
            "port": 3306,
            "skipDBCreation": False,
            "username": model_registry_db_secret_negative_test.string_data["database-user"],
        }
        with pytest.raises(  # noqa: SIM117
            ForbiddenError,  # UnprocessibleEntityError
            match=f"namespace must be {py_config['model_registry_namespace']}",
        ):
            with ModelRegistry(
                name=MR_INSTANCE_NAME,
                namespace=model_registry_namespace_for_negative_tests.name,
                label={
                    Annotations.KubernetesIo.NAME: MR_INSTANCE_NAME,
                    Annotations.KubernetesIo.INSTANCE: MR_INSTANCE_NAME,
                    Annotations.KubernetesIo.PART_OF: MR_OPERATOR_NAME,
                    Annotations.KubernetesIo.CREATED_BY: MR_OPERATOR_NAME,
                },
                rest={},
                kube_rbac_proxy={},
                mysql=my_sql_dict,
                wait_for_resource=True,
            ):
                return
