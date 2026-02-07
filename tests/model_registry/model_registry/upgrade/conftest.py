import pytest
from typing import Any, Generator
from pytest import FixtureRequest
from model_registry.types import RegisteredModel
from kubernetes.dynamic import DynamicClient
from pytest import Config
from model_registry import ModelRegistry as ModelRegistryClient
from class_generator.parsers.explain_parser import ResourceNotFoundError
from tests.model_registry.constants import MR_INSTANCE_BASE_NAME, KUBERBACPROXY_STR
from utilities.constants import Protocols
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from simple_logger.logger import get_logger
from tests.model_registry.utils import (
    wait_for_default_resource_cleanedup,
    get_mr_standard_labels,
    get_mr_service_by_label,
    get_endpoint_from_mr_service,
)
from utilities.general import wait_for_pods_running

LOGGER = get_logger(name=__name__)
MR_DEFAULT_DB_NAME: str = f"{MR_INSTANCE_BASE_NAME}1"


@pytest.fixture(scope="class")
def model_registry_instance_default_db(
    pytestconfig: Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
    model_registry_metadata_db_resources: dict[Any, Any],
    model_registry_namespace: str,
) -> Generator[ModelRegistry, None, None]:
    """
    Create model registry instance specifically with default postgres database.
    """
    if pytestconfig.option.post_upgrade:
        # In post-upgrade, connect to existing instance
        mr_instance = ModelRegistry(
            client=admin_client, name=MR_DEFAULT_DB_NAME, namespace=model_registry_namespace, ensure_exists=True
        )
        yield mr_instance
        mr_instance.delete(wait=True)
        # Clean up default postgres resources
        wait_for_default_resource_cleanedup(admin_client=admin_client, namespace_name=model_registry_namespace)

    else:
        # In pre-upgrade, create new instance with default postgres
        with ModelRegistry(
            client=admin_client,
            name=MR_DEFAULT_DB_NAME,
            namespace=model_registry_namespace,
            label=get_mr_standard_labels(resource_name=MR_DEFAULT_DB_NAME),
            rest={},
            kube_rbac_proxy={},
            mysql=None,
            postgres={"generateDeployment": True},
            wait_for_resource=True,
            teardown=teardown_resources,
        ) as mr_postgres:
            mr_postgres.wait_for_condition(condition="Available", status="True")
            mr_postgres.wait_for_condition(condition=KUBERBACPROXY_STR, status="True")
            wait_for_pods_running(
                admin_client=admin_client, namespace_name=model_registry_namespace, number_of_consecutive_checks=6
            )
            yield mr_postgres


@pytest.fixture(scope="class")
def model_registry_default_db_instance_rest_endpoint(
    admin_client: DynamicClient, model_registry_instance_default_db: ModelRegistry
) -> str:
    """
    Get the REST endpoint(s) for the model registry instance.
    """
    # get all the services:
    mr_service = get_mr_service_by_label(
        client=admin_client,
        namespace_name=model_registry_instance_default_db.namespace,
        mr_instance=model_registry_instance_default_db,
    )

    if not mr_service:
        raise ResourceNotFoundError("No model registry services found")
    return get_endpoint_from_mr_service(svc=mr_service, protocol=Protocols.REST)


@pytest.fixture(scope="class")
def model_registry_client_default_db(
    current_client_token: str,
    model_registry_default_db_instance_rest_endpoint: str,
) -> ModelRegistryClient:
    server, port = model_registry_default_db_instance_rest_endpoint.split(":")
    mr_client = ModelRegistryClient(
        server_address=f"{Protocols.HTTPS}://{server}",
        port=int(port),
        author="opendatahub-test",
        user_token=current_client_token,
        is_secure=False,
    )

    if not mr_client:
        raise ResourceNotFoundError("No model registry clients created")
    return mr_client


@pytest.fixture(scope="class")
def registered_model_default_db(
    request: FixtureRequest, model_registry_client_default_db: ModelRegistryClient
) -> Generator[RegisteredModel, None, None]:
    yield model_registry_client_default_db.register_model(
        name=request.param.get("model_name"),
        uri=request.param.get("model_uri"),
        version=request.param.get("model_version"),
        description=request.param.get("model_description"),
        model_format_name=request.param.get("model_format"),
        model_format_version=request.param.get("model_format_version"),
        storage_key=request.param.get("model_storage_key"),
        storage_path=request.param.get("model_storage_path"),
        metadata=request.param.get("model_metadata"),
    )
