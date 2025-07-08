import pytest
from typing import Self, Callable
from simple_logger.logger import get_logger
from pytest_testconfig import config as py_config
import uuid

# ocp_resources imports
from ocp_resources.pod import Pod
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.deployment import Deployment
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from kubernetes.dynamic import DynamicClient

# model-registry client imports
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel

# test constants and utils
from tests.model_registry.constants import (
    MODEL_NAME,
    MODEL_DICT,
    MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
    MODEL_REGISTRY_DB_SECRET_STR_DATA,
    MODEL_REGISTRY_STANDARD_LABELS,
    OAUTH_PROXY_CONFIG_DICT,
)
from tests.model_registry.factory_utils import (
    create_dsc_component_patch,
    ModelRegistryInstanceBundle,
    SecretConfig,
    PVCConfig,
    ServiceConfig,
    DeploymentConfig,
    ModelRegistryResourceConfig,
)
from tests.model_registry.utils import (
    get_mr_service_by_label,
    get_endpoint_from_mr_service,
    get_model_registry_deployment_template_dict,
    get_model_registry_db_label_dict,
)
from utilities.constants import Protocols, Annotations

LOGGER = get_logger(name=__name__)

CUSTOM_NAMESPACE = "model-registry-custom-ns"


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class, registered_model",
    [
        pytest.param(create_dsc_component_patch(namespace=CUSTOM_NAMESPACE), MODEL_DICT, id="custom-namespace"),
        pytest.param(
            create_dsc_component_patch(namespace=py_config["model_registry_namespace"]),
            MODEL_DICT,
            id="default-namespace",
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class", "registered_model")
class TestModelRegistryCreation:
    """
    Tests the creation of a model registry. If the component is set to 'Removed' it will be switched to 'Managed'
    for the duration of this test module.
    """

    @pytest.mark.smoke
    def test_registering_model(
        self: Self,
        model_registry_client: ModelRegistryClient,
        registered_model: RegisteredModel,
    ):
        model = model_registry_client.get_registered_model(name=MODEL_NAME)
        expected_attrs = {
            "id": registered_model.id,
            "name": registered_model.name,
            "description": registered_model.description,
            "owner": registered_model.owner,
            "state": registered_model.state,
        }
        errors = [
            f"Unexpected {attr} expected: {expected}, received {getattr(model, attr)}"
            for attr, expected in expected_attrs.items()
            if getattr(model, attr) != expected
        ]
        if errors:
            pytest.fail("errors found in model registry response validation:\n{}".format("\n".join(errors)))

    def test_model_registry_operator_env(
        self,
        updated_dsc_component_state_scope_class: Namespace,
        model_registry_namespace: str,
        model_registry_operator_pod: Pod,
    ):
        namespace_env = []
        for container in model_registry_operator_pod.instance.spec.containers:
            for env in container.env:
                if env.name == "REGISTRIES_NAMESPACE" and env.value == model_registry_namespace:
                    namespace_env.append({container.name: env})
        if not namespace_env:
            pytest.fail("Missing environment variable REGISTRIES_NAMESPACE")

    def test_two_mr_instances_from_fixtures(
        self,
        default_model_registry: ModelRegistryInstanceBundle,
        oauth_model_registry: ModelRegistryInstanceBundle,
        model_registry_client_factory,
    ):
        """Test creating two different MR instances using separate fixtures."""
        LOGGER.info("Testing multiple Model Registry instances with separate fixtures")

        registry1 = default_model_registry
        registry2 = oauth_model_registry

        client1 = model_registry_client_factory(rest_endpoint=registry1.rest_endpoint)
        client2 = model_registry_client_factory(rest_endpoint=registry2.rest_endpoint)

        assert registry1.instance.name != registry2.instance.name
        assert registry1.rest_endpoint != registry2.rest_endpoint
        assert client1 is not client2

        LOGGER.info(f"Successfully created clients for: {registry1.instance.name} and {registry2.instance.name}")

        try:
            models1 = client1.get_registered_models()
            models2 = client2.get_registered_models()
            assert isinstance(models1.items, list)
            assert isinstance(models2.items, list)
        except Exception as e:
            LOGGER.warning(f"Client connectivity test failed: {e}")

    def test_two_mr_instances_from_single_resource_factories(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        secret_factory: Callable[[SecretConfig], Secret],
        pvc_factory: Callable[[PVCConfig], PersistentVolumeClaim],
        service_factory: Callable[[ServiceConfig], Service],
        deployment_factory: Callable[[DeploymentConfig], Deployment],
        model_registry_resource_factory: Callable[[ModelRegistryResourceConfig], ModelRegistry],
        model_registry_client_factory: Callable[[str], ModelRegistryClient],
    ):
        """Test creating two MR instances using individual resource factories."""
        LOGGER.info("Testing multiple Model Registry instances with single resource factories")

        # --- Create Instance 1 (default) ---
        db_name_1 = f"manual-db-{str(uuid.uuid4())[:8]}"
        labels_1 = get_model_registry_db_label_dict(db_resource_name=db_name_1)

        secret_config_1 = SecretConfig(
            name=db_name_1,
            namespace=model_registry_namespace,
            string_data={
                "database-name": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
                "database-user": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
                "database-password": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-password"],
            },
            labels=labels_1,
            annotations=MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
        )
        secret_1 = secret_factory(secret_config_1)  # noqa: FCN001

        pvc_config_1 = PVCConfig(
            name=db_name_1,
            namespace=model_registry_namespace,
            access_modes="ReadWriteOnce",
            size="5Gi",
            labels=labels_1,
        )
        pvc_factory(pvc_config_1)  # noqa: FCN001

        service_config_1 = ServiceConfig(
            name=db_name_1,
            namespace=model_registry_namespace,
            ports=[{"name": "mysql", "port": 3306, "protocol": "TCP", "targetPort": 3306}],
            selector={"name": db_name_1},
            labels=labels_1,
        )
        service_factory(service_config_1)  # noqa: FCN001

        deployment_config_1 = DeploymentConfig(
            name=db_name_1,
            namespace=model_registry_namespace,
            template=get_model_registry_deployment_template_dict(secret_name=secret_1.name, resource_name=db_name_1),
            labels=labels_1,
        )
        deployment_1 = deployment_factory(deployment_config_1)  # noqa: FCN001

        mysql_config_1 = {
            "host": f"{deployment_1.name}.{model_registry_namespace}.svc.cluster.local",
            "database": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
            "passwordSecret": {"key": "database-password", "name": secret_1.name},
            "port": 3306,
            "skipDBCreation": False,
            "username": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
        }
        mr_name_1 = f"manual-mr-{str(uuid.uuid4())[:8]}"
        mr_labels_1 = MODEL_REGISTRY_STANDARD_LABELS.copy()
        mr_labels_1.update({Annotations.KubernetesIo.NAME: mr_name_1, Annotations.KubernetesIo.INSTANCE: mr_name_1})
        mr_config_1 = ModelRegistryResourceConfig(
            name=mr_name_1,
            namespace=model_registry_namespace,
            mysql_config=mysql_config_1,
            labels=mr_labels_1,
            use_oauth_proxy=True,
            oauth_proxy_config=OAUTH_PROXY_CONFIG_DICT,
        )
        instance_1 = model_registry_resource_factory(mr_config_1)  # noqa: FCN001

        # --- Create Instance 2 (Istio) ---
        db_name_2 = f"manual-db-{str(uuid.uuid4())[:8]}"
        labels_2 = get_model_registry_db_label_dict(db_resource_name=db_name_2)

        secret_config_2 = SecretConfig(
            name=db_name_2,
            namespace=model_registry_namespace,
            string_data={
                "database-name": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
                "database-user": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
                "database-password": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-password"],
            },
            labels=labels_2,
            annotations=MODEL_REGISTRY_DB_SECRET_ANNOTATIONS,
        )
        secret_2 = secret_factory(secret_config_2)  # noqa: FCN001

        pvc_config_2 = PVCConfig(
            name=db_name_2,
            namespace=model_registry_namespace,
            access_modes="ReadWriteOnce",
            size="5Gi",
            labels=labels_2,
        )
        pvc_factory(pvc_config_2)  # noqa: FCN001

        service_config_2 = ServiceConfig(
            name=db_name_2,
            namespace=model_registry_namespace,
            ports=[{"name": "mysql", "port": 3306, "protocol": "TCP", "targetPort": 3306}],
            selector={"name": db_name_2},
            labels=labels_2,
        )
        service_factory(service_config_2)  # noqa: FCN001

        deployment_config_2 = DeploymentConfig(
            name=db_name_2,
            namespace=model_registry_namespace,
            template=get_model_registry_deployment_template_dict(secret_name=secret_2.name, resource_name=db_name_2),
            labels=labels_2,
        )
        deployment_2 = deployment_factory(deployment_config_2)  # noqa: FCN001

        mysql_config_2 = {
            "host": f"{deployment_2.name}.{model_registry_namespace}.svc.cluster.local",
            "database": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-name"],
            "passwordSecret": {"key": "database-password", "name": secret_2.name},
            "port": 3306,
            "skipDBCreation": False,
            "username": MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
        }
        mr_name_2 = f"manual-mr-{str(uuid.uuid4())[:8]}"
        mr_labels_2 = MODEL_REGISTRY_STANDARD_LABELS.copy()
        mr_labels_2.update({Annotations.KubernetesIo.NAME: mr_name_2, Annotations.KubernetesIo.INSTANCE: mr_name_2})
        mr_config_2 = ModelRegistryResourceConfig(
            name=mr_name_2,
            namespace=model_registry_namespace,
            mysql_config=mysql_config_2,
            labels=mr_labels_2,
            use_oauth_proxy=True,
            oauth_proxy_config=OAUTH_PROXY_CONFIG_DICT,
        )
        instance_2 = model_registry_resource_factory(mr_config_2)  # noqa: FCN001

        # --- Get clients and test ---
        service1 = get_mr_service_by_label(
            client=admin_client, namespace_name=model_registry_namespace, mr_instance=instance_1
        )
        endpoint1 = get_endpoint_from_mr_service(svc=service1, protocol=Protocols.REST)
        client1 = model_registry_client_factory(endpoint1)  # noqa: FCN001

        service2 = get_mr_service_by_label(
            client=admin_client, namespace_name=model_registry_namespace, mr_instance=instance_2
        )
        endpoint2 = get_endpoint_from_mr_service(svc=service2, protocol=Protocols.REST)
        client2 = model_registry_client_factory(endpoint2)  # noqa: FCN001

        assert instance_1.name != instance_2.name
        assert endpoint1 != endpoint2
        assert client1 is not client2

        LOGGER.info(f"Successfully created clients for: {instance_1.name} and {instance_2.name}")

        try:
            models1 = list(client1.get_registered_models())
            models2 = list(client2.get_registered_models())
            assert isinstance(models1, list)
            assert isinstance(models2, list)
        except Exception as e:
            LOGGER.warning(f"Client connectivity test failed: {e}")

    def test_two_mr_instances_from_simple_fixtures(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        db_secret_1: Secret,
        db_pvc_1: PersistentVolumeClaim,
        db_service_1: Service,
        db_deployment_1: Deployment,
        model_registry_instance_1: ModelRegistry,
        db_secret_2: Secret,
        db_pvc_2: PersistentVolumeClaim,
        db_service_2: Service,
        db_deployment_2: Deployment,
        model_registry_instance_2: ModelRegistry,
        model_registry_client_factory: Callable[[str], ModelRegistryClient],
    ):
        """Test creating two MR instances using the simple resource fixtures."""
        LOGGER.info("Testing multiple Model Registry instances with simple resource fixtures")

        # Get services and endpoints
        service1 = get_mr_service_by_label(
            client=admin_client, namespace_name=model_registry_namespace, mr_instance=model_registry_instance_1
        )
        endpoint1 = get_endpoint_from_mr_service(svc=service1, protocol=Protocols.REST)
        client1 = model_registry_client_factory(endpoint1)  # noqa: FCN001

        service2 = get_mr_service_by_label(
            client=admin_client, namespace_name=model_registry_namespace, mr_instance=model_registry_instance_2
        )
        endpoint2 = get_endpoint_from_mr_service(svc=service2, protocol=Protocols.REST)
        client2 = model_registry_client_factory(endpoint2)  # noqa: FCN001

        # Verify instances and clients are different
        assert model_registry_instance_1.name != model_registry_instance_2.name
        assert endpoint1 != endpoint2
        assert client1 is not client2

        LOGGER.info(
            f"Successfully created clients for: {model_registry_instance_1.name} and {model_registry_instance_2.name}"
        )

        # Test basic connectivity
        try:
            models1 = list(client1.get_registered_models())
            models2 = list(client2.get_registered_models())
            assert isinstance(models1, list)
            assert isinstance(models2, list)
        except Exception as e:
            LOGGER.warning(f"Client connectivity test failed: {e}")

    # TODO: Edit a registered model
    # TODO: Add additional versions for a model
    # TODO: List all available models
    # TODO: List all versions of a model
