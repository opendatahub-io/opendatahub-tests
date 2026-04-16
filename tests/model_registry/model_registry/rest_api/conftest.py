import base64
import copy
import os
import tempfile
from collections.abc import Generator
from typing import Any

import portforward
import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_registry.constants import (
    CA_CONFIGMAP_NAME,
    CA_FILE_PATH,
    CA_MOUNT_PATH,
    DB_RESOURCE_NAME,
    KUBERBACPROXY_STR,
    SECURE_MR_NAME,
)
from tests.model_registry.model_registry.rest_api.constants import MODEL_REGISTER_DATA, MODEL_REGISTRY_BASE_URI
from tests.model_registry.model_registry.rest_api.utils import (
    execute_model_registry_patch_command,
    generate_ca_and_server_cert,
    get_mr_deployment,
    register_model_rest_api,
)
from tests.model_registry.utils import (
    add_db_certs_volumes_to_deployment,
    apply_db_args_and_volume_mounts,
    get_external_db_config,
    get_model_registry_deployment_template_dict,
    get_mr_standard_labels,
)
from utilities.certificates_utils import create_ca_bundle_with_router_cert, create_k8s_secret
from utilities.exceptions import MissingParameter
from utilities.general import generate_random_name, wait_for_pods_running
from utilities.infra import create_ns
from utilities.operator_utils import get_cluster_service_version
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry

LOGGER = structlog.get_logger(name=__name__)


POSTGRES_FILE_PATH: str = "/etc/server-cert"


@pytest.fixture(scope="class")
def registered_model_rest_api(
    request: pytest.FixtureRequest,
    model_registry_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> dict[str, Any]:
    return register_model_rest_api(
        model_registry_rest_url=model_registry_rest_url[0],
        model_registry_rest_headers=model_registry_rest_headers,
        data_dict=request.param,
    )


@pytest.fixture()
def updated_model_registry_resource(
    request: pytest.FixtureRequest,
    model_registry_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    registered_model_rest_api: dict[str, Any],
) -> dict[str, Any]:
    """
    Generic fixture to update any model registry resource via PATCH request.

    Expects request.param to contain:
        - resource_name: Key to identify the resource in registered_model_rest_api
        - api_name: API endpoint name for the resource type
        - data: JSON data to send in the PATCH request

    Returns:
       Dictionary containing the updated resource data
    """
    resource_name = request.param.get("resource_name")
    api_name = request.param.get("api_name")
    if not (api_name and resource_name):
        raise MissingParameter("resource_name and api_name are required parameters for this fixture.")
    resource_id = registered_model_rest_api[resource_name]["id"]
    assert resource_id, f"Resource id not found: {registered_model_rest_api[resource_name]}"
    return execute_model_registry_patch_command(
        url=f"{model_registry_rest_url[0]}{MODEL_REGISTRY_BASE_URI}{api_name}/{resource_id}",
        headers=model_registry_rest_headers,
        data_json=request.param["data"],
    )


@pytest.fixture(scope="class")
def patch_invalid_ca(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    request: pytest.FixtureRequest,
) -> Generator[str, Any, Any]:
    """
    Patches the ConfigMap with an invalid CA certificate.
    """
    ca_configmap_name = request.param.get("ca_configmap_name", "odh-trusted-ca-bundle")
    ca_file_name = request.param.get("ca_file_name", "invalid-ca.crt")
    ca_file_path = f"{CA_MOUNT_PATH}/{ca_file_name}"
    LOGGER.info(f"Patching the {ca_configmap_name} ConfigMap with an invalid CA certificate: {ca_file_path}")
    ca_data = {ca_file_name: "-----BEGIN CERTIFICATE-----\nINVALIDCERTIFICATE\n-----END CERTIFICATE-----"}
    ca_configmap = ConfigMap(
        client=admin_client,
        name=ca_configmap_name,
        namespace=model_registry_namespace,
        ensure_exists=True,
    )
    patch = {
        "metadata": {
            "name": ca_configmap_name,
            "namespace": model_registry_namespace,
        },
        "data": ca_data,
    }
    with ResourceEditor(patches={ca_configmap: patch}):
        LOGGER.info(f"Patched the {ca_configmap_name} ConfigMap with an invalid CA certificate: {ca_file_path}")
        yield ca_file_path


@pytest.fixture(scope="class")
def db_backend_under_test(request: pytest.FixtureRequest) -> str:
    """
    Fixture to provide the database backend type being tested.

    Args:
        request: The pytest request object containing test parameters

    Returns:
        str: The database backend type (e.g., "mysql", "postgres")
    """
    return getattr(request, "param", "mysql")


@pytest.fixture(scope="class")
def external_db_template_with_ca(
    model_registry_metadata_db_resources: dict[Any, Any], db_backend_under_test: str
) -> dict[str, Any]:
    """
    Patches the external database template with the CA file path and volume mount.
    """
    db_deployment_template = get_model_registry_deployment_template_dict(
        secret_name=model_registry_metadata_db_resources[Secret][0].name,
        resource_name=DB_RESOURCE_NAME,
        db_backend=db_backend_under_test,
    )

    if db_backend_under_test == "mysql":
        db_deployment_template["spec"]["containers"][0]["args"].append(f"--ssl-ca={CA_FILE_PATH}")

    elif db_backend_under_test == "postgres":
        postgres_ssl_args = [
            "postgres",
            "-c",
            "ssl=on",
            "-c",
            f"ssl_cert_file={POSTGRES_FILE_PATH}/tls.crt",
            "-c",
            f"ssl_key_file={POSTGRES_FILE_PATH}/tls.key",
            "-c",
            f"ssl_ca_file={POSTGRES_FILE_PATH}/ca.crt",
        ]
        db_deployment_template["spec"]["containers"][0].get("args", []).extend(postgres_ssl_args)
    db_deployment_template["spec"]["containers"][0]["volumeMounts"].append({
        "mountPath": CA_MOUNT_PATH,
        "name": CA_CONFIGMAP_NAME,
        "readOnly": True,
    })
    db_deployment_template["spec"]["volumes"].append({
        "name": CA_CONFIGMAP_NAME,
        "configMap": {"name": CA_CONFIGMAP_NAME},
    })
    return db_deployment_template


@pytest.fixture(scope="class")
def deploy_secure_db_mr(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    external_db_template_with_ca: dict[str, Any],
    patch_external_deployment_with_ssl_ca: Deployment,
    db_backend_under_test: str,
) -> Generator[ModelRegistry]:
    """
    Deploy a secure database and Model Registry instance.
    """
    param = getattr(request, "param", {})
    db_config = get_external_db_config(
        base_name=DB_RESOURCE_NAME, namespace=model_registry_namespace, db_backend=db_backend_under_test
    )
    if "sslRootCertificateConfigMap" in param:
        db_config["sslRootCertificateConfigMap"] = param["sslRootCertificateConfigMap"]
    with ModelRegistry(
        client=admin_client,
        name=SECURE_MR_NAME,
        namespace=model_registry_namespace,
        label=get_mr_standard_labels(resource_name=SECURE_MR_NAME),
        rest={},
        kube_rbac_proxy={},
        mysql=db_config if db_backend_under_test == "mysql" else None,
        postgres=db_config if db_backend_under_test == "postgres" else None,
        wait_for_resource=True,
    ) as mr:
        mr.wait_for_condition(condition="Available", status="True")
        mr.wait_for_condition(condition=KUBERBACPROXY_STR, status="True")
        wait_for_pods_running(
            admin_client=admin_client, namespace_name=model_registry_namespace, number_of_consecutive_checks=6
        )
        yield mr


@pytest.fixture()
def local_ca_bundle(request: pytest.FixtureRequest, admin_client: DynamicClient) -> Generator[str, Any, Any]:
    """
    Creates a local CA bundle file by fetching the CA bundle from a ConfigMap and appending the router CA from a Secret.
    Args:
        request: The pytest request object
        admin_client: The admin client to get the CA bundle from a ConfigMap and append the router CA from a Secret.
    Returns:
        Generator[str, Any, Any]: A generator that yields the CA bundle path.
    """
    ca_bundle_path = getattr(request, "param", {}).get("cert_name", "ca-bundle.crt")
    create_ca_bundle_with_router_cert(
        client=admin_client,
        namespace=py_config["model_registry_namespace"],
        ca_bundle_path=ca_bundle_path,
        cert_name=ca_bundle_path,
    )
    yield ca_bundle_path

    os.remove(ca_bundle_path)


@pytest.fixture(scope="class")
def ca_configmap_for_test(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    external_db_ssl_artifact_paths: dict[str, Any],
) -> Generator[ConfigMap]:
    """
    Creates a test-specific ConfigMap for the CA bundle, using the generated CA cert.

    Args:
        admin_client: The admin client to create the ConfigMap
        model_registry_namespace: The namespace of the model registry
        external_db_ssl_artifact_paths: The artifacts and secrets for the external database SSL connection

    Returns:
        Generator[ConfigMap, None, None]: A generator that yields the ConfigMap instance.
    """
    with open(external_db_ssl_artifact_paths["ca_crt"], "r") as f:
        ca_content = f.read()
    if not ca_content:
        LOGGER.info("CA content is empty")
        raise MissingParameter("CA content is empty")
    cm_name = "db-ca-configmap"
    with ConfigMap(
        client=admin_client,
        name=cm_name,
        namespace=model_registry_namespace,
        data={"ca-bundle.crt": ca_content},
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def patch_external_deployment_with_ssl_ca(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    external_db_ssl_secrets: dict[str, Any],
    db_backend_under_test: str,
) -> Generator[Deployment, Any, Any]:
    """
    Patch the external database deployment to use the test CA bundle,
    and mount the server cert/key for SSL.
    """
    model_registry_db_deployments = get_mr_deployment(admin_client=admin_client, mr_namespace=model_registry_namespace)
    if request.param.get("ca_configmap_for_test"):
        LOGGER.info("Invoking ca_configmap_for_test fixture")
        request.getfixturevalue(argname="ca_configmap_for_test")
    ca_configmap_name = request.param.get("ca_configmap_name", "db-ca-configmap")
    if db_backend_under_test == "mysql":
        ca_mount_path = request.param.get("ca_mount_path", "/etc/mysql/ssl")
    elif db_backend_under_test == "postgres":
        ca_mount_path = request.param.get("ca_mount_path", "/etc")

    deployment = model_registry_db_deployments[0].instance.to_dict()
    spec = deployment["spec"]["template"]["spec"]
    db_containers = [container for container in spec["containers"] if container["name"] == db_backend_under_test]
    assert db_containers, f"{db_backend_under_test} container not found"

    db_container = apply_db_args_and_volume_mounts(
        db_container=db_containers[0],
        ca_configmap_name=ca_configmap_name,
        ca_mount_path=ca_mount_path,
        db_backend=db_backend_under_test,
    )
    volumes = add_db_certs_volumes_to_deployment(
        spec=spec, ca_configmap_name=ca_configmap_name, db_backend=db_backend_under_test
    )

    patch = {"spec": {"template": {"spec": {"volumes": volumes, "containers": [db_container]}}}}
    with ResourceEditor(patches={model_registry_db_deployments[0]: patch}):
        wait_for_pods_running(
            admin_client=admin_client, namespace_name=model_registry_namespace, number_of_consecutive_checks=3
        )
        model_registry_db_deployments[0].wait_for_condition(condition="Available", status="True")
        yield model_registry_db_deployments[0]


@pytest.fixture(scope="class")
def external_db_ssl_artifact_paths() -> Generator[dict[str, str]]:
    """
    Generates external database SSL certificate and key files in a temporary directory
    and provides their paths.

    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield generate_ca_and_server_cert(tmp_dir=tmp_dir)


@pytest.fixture(scope="class")
def external_db_ssl_secrets(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    external_db_ssl_artifact_paths: dict[str, str],
) -> Generator[dict[str, Secret]]:
    """
    Creates Kubernetes secrets for external database SSL artifacts.
    """
    ca_secret = create_k8s_secret(
        client=admin_client,
        namespace=model_registry_namespace,
        name="db-ca",
        file_path=external_db_ssl_artifact_paths["ca_crt"],
        key_name="ca.crt",
    )
    server_cert_secret = create_k8s_secret(
        client=admin_client,
        namespace=model_registry_namespace,
        name="db-server-cert",
        file_path=external_db_ssl_artifact_paths["server_crt"],
        key_name="tls.crt",
    )
    server_key_secret = create_k8s_secret(
        client=admin_client,
        namespace=model_registry_namespace,
        name="db-server-key",
        file_path=external_db_ssl_artifact_paths["server_key"],
        key_name="tls.key",
    )

    yield {
        "ca_secret": ca_secret,
        "server_cert_secret": server_cert_secret,
        "server_key_secret": server_key_secret,
    }
    if ca_secret.exists:
        ca_secret.delete(wait=True)
    if server_cert_secret.exists:
        server_cert_secret.delete(wait=True)
    if server_key_secret.exists:
        server_key_secret.delete(wait=True)


@pytest.fixture(scope="function")
def model_data_for_test() -> Generator[dict[str, Any]]:
    """
    Generates a model data for the test.

    Returns:
        dict[str, Any]: The model data for the test
    """
    model_name = generate_random_name(prefix="model-rest-api")
    model_data = copy.deepcopy(MODEL_REGISTER_DATA)
    model_data["register_model_data"]["name"] = model_name
    yield model_data


@pytest.fixture()
def model_registry_default_postgres_deployment_match_label(
    model_registry_namespace: str, admin_client: DynamicClient, model_registry_instance: list[ModelRegistry]
) -> dict[str, str]:
    """
    Returns the matchLabels from the default postgres deployment for filtering pods.
    """
    deployment = Deployment(
        client=admin_client,
        namespace=model_registry_namespace,
        name=f"{model_registry_instance[0].name}-postgres",
        ensure_exists=True,
    )
    return deployment.instance.spec.selector.matchLabels


# Model Registry Deployment Fixtures (similar to HuggingFace deployment fixtures)


class ModelRegistryDeploymentError(Exception):
    """Exception raised when model registry deployment fails."""


class PredictorPodNotFoundError(Exception):
    """Exception raised when predictor pods are not found for an InferenceService."""


def get_openvino_image_from_rhoai_csv(admin_client: DynamicClient) -> str:
    """
    Get the OpenVINO model server image from the RHOAI ClusterServiceVersion.

    Returns:
        str: The OpenVINO model server image URL from RHOAI CSV

    Raises:
        Exception: If unable to find the image in the CSV
    """
    # Get the RHOAI CSV using the utility function
    csv = get_cluster_service_version(
        client=admin_client, prefix="rhods-operator", namespace=py_config["applications_namespace"]
    )

    # Look for OpenVINO image in spec.relatedImages
    related_images = csv.instance.spec.get("relatedImages", [])

    for image_info in related_images:
        image_url = image_info.get("image", "")
        if "odh-openvino-model-server" in image_url:
            LOGGER.info(f"Found OpenVINO image from RHOAI CSV: {image_url}")
            return image_url

    raise ModelRegistryDeploymentError("Could not find odh-openvino-model-server image in RHOAI CSV relatedImages")


@pytest.fixture(scope="class")
def model_registry_deployment_ns(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    """
    Create a dedicated namespace for Model Registry model deployments and testing.
    Similar to hugging_face_deployment_ns but specifically for registered model serving tests.
    """
    with create_ns(
        name="mr-deployment-ns",
        admin_client=admin_client,
    ) as ns:
        LOGGER.info(f"Created Model Registry deployment namespace: {ns.name}")
        yield ns


@pytest.fixture(scope="class")
def model_registry_connection_secret(
    admin_client: DynamicClient,
    model_registry_deployment_ns: Namespace,
    registered_model_rest_api: dict[str, Any],
) -> Generator[Secret, Any, Any]:
    """
    Create a connection secret for the registered model URI.
    This secret is required by the ODH admission webhook when creating InferenceServices
    with the opendatahub.io/connections annotation.
    """
    resource_name = "mr-test-inference-service-connection"
    # Use the model URI from the registered model
    register_model_data = registered_model_rest_api.get("register_model", {})
    model_uri = register_model_data.get("external_id", "hf://jonburdo/test2")

    # Base64 encode the model URI
    encoded_uri = base64.b64encode(model_uri.encode()).decode()

    # Annotations matching the connection secret structure
    annotations = {
        "opendatahub.io/connection-type-protocol": "uri",
        "opendatahub.io/connection-type-ref": "uri-v1",
        "openshift.io/display-name": resource_name,
    }

    # Labels for ODH integration
    labels = {
        "opendatahub.io/dashboard": "false",
    }

    with Secret(
        client=admin_client,
        name=resource_name,
        namespace=model_registry_deployment_ns.name,
        annotations=annotations,
        label=labels,
        data_dict={"URI": encoded_uri},
        teardown=True,
    ) as connection_secret:
        LOGGER.info(
            f"Created Model Registry connection secret: {resource_name} in "
            f"namespace: {model_registry_deployment_ns.name}"
        )
        yield connection_secret


@pytest.fixture(scope="class")
def model_registry_serving_runtime(
    admin_client: DynamicClient,
    model_registry_deployment_ns: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    """
    Create a ServingRuntime for OpenVINO Model Server to support registered models.
    Based on the HuggingFace serving runtime with complete ODH dashboard integration.
    """
    runtime_name = "mr-test-runtime"

    # Complete annotations matching manually created examples
    annotations = {
        "opendatahub.io/apiProtocol": "REST",
        "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
        "opendatahub.io/runtime-version": "v2025.4",
        "opendatahub.io/serving-runtime-scope": "global",
        "opendatahub.io/template-display-name": "OpenVINO Model Server",
        "opendatahub.io/template-name": "kserve-ovms",
        "openshift.io/display-name": "OpenVINO Model Server",
    }

    # Labels for ODH dashboard integration
    labels = {
        "opendatahub.io/dashboard": "true",
    }

    # Supported model formats
    supported_model_formats = [
        {"autoSelect": True, "name": "openvino_ir", "version": "opset13"},
        {"name": "onnx", "version": "1"},
        {"autoSelect": True, "name": "tensorflow", "version": "1"},
        {"autoSelect": True, "name": "tensorflow", "version": "2"},
        {"autoSelect": True, "name": "paddle", "version": "2"},
        {"autoSelect": True, "name": "pytorch", "version": "2"},
    ]

    # Complete ServingRuntime specification
    runtime_spec = {
        "annotations": {
            "opendatahub.io/kserve-runtime": "ovms",
            "prometheus.io/path": "/metrics",
            "prometheus.io/port": "8888",
        },
        "containers": [
            {
                "args": [
                    "--model_name={{.Name}}",
                    "--port=8001",
                    "--rest_port=8888",
                    "--model_path=/mnt/models",
                    "--file_system_poll_wait_seconds=0",
                    "--metrics_enable",
                ],
                "image": get_openvino_image_from_rhoai_csv(admin_client),
                "name": "kserve-container",
                "ports": [{"containerPort": 8888, "protocol": "TCP"}],
            }
        ],
        "multiModel": False,
        "protocolVersions": ["v2", "grpc-v2"],
        "supportedModelFormats": supported_model_formats,
    }

    # Create the ServingRuntime with complete configuration
    runtime_dict = {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": runtime_name,
            "namespace": model_registry_deployment_ns.name,
            "annotations": annotations,
            "labels": labels,
        },
        "spec": runtime_spec,
    }

    with ServingRuntime(
        client=admin_client,
        kind_dict=runtime_dict,
        teardown=True,
    ) as serving_runtime:
        LOGGER.info(
            f"Created OpenVINO ServingRuntime: {runtime_name} in namespace: {model_registry_deployment_ns.name}"
        )
        yield serving_runtime


@pytest.fixture(scope="class")
def model_registry_inference_service(
    admin_client: DynamicClient,
    model_registry_deployment_ns: Namespace,
    model_registry_serving_runtime: ServingRuntime,
    model_registry_connection_secret: Secret,
    registered_model_rest_api: dict[str, Any],
) -> Generator[InferenceService, Any, Any]:
    """
    Create an InferenceService for testing registered models.
    Based on the HuggingFace InferenceService with comprehensive ODH dashboard integration.
    """
    name = "mr-test-inference-service"
    # Use the model URI from the registered model
    register_model_data = registered_model_rest_api.get("register_model", {})
    model_uri = register_model_data.get("external_id", "hf://jonburdo/test2")
    model_name = register_model_data.get("name", "my-model")
    runtime_name = model_registry_serving_runtime.name

    # Resources
    resources = {"limits": {"cpu": "2", "memory": "4Gi"}, "requests": {"cpu": "2", "memory": "4Gi"}}

    # Labels for ODH dashboard integration
    labels = {
        "opendatahub.io/dashboard": "true",
    }

    # Comprehensive annotations matching ODH integration
    annotations = {
        "opendatahub.io/connections": model_registry_connection_secret.name,
        "opendatahub.io/hardware-profile-name": "default-profile",
        "opendatahub.io/hardware-profile-namespace": "redhat-ods-applications",
        "opendatahub.io/model-type": "predictive",
        "openshift.io/description": f"Model from registry: {model_name}",
        "openshift.io/display-name": f"registry/{name}",
        "security.opendatahub.io/enable-auth": "false",
        "serving.kserve.io/deploymentMode": "RawDeployment",
    }

    # Predictor configuration
    predictor_dict = {
        "automountServiceAccountToken": False,
        "deploymentStrategy": {"type": "RollingUpdate"},
        "maxReplicas": 1,
        "minReplicas": 1,
        "model": {
            "modelFormat": {"name": "onnx", "version": "1"},
            "name": "",
            "resources": resources,
            "runtime": runtime_name,
            "storageUri": model_uri,
        },
    }

    with InferenceService(
        client=admin_client,
        name=name,
        namespace=model_registry_deployment_ns.name,
        annotations=annotations,
        label=labels,
        predictor=predictor_dict,
        teardown=True,
    ) as inference_service:
        # Wait for InferenceService to become Ready
        inference_service.wait_for_condition(
            condition="Ready",
            status="True",
            timeout=600,  # 10 minutes timeout for model loading
        )
        LOGGER.info(
            f"Created Model Registry InferenceService: {name} in namespace: {model_registry_deployment_ns.name}"
        )
        yield inference_service


@pytest.fixture(scope="class")
def model_registry_predictor_pod(
    admin_client: DynamicClient,
    model_registry_deployment_ns: Namespace,
    model_registry_inference_service: InferenceService,
) -> Pod:
    """
    Get the predictor pod for the Model Registry InferenceService.
    """
    namespace = model_registry_deployment_ns.name
    label_selector = f"serving.kserve.io/inferenceservice={model_registry_inference_service.name}"

    pods = Pod.get(
        client=admin_client,
        namespace=namespace,
        label_selector=label_selector,
    )

    predictor_pods = [pod for pod in pods if "predictor" in pod.name]
    if not predictor_pods:
        raise PredictorPodNotFoundError(
            f"No predictor pods found for InferenceService {model_registry_inference_service.name}"
        )

    pod = predictor_pods[0]  # Use the first predictor pod
    LOGGER.info(f"Found predictor pod: {pod.name} in namespace: {namespace}")
    return pod


@pytest.fixture(scope="class")
def model_registry_model_portforward(
    model_registry_deployment_ns: Namespace,
    model_registry_inference_service: InferenceService,
    model_registry_predictor_pod: Pod,
) -> Generator[str, Any]:
    """
    Port-forwards the Model Registry OpenVINO model server pod to access the model API locally.
    Equivalent CLI:
      oc -n mr-deployment-ns port-forward pod/<pod-name> 8080:8888
    """
    namespace = model_registry_deployment_ns.name
    local_port = 9998  # Different from HF to avoid conflicts
    remote_port = 8888  # OpenVINO Model Server REST port
    local_url = f"http://localhost:{local_port}/v1/models"

    try:
        with portforward.forward(
            pod_or_service=model_registry_predictor_pod.name,
            namespace=namespace,
            from_port=local_port,
            to_port=remote_port,
            waiting=20,
        ):
            LOGGER.info(f"Model Registry model port-forward established: {local_url}")
            LOGGER.info(f"Test with: curl -s {local_url}/{model_registry_inference_service.name}")
            yield local_url
    except Exception as expt:
        LOGGER.error(f"Failed to set up port forwarding for pod {model_registry_predictor_pod.name}: {expt}")
        raise
