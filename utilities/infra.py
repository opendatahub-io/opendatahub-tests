from __future__ import annotations

import json
import os
import shlex
from contextlib import contextmanager
from functools import cache
from typing import Any, Generator, Optional

import kubernetes
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.catalog_source import CatalogSource
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.infrastructure import Infrastructure
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.project_project_openshift_io import Project
from ocp_resources.project_request import ProjectRequest
from ocp_resources.role import Role
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pyhelper_utils.shell import run_command
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.constants import Labels, KServeDeploymentType, Annotations, MODELMESH_SERVING
from utilities.exceptions import InvalidStorageArgumentError, FailedPodsError
from utilities.general import get_s3_secret_dict
from utilities.jira import is_jira_open

LOGGER = get_logger(name=__name__)
TIMEOUT_2MIN = 2 * 60


@contextmanager
def create_ns(
    name: str,
    admin_client: Optional[DynamicClient] = None,
    unprivileged_client: Optional[DynamicClient] = None,
    teardown: bool = True,
    delete_timeout: int = 4 * 60,
    labels: Optional[dict[str, str]] = None,
) -> Generator[Namespace | Project, Any, Any]:
    """
    Create namespace with admin or unprivileged client.

    Args:
        name (str): namespace name.
        admin_client (DynamicClient): admin client.
        unprivileged_client (UnprivilegedClient): unprivileged client.
        teardown (bool): should run resource teardown
        delete_timeout (int): delete timeout.
        labels (dict[str, str]): labels dict to set for namespace

    Yields:
        Namespace | Project: namespace or project

    """
    if unprivileged_client:
        with ProjectRequest(name=name, client=unprivileged_client, teardown=teardown):
            project = Project(
                name=name,
                client=unprivileged_client,
                teardown=teardown,
                delete_timeout=delete_timeout,
            )
            project.wait_for_status(status=project.Status.ACTIVE, timeout=TIMEOUT_2MIN)
            yield project

    else:
        with Namespace(
            client=admin_client,
            name=name,
            label=labels,
            teardown=teardown,
            delete_timeout=delete_timeout,
        ) as ns:
            ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=TIMEOUT_2MIN)
            yield ns


def wait_for_inference_deployment_replicas(
    client: DynamicClient,
    isvc: InferenceService,
    runtime_name: str | None,
    expected_num_deployments: int = 1,
) -> list[Deployment]:
    """
    Wait for inference deployment replicas to complete.

    Args:
        client (DynamicClient): Dynamic client.
        isvc (InferenceService): InferenceService object
        runtime_name (str): ServingRuntime name.
        expected_num_deployments (int): Expected number of deployments per InferenceService.

    Returns:
        list[Deployment]: List of Deployment objects for InferenceService.

    """
    ns = isvc.namespace
    label_selector = create_isvc_label_selector_str(isvc=isvc, resource_type="deployment", runtime_name=runtime_name)

    deployments = list(
        Deployment.get(
            label_selector=label_selector,
            client=client,
            namespace=isvc.namespace,
        )
    )

    LOGGER.info("Waiting for inference deployment replicas to complete")
    if len(deployments) == expected_num_deployments:
        for deployment in deployments:
            if deployment.exists:
                deployment.wait_for_replicas()

        return deployments

    elif len(deployments) > expected_num_deployments:
        raise ResourceNotUniqueError(f"Multiple predictor deployments found in namespace {ns}")

    else:
        raise ResourceNotFoundError(f"Predictor deployment not found in namespace {ns}")


@contextmanager
def s3_endpoint_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
) -> Generator[Secret, Any, Any]:
    """
    Create S3 endpoint secret.

    Args:
        admin_client (DynamicClient): Dynamic client.
        name (str): Secret name.
        namespace (str): Secret namespace name.
        aws_access_key (str): Secret access key.
        aws_secret_access_key (str): Secret access key.
        aws_s3_bucket (str): Secret s3 bucket.
        aws_s3_endpoint (str): Secret s3 endpoint.
        aws_s3_region (str): Secret s3 region.

    Yield:
        Secret: Secret object

    """
    # DO not create secret if exists in the namespace
    os.environ["REUSE_IF_RESOURCE_EXISTS"] = f"{{Secret: {{{name}: {namespace}}}}}"

    with Secret(
        client=admin_client,
        name=name,
        namespace=namespace,
        annotations={"opendatahub.io/connection-type": "s3"},
        # the labels are needed to set the secret as data connection by odh-model-controller
        label={"opendatahub.io/managed": "true", Labels.OpenDataHub.DASHBOARD: "true"},
        data_dict=get_s3_secret_dict(
            aws_access_key=aws_access_key,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_bucket=aws_s3_bucket,
            aws_s3_endpoint=aws_s3_endpoint,
            aws_s3_region=aws_s3_region,
        ),
        wait_for_resource=True,
    ) as secret:
        yield secret


@contextmanager
def create_isvc_view_role(
    client: DynamicClient,
    isvc: InferenceService,
    name: str,
    resource_names: Optional[list[str]] = None,
) -> Generator[Role, Any, Any]:
    """
    Create a view role for an InferenceService.

    Args:
        client (DynamicClient): Dynamic client.
        isvc (InferenceService): InferenceService object.
        name (str): Role name.
        resource_names (list[str]): Resource names to be attached to role.

    Yields:
        Role: Role object.

    """
    rules = [
        {
            "apiGroups": [isvc.api_group],
            "resources": ["inferenceservices"],
            "verbs": ["get"],
        },
    ]

    if resource_names:
        rules[0].update({"resourceNames": resource_names})

    with Role(
        client=client,
        name=name,
        namespace=isvc.namespace,
        rules=rules,
    ) as role:
        yield role


def login_with_user_password(api_address: str, user: str, password: str | None = None) -> bool:
    """
    Log in to an OpenShift cluster using a username and password.

    Args:
        api_address (str): The API address of the OpenShift cluster.
        user (str): Cluster's username
        password (str, optional): Cluster's password

    Returns:
        bool: True if login is successful otherwise False.
    """
    login_command: str = f"oc login  --insecure-skip-tls-verify=true {api_address} -u {user}"
    if password:
        login_command += f" -p '{password}'"

    _, out, _ = run_command(command=shlex.split(login_command), hide_log_command=True)

    return "Login successful" in out


@cache
def is_self_managed_operator(client: DynamicClient) -> bool:
    """
    Check if the operator is self-managed.
    """
    if py_config["distribution"] == "upstream":
        return True

    if CatalogSource(
        client=client,
        name="addon-managed-odh-catalog",
        namespace=py_config["applications_namespace"],
    ).exists:
        return False

    return True


@cache
def is_managed_cluster(client: DynamicClient) -> bool:
    """
    Check if the cluster is managed.
    """
    infra = Infrastructure(client=client, name="cluster")

    if not infra.exists:
        LOGGER.warning(f"Infrastructure {infra.name} resource does not exist in the cluster")
        return False

    platform_statuses = infra.instance.status.platformStatus

    for entries in platform_statuses.values():
        if isinstance(entries, kubernetes.dynamic.resource.ResourceField):
            if tags := entries.resourceTags:
                return next(b["value"] == "true" for b in tags if b["key"] == "red-hat-managed")

    return False


def get_services_by_isvc_label(
    client: DynamicClient, isvc: InferenceService, runtime_name: str | None = None
) -> list[Service]:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        isvc (InferenceService): InferenceService object.
        runtime_name (str): ServingRuntime name

    Returns:
        list[Service]: A list of all matching services

    Raises:
        ResourceNotFoundError: if no services are found.
    """
    label_selector = create_isvc_label_selector_str(isvc=isvc, resource_type="service", runtime_name=runtime_name)

    if svcs := [
        svc
        for svc in Service.get(
            dyn_client=client,
            namespace=isvc.namespace,
            label_selector=label_selector,
        )
    ]:
        return svcs

    raise ResourceNotFoundError(f"{isvc.name} has no services")


def get_pods_by_isvc_label(client: DynamicClient, isvc: InferenceService, runtime_name: str | None = None) -> list[Pod]:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        isvc (InferenceService):InferenceService object.
        runtime_name (str): ServingRuntime name

    Returns:
        list[Pod]: A list of all matching pods

    Raises:
        ResourceNotFoundError: if no pods are found.
    """
    label_selector = create_isvc_label_selector_str(isvc=isvc, resource_type="pod", runtime_name=runtime_name)

    if pods := [
        pod
        for pod in Pod.get(
            dyn_client=client,
            namespace=isvc.namespace,
            label_selector=label_selector,
        )
    ]:
        return pods

    raise ResourceNotFoundError(f"{isvc.name} has no pods")


def get_openshift_token() -> str:
    """
    Get the OpenShift token.

    Returns:
        str: The OpenShift token.

    """
    return run_command(command=shlex.split("oc whoami -t"))[1].strip()


def get_kserve_storage_initialize_image(client: DynamicClient) -> str:
    """
    Get the image used to storage-initializer.

    Args:
        client (DynamicClient): DynamicClient client.

    Returns:
        str: The image used to storage-initializer.

    Raises:
        ResourceNotFoundError: if the config map does not exist.

    """
    kserve_cm = ConfigMap(
        client=client,
        name="inferenceservice-config",
        namespace=py_config["applications_namespace"],
    )

    if not kserve_cm.exists:
        raise ResourceNotFoundError(f"{kserve_cm.name} config map does not exist")

    return json.loads(kserve_cm.instance.data.storageInitializer)["image"]


def get_inference_serving_runtime(isvc: InferenceService) -> ServingRuntime:
    """
    Get the serving runtime for the inference service.

    Args:
        isvc (InferenceService):InferenceService object.

    Returns:
        ServingRuntime: ServingRuntime object.

    Raises:
        ResourceNotFoundError: if the serving runtime does not exist.

    """
    runtime = ServingRuntime(
        client=isvc.client,
        namespace=isvc.namespace,
        name=isvc.instance.spec.predictor.model.runtime,
    )

    if runtime.exists:
        return runtime

    raise ResourceNotFoundError(f"{isvc.name} runtime {runtime.name} does not exist")


def get_model_mesh_route(client: DynamicClient, isvc: InferenceService) -> Route:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        isvc (InferenceService):InferenceService object.

    Returns:
        Route: inference service route

    Raises:
        ResourceNotFoundError: if route was found.
    """
    if routes := [
        route
        for route in Route.get(
            dyn_client=client,
            namespace=isvc.namespace,
            label_selector=f"inferenceservice-name={isvc.name}",
        )
    ]:
        return routes[0]

    raise ResourceNotFoundError(f"{isvc.name} has no routes")


def create_inference_token(model_service_account: ServiceAccount) -> str:
    """
    Generates an inference token for the given model service account.

    Args:
        model_service_account (ServiceAccount): An object containing the namespace and name
                               of the service account.

    Returns:
        str: The generated inference token.
    """
    return run_command(
        shlex.split(f"oc create token -n {model_service_account.namespace} {model_service_account.name}")
    )[1].strip()


def create_isvc_label_selector_str(isvc: InferenceService, resource_type: str, runtime_name: str | None = None) -> str:
    """
    Creates a label selector string for the given InferenceService.

    Args:
        isvc (InferenceService): InferenceService object
        resource_type (str): Type of the resource: service or other for model mesh
        runtime_name (str): ServingRuntime name

    Returns:
        str: Label selector string

    Raises:
        ValueError: If the deployment mode is not supported

    """
    deployment_mode = isvc.instance.metadata.annotations.get(Annotations.KserveIo.DEPLOYMENT_MODE)
    if deployment_mode in (
        KServeDeploymentType.SERVERLESS,
        KServeDeploymentType.RAW_DEPLOYMENT,
    ):
        return f"{isvc.ApiGroup.SERVING_KSERVE_IO}/inferenceservice={isvc.name}"

    elif deployment_mode == KServeDeploymentType.MODEL_MESH:
        if resource_type == "service":
            return f"modelmesh-service={MODELMESH_SERVING}"
        else:
            return f"name={MODELMESH_SERVING}-{runtime_name}"

    else:
        raise ValueError(f"Unknown deployment mode {deployment_mode}")


def verify_no_failed_pods(client: DynamicClient, isvc: InferenceService, runtime_name: str | None) -> None:
    """
    Verify no failed pods.

    Args:
        client (DynamicClient): DynamicClient object
        isvc (InferenceService): InferenceService object
        runtime_name (str): ServingRuntime name

    Raises:
            FailedPodsError: If any pod is in failed state

    """
    failed_pods: dict[str, Any] = {}

    LOGGER.info("Verifying no failed pods")
    for pods in TimeoutSampler(
        wait_timeout=5 * 60,
        sleep=10,
        func=get_pods_by_isvc_label,
        client=client,
        isvc=isvc,
        runtime_name=runtime_name,
    ):
        if pods:
            if all([pod.instance.status.phase == pod.Status.RUNNING for pod in pods]):
                return

            for pod in pods:
                pod_status = pod.instance.status
                if pod_status.containerStatuses:
                    for container_status in pod_status.containerStatuses:
                        if (state := container_status.state.waiting) and state.reason == pod.Status.IMAGE_PULL_BACK_OFF:
                            failed_pods[pod.name] = pod_status

                if init_container_status := pod_status.initContainerStatuses:
                    if container_terminated := init_container_status[0].lastState.terminated:
                        if container_terminated.reason == "Error":
                            failed_pods[pod.name] = pod_status

                elif pod_status.phase in (
                    pod.Status.CRASH_LOOPBACK_OFF,
                    pod.Status.FAILED,
                    pod.Status.IMAGE_PULL_BACK_OFF,
                    pod.Status.ERR_IMAGE_PULL,
                ):
                    failed_pods[pod.name] = pod_status

            if failed_pods:
                raise FailedPodsError(pods=failed_pods)


@contextmanager
def create_isvc(
    client: DynamicClient,
    name: str,
    namespace: str,
    deployment_mode: str,
    model_format: str,
    runtime: str,
    storage_uri: Optional[str] = None,
    storage_key: Optional[str] = None,
    storage_path: Optional[str] = None,
    wait: bool = True,
    enable_auth: bool = False,
    external_route: Optional[bool] = None,
    model_service_account: Optional[str] = "",
    min_replicas: Optional[int] = None,
    argument: Optional[list[str]] = None,
    resources: Optional[dict[str, Any]] = None,
    volumes: Optional[dict[str, Any]] = None,
    volumes_mounts: Optional[dict[str, Any]] = None,
    model_version: Optional[str] = None,
    wait_for_predictor_pods: bool = True,
    autoscaler_mode: Optional[str] = None,
    multi_node_worker_spec: Optional[dict[str, int]] = None,
) -> Generator[InferenceService, Any, Any]:
    """
    Create InferenceService object.

    Args:
        client (DynamicClient): DynamicClient object
        name (str): InferenceService name
        namespace (str): Namespace name
        deployment_mode (str): Deployment mode
        model_format (str): Model format
        runtime (str): ServingRuntime name
        storage_uri (str): Storage URI
        storage_key (str): Storage key
        storage_path (str): Storage path
        wait (bool): Wait for InferenceService to be ready
        enable_auth (bool): Enable authentication
        external_route (bool): External route
        model_service_account (str): Model service account
        min_replicas (int): Minimum replicas
        argument (list[str]): Argument
        resources (dict[str, Any]): Resources
        volumes (dict[str, Any]): Volumes
        volumes_mounts (dict[str, Any]): Volumes mounts
        model_version (str): Model version
        wait_for_predictor_pods (bool): Wait for predictor pods
        autoscaler_mode (str): Autoscaler mode
        multi_node_worker_spec (dict[str, int]): Multi node worker spec
        wait_for_predictor_pods (bool): Wait for predictor pods

    Yields:
        InferenceService: InferenceService object

    """
    labels: dict[str, str] = {}
    predictor_dict: dict[str, Any] = {
        "minReplicas": min_replicas,
        "model": {
            "modelFormat": {"name": model_format},
            "version": "1",
            "runtime": runtime,
        },
    }

    if model_version:
        predictor_dict["model"]["modelFormat"]["version"] = model_version

    _check_storage_arguments(storage_uri=storage_uri, storage_key=storage_key, storage_path=storage_path)
    if storage_uri:
        predictor_dict["model"]["storageUri"] = storage_uri
    elif storage_key:
        predictor_dict["model"]["storage"] = {"key": storage_key, "path": storage_path}
    if model_service_account:
        predictor_dict["serviceAccountName"] = model_service_account

    if min_replicas:
        predictor_dict["minReplicas"] = min_replicas
    if argument:
        predictor_dict["model"]["args"] = argument
    if resources:
        predictor_dict["model"]["resources"] = resources
    if volumes_mounts:
        predictor_dict["model"]["volumeMounts"] = volumes_mounts
    if volumes:
        predictor_dict["volumes"] = volumes

    annotations = {Annotations.KserveIo.DEPLOYMENT_MODE: deployment_mode}

    if deployment_mode == KServeDeploymentType.SERVERLESS:
        annotations.update({
            "serving.knative.openshift.io/enablePassthrough": "true",
            "sidecar.istio.io/inject": "true",
            "sidecar.istio.io/rewriteAppHTTPProbers": "true",
        })

    if enable_auth:
        # model mesh auth is set in servingruntime
        if deployment_mode == KServeDeploymentType.SERVERLESS:
            annotations[Annotations.KserveAuth.SECURITY] = "true"
        elif deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
            labels[Labels.KserveAuth.SECURITY] = "true"

    # default to True if deployment_mode is Serverless (default behavior of Serverless) if was not provided by the user
    # model mesh external route is set in servingruntime
    if external_route is None and deployment_mode == KServeDeploymentType.SERVERLESS:
        external_route = True

    if external_route and deployment_mode == KServeDeploymentType.RAW_DEPLOYMENT:
        labels["networking.kserve.io/visibility"] = "exposed"

    if deployment_mode == KServeDeploymentType.SERVERLESS and external_route is False:
        labels["networking.knative.dev/visibility"] = "cluster-local"

    if autoscaler_mode:
        annotations["serving.kserve.io/autoscalerClass"] = autoscaler_mode

    if multi_node_worker_spec is not None:
        predictor_dict["workerSpec"] = multi_node_worker_spec

    with InferenceService(
        client=client,
        name=name,
        namespace=namespace,
        annotations=annotations,
        predictor=predictor_dict,
        label=labels,
    ) as inference_service:
        if wait_for_predictor_pods:
            verify_no_failed_pods(client=client, isvc=inference_service, runtime_name=runtime)
            wait_for_inference_deployment_replicas(client=client, isvc=inference_service, runtime_name=runtime)

        if wait:
            # Modelmesh 2nd server in the ns will fail to be Ready; isvc needs to be re-applied
            if is_jira_open(jira_id="RHOAIENG-13636") and deployment_mode == KServeDeploymentType.MODEL_MESH:
                for isvc in InferenceService.get(dyn_client=client, namespace=namespace):
                    _runtime = get_inference_serving_runtime(isvc=isvc)
                    isvc_annotations = isvc.instance.metadata.annotations
                    if (
                        _runtime.name != runtime
                        and isvc_annotations
                        and isvc_annotations.get(Annotations.KserveIo.DEPLOYMENT_MODE)
                        == KServeDeploymentType.MODEL_MESH
                    ):
                        LOGGER.warning(
                            "Bug RHOAIENG-13636 - re-creating isvc if there's already a modelmesh isvc in the namespace"
                        )
                        inference_service.clean_up()
                        inference_service.deploy()

                        break

            inference_service.wait_for_condition(
                condition=inference_service.Condition.READY,
                status=inference_service.Condition.Status.TRUE,
                timeout=15 * 60,
            )

        yield inference_service


def _check_storage_arguments(
    storage_uri: Optional[str],
    storage_key: Optional[str],
    storage_path: Optional[str],
) -> None:
    """
    Check if storage_uri, storage_key and storage_path are valid.

    Args:
        storage_uri (str): URI of the storage.
        storage_key (str): Key of the storage.
        storage_path (str): Path of the storage.

    Raises:
        InvalidStorageArgumentError: If storage_uri, storage_key and storage_path are not valid.
    """
    if (storage_uri and storage_path) or (not storage_uri and not storage_key) or (storage_key and not storage_path):
        raise InvalidStorageArgumentError(storage_uri=storage_uri, storage_key=storage_key, storage_path=storage_path)
