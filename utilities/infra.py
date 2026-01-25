import json
import os
import re
import shlex
import stat
import tarfile
import zipfile
from contextlib import contextmanager
from functools import cache
from typing import Any, Generator, Optional, Set, Callable

import kubernetes
import platform
import pytest
import requests
import urllib3
from _pytest._py.path import LocalPath
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import (
    NotFoundError,
    ResourceNotFoundError,
)

from ocp_resources.authentication_config_openshift_io import Authentication
from ocp_resources.catalog_source import CatalogSource
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.config_map import ConfigMap
from ocp_resources.config_imageregistry_operator_openshift_io import Config
from ocp_resources.console_cli_download import ConsoleCLIDownload
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.dsc_initialization import DSCInitialization
from ocp_resources.exceptions import MissingResourceError
from ocp_resources.inference_graph import InferenceGraph
from ocp_resources.inference_service import InferenceService
from ocp_resources.infrastructure import Infrastructure
from ocp_resources.namespace import Namespace
from ocp_resources.node_config_openshift_io import Node
from ocp_resources.pod import Pod
from ocp_resources.project_project_openshift_io import Project
from ocp_resources.project_request import ProjectRequest
from ocp_resources.resource import Resource, ResourceEditor, get_client
from ocp_resources.role import Role
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from ocp_utilities.exceptions import NodeNotReadyError, NodeUnschedulableError
from ocp_utilities.infra import (
    assert_nodes_in_healthy_condition,
    assert_nodes_schedulable,
)
from pyhelper_utils.shell import run_command
from pytest_testconfig import config as py_config
from semver import Version
from simple_logger.logger import get_logger

from ocp_resources.subscription import Subscription
from utilities.constants import ApiGroups, Labels, Timeout, RHOAI_OPERATOR_NAMESPACE
from utilities.constants import KServeDeploymentType
from utilities.constants import Annotations
from utilities.exceptions import ClusterLoginError, FailedPodsError, ResourceNotReadyError, UnexpectedResourceCountError
from timeout_sampler import TimeoutExpiredError, TimeoutSampler, TimeoutWatch, retry
import utilities.general
from ocp_resources.utils.constants import DEFAULT_CLUSTER_RETRY_EXCEPTIONS
from utilities.general import generate_random_name

LOGGER = get_logger(name=__name__)


@contextmanager
def create_ns(
    admin_client: DynamicClient,
    name: str | None = None,
    unprivileged_client: DynamicClient | None = None,
    teardown: bool = True,
    delete_timeout: int = Timeout.TIMEOUT_4MIN,
    labels: dict[str, str] | None = None,
    ns_annotations: dict[str, str] | None = None,
    model_mesh_enabled: bool = False,
    add_dashboard_label: bool = False,
    add_kueue_label: bool = False,
    randomize_name: bool = False,
    pytest_request: FixtureRequest | None = None,
) -> Generator[Namespace | Project, Any, Any]:
    """
    Create namespace with admin or unprivileged client.

    For a namespace / project which contains Serverless ISVC,  there is a workaround for RHOAIENG-19969.
    Currently, when Serverless ISVC is deleted and the namespace is deleted, namespace "SomeResourcesRemain" is True.
    This is because the serverless pods are not immediately deleted resulting in prolonged namespace deletion.
    Waiting for the pod(s) to be deleted before cleanup, eliminates the issue.

    Args:
        name (str): namespace name.
            Can be overwritten by `request.param["name"]`
        admin_client (DynamicClient): admin client.
        unprivileged_client (UnprivilegedClient): unprivileged client.
        teardown (bool): should run resource teardown
        delete_timeout (int): delete timeout.
        labels (dict[str, str]): labels dict to set for namespace
        ns_annotations (dict[str, str]): annotations dict to set for namespace
            Can be overwritten by `request.param["annotations"]`
        model_mesh_enabled (bool): if True, model mesh will be enabled in namespace.
            Can be overwritten by `request.param["modelmesh-enabled"]`
        add_dashboard_label (bool): if True, dashboard label will be added to namespace
            Can be overwritten by `request.param["add-dashboard-label"]`
        pytest_request (FixtureRequest): pytest request
        randomize_name (bool): if True, randomize the namespace name.
            Can be overwritten by `request.param["randomize_name"]`

    Yields:
        Namespace | Project: namespace or project

    """
    if pytest_request:
        name = pytest_request.param.get("name", name)
        ns_annotations = pytest_request.param.get("annotations", ns_annotations)
        model_mesh_enabled = pytest_request.param.get("modelmesh-enabled", model_mesh_enabled)
        add_dashboard_label = pytest_request.param.get("add-dashboard-label", add_dashboard_label)
        add_kueue_label = pytest_request.param.get("add-kueue-label", add_kueue_label)
        randomize_name = pytest_request.param.get("randomize_name", randomize_name)

    # Handle randomization
    base_name = name or "test-namespace"
    if randomize_name:
        name = generate_random_name(prefix=base_name, length=4)

    namespace_kwargs = {
        "name": name,
        "teardown": teardown,
        "delete_timeout": delete_timeout,
        "label": labels or {},
    }

    if ns_annotations:
        namespace_kwargs["annotations"] = ns_annotations

    if model_mesh_enabled:
        namespace_kwargs["label"]["modelmesh-enabled"] = "true"  # type: ignore

    if add_dashboard_label:
        namespace_kwargs["label"][Labels.OpenDataHub.DASHBOARD] = "true"  # type: ignore

    if add_kueue_label:
        namespace_kwargs["label"][Labels.Kueue.MANAGED] = "true"  # type: ignore

    if not unprivileged_client:
        namespace_kwargs["client"] = admin_client
        with Namespace(**namespace_kwargs) as ns:
            ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=Timeout.TIMEOUT_2MIN)
            yield ns
            if teardown:
                wait_for_serverless_pods_deletion(resource=ns, admin_client=admin_client)
    else:
        namespace_kwargs["client"] = unprivileged_client
        project = ProjectRequest(**namespace_kwargs).deploy()
        if _labels := namespace_kwargs.get("label", {}):
            # To patch the namespace, admin client is required
            ns = Namespace(client=admin_client, name=name)
            ResourceEditor({
                ns: {
                    "metadata": {
                        "labels": _labels,
                    }
                }
            }).update()
        yield project
        if teardown:
            wait_for_serverless_pods_deletion(resource=project, admin_client=admin_client)
            # cleanup must be done with admin admin_client
            project.client = admin_client
            project.clean_up()


def wait_for_replicas_in_deployment(deployment: Deployment, replicas: int, timeout: int = Timeout.TIMEOUT_2MIN) -> None:
    """
    Wait for replicas in deployment to updated in spec.

    Args:
        deployment (Deployment): Deployment object
        replicas (int): number of replicas to be set in spec.replicas
        timeout (int): Time to wait for the model deployment.

    Raises:
        TimeoutExpiredError: If replicas are not updated in spec.

    """
    _replicas: int | None = None

    try:
        for sample in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: deployment.instance,
        ):
            if sample and (_replicas := sample.spec.replicas) == replicas:
                return

    except TimeoutExpiredError:
        LOGGER.error(
            f"Replicas are not updated in spec.replicas for deployment {deployment.name}.Current replicas: {_replicas}"
        )
        raise


def wait_for_inference_deployment_replicas(
    client: DynamicClient,
    isvc: InferenceService,
    runtime_name: str | None = None,
    expected_num_deployments: int = 1,
    labels: str = "",
    deployed: bool = True,
    timeout: int = Timeout.TIMEOUT_5MIN,
) -> list[Deployment]:
    """
    Wait for inference deployment replicas to complete.

    Args:
        client (DynamicClient): Dynamic client.
        isvc (InferenceService): InferenceService object
        runtime_name (str): ServingRuntime name.
        expected_num_deployments (int): Expected number of deployments per InferenceService.
        labels (str): Comma seperated list of labels, in key=value format, used to filter deployments.
        deployed (bool): True for replicas deployed, False for no replicas.
        timeout (int): Time to wait for the model deployment.

    Returns:
        list[Deployment]: List of Deployment objects for InferenceService.

    Raises:
        TimeoutExpiredError: If an exception is raised when retrieving deployments or
                             timeout expires when checking replicas.
        UnexpectedResourceCountError: If the expected number of deployments is not found after timeout.
        ResourceNotFoundError: If any of the retrieved deployments are found to no longer exist.
    """
    timeout_watcher = TimeoutWatch(timeout=timeout)
    ns = isvc.namespace
    label_selector = utilities.general.create_isvc_label_selector_str(
        isvc=isvc, resource_type="deployment", runtime_name=runtime_name
    )
    if labels:
        label_selector += f",{labels}"

    deployment_list = []
    try:
        for deployments in TimeoutSampler(
            wait_timeout=timeout_watcher.remaining_time(),
            sleep=5,
            exceptions_dict=DEFAULT_CLUSTER_RETRY_EXCEPTIONS,
            func=Deployment.get,
            label_selector=label_selector,
            client=client,
            namespace=ns,
        ):
            deployment_list = list(deployments)
            if len(deployment_list) == expected_num_deployments:
                break
    except TimeoutExpiredError as e:
        # If the last exception raised prior to the timeout expiring is None, this means that
        # the deployments were successfully retrieved, but the expected number was not found.
        if e.last_exp is None:
            raise UnexpectedResourceCountError(
                f"Expected {expected_num_deployments} predictor deployments to be found in "
                f"namespace {ns} after timeout, but found {len(deployment_list)}."
            )
        raise

    LOGGER.info("Waiting for inference deployment replicas to complete")
    for deployment in deployment_list:
        if deployment.exists:
            # Raw deployment: if min replicas is more than 1, wait for min replicas
            # to be set in deployment spec by HPA
            if (
                isvc.instance.metadata.annotations.get("serving.kserve.io/deploymentMode")
                == KServeDeploymentType.RAW_DEPLOYMENT
            ):
                wait_for_replicas_in_deployment(
                    deployment=deployment,
                    replicas=isvc.instance.spec.predictor.get("minReplicas", 1),
                    timeout=timeout_watcher.remaining_time(),
                )

            deployment.wait_for_replicas(deployed=deployed, timeout=timeout_watcher.remaining_time())
        else:
            raise ResourceNotFoundError(f"Predictor deployment {deployment.name} does not exist on the server.")

    return deployment_list


@contextmanager
def s3_endpoint_secret(
    client: DynamicClient,
    name: str,
    namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
    teardown: bool = True,
) -> Generator[Secret, Any, Any]:
    """
    Create S3 endpoint secret.

    Args:
        client (DynamicClient): Dynamic client.
        name (str): Secret name.
        namespace (str): Secret namespace name.
        aws_access_key (str): Secret access key.
        aws_secret_access_key (str): Secret access key.
        aws_s3_bucket (str): Secret s3 bucket.
        aws_s3_endpoint (str): Secret s3 endpoint.
        aws_s3_region (str): Secret s3 region.
        teardown (bool): Whether to delete the secret.

    Yield:
        Secret: Secret object

    """
    secret_kwargs = {"client": client, "name": name, "namespace": namespace}
    secret = Secret(**secret_kwargs)

    if secret.exists:
        LOGGER.info(f"Secret {name} already exists in namespace {namespace}")
        yield secret

    else:
        # Determine usehttps based on endpoint protocol
        usehttps = 0
        if aws_s3_endpoint.startswith("https://"):
            usehttps = 1
        with Secret(
            annotations={
                f"{ApiGroups.OPENDATAHUB_IO}/connection-type": "s3",
                "serving.kserve.io/s3-endpoint": (aws_s3_endpoint.replace("https://", "").replace("http://", "")),
                "serving.kserve.io/s3-region": aws_s3_region,
                "serving.kserve.io/s3-useanoncredential": "false",
                "serving.kserve.io/s3-verifyssl": "0",
                "serving.kserve.io/s3-usehttps": str(usehttps),
            },
            # the labels are needed to set the secret as data connection by odh-model-controller
            label={
                Labels.OpenDataHubIo.MANAGED: "true",
                Labels.OpenDataHub.DASHBOARD: "true",
            },
            data_dict=utilities.general.get_s3_secret_dict(
                aws_access_key=aws_access_key,
                aws_secret_access_key=aws_secret_access_key,
                aws_s3_bucket=aws_s3_bucket,
                aws_s3_endpoint=aws_s3_endpoint,
                aws_s3_region=aws_s3_region,
            ),
            wait_for_resource=True,
            teardown=teardown,
            **secret_kwargs,
        ) as secret:
            yield secret


@contextmanager
def create_isvc_view_role(
    client: DynamicClient,
    isvc: InferenceService,
    name: str,
    resource_names: Optional[list[str]] = None,
    teardown: bool = True,
) -> Generator[Role, Any, Any]:
    """
    Create a view role for an InferenceService.

    Args:
        client (DynamicClient): Dynamic client.
        isvc (InferenceService): InferenceService object.
        name (str): Role name.
        resource_names (list[str]): Resource names to be attached to role.
        teardown (bool): Whether to delete the role.

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
        teardown=teardown,
    ) as role:
        yield role


@contextmanager
def create_inference_graph_view_role(
    client: DynamicClient,
    namespace: str,
    name: str,
    resource_names: Optional[list[str]] = None,
    teardown: bool = True,
) -> Generator[Role, Any, Any]:
    """
    Create a view role for an InferenceGraph.

    Args:
        client (DynamicClient): Dynamic client.
        namespace (str): Namespace to create the Role.
        name (str): Role name.
        resource_names (list[str]): Resource names to be attached to role.
        teardown (bool): Whether to delete the role.

    Yields:
        Role: Role object.

    """
    rules = [
        {
            "apiGroups": [InferenceGraph.api_group],
            "resources": ["inferencegraphs"],
            "verbs": ["get"],
        },
    ]

    if resource_names:
        rules[0].update({"resourceNames": resource_names})

    with Role(
        client=client,
        name=name,
        namespace=namespace,
        rules=rules,
        teardown=teardown,
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

    _, out, err = run_command(command=shlex.split(login_command), hide_log_command=True)

    if err and err.lower().startswith("error"):
        raise ClusterLoginError(user=user)

    if re.search(r"Login successful|Logged into", out):
        return True

    return False


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

    for entry in platform_statuses.values():
        if isinstance(entry, kubernetes.dynamic.resource.ResourceField):
            if tags := entry.resourceTags:
                LOGGER.info(f"Infrastructure {infra.name} resource tags: {tags}")
                return any([tag["value"] == "true" for tag in tags if tag["key"] == "red-hat-managed"])

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
    label_selector = utilities.general.create_isvc_label_selector_str(
        isvc=isvc, resource_type="service", runtime_name=runtime_name
    )

    if svcs := [
        svc
        for svc in Service.get(
            client=client,
            namespace=isvc.namespace,
            label_selector=label_selector,
        )
    ]:
        return svcs

    raise ResourceNotFoundError(f"{isvc.name} has no services")


def get_pods_by_ig_label(client: DynamicClient, ig: InferenceGraph) -> list[Pod]:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        ig (InferenceGraph): InferenceGraph object.

    Returns:
        list[Pod]: A list of all matching pods

    Raises:
        ResourceNotFoundError: if no services are found.
    """
    label_selector = utilities.general.create_ig_pod_label_selector_str(ig=ig)

    if pods := [
        pod
        for pod in Pod.get(
            client=client,
            namespace=ig.namespace,
            label_selector=label_selector,
        )
    ]:
        return pods

    raise ResourceNotFoundError(f"{ig.name} has no pods")


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
    label_selector = utilities.general.create_isvc_label_selector_str(
        isvc=isvc, resource_type="pod", runtime_name=runtime_name
    )

    if pods := [
        pod
        for pod in Pod.get(
            client=client,
            namespace=isvc.namespace,
            label_selector=label_selector,
        )
    ]:
        return pods

    raise ResourceNotFoundError(f"{isvc.name} has no pods")


def get_openshift_token(client: DynamicClient | None = None) -> str:
    """
    Get the OpenShift token.

    Returns:
        str: The OpenShift token.

    """
    client = client or get_client()
    LOGGER.info("Getting OpenShift token")
    bearer_str = client.configuration.api_key["authorization"]
    assert bearer_str, "No OpenShift token found from client instance"
    match = re.search(r"Bearer (.+)", bearer_str)
    assert match, "No OpenShift token found"
    return match.group(1)


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


def get_model_route(client: DynamicClient, isvc: InferenceService) -> Route:
    """
    Get model route using  InferenceService
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
            client=client,
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


@contextmanager
def update_configmap_data(
    client: DynamicClient, name: str, namespace: str, data: dict[str, Any]
) -> Generator[ConfigMap, Any, Any]:
    """
    Update the data of a configmap.

    Args:
        client (DynamicClient): DynamicClient client.
        name (str): Name of the configmap.
        namespace (str): Namespace of the configmap.
        data (dict[str, Any]): Data to update the configmap with.

    Yields:
        ConfigMap: The updated configmap.

    """
    config_map = ConfigMap(client=client, name=name, namespace=namespace)

    # Some CM resources may already be present as they are usually created when doing exploratory testing
    if config_map.exists:
        with ResourceEditor(patches={config_map: {"data": data}}):
            yield config_map

    else:
        config_map.data = data
        with config_map as cm:
            yield cm


def verify_no_failed_pods(
    client: DynamicClient,
    isvc: InferenceService,
    runtime_name: str | None = None,
    timeout: int = Timeout.TIMEOUT_5MIN,
) -> None:
    """
    Verify pods created and no failed pods.

    Args:
        client (DynamicClient): DynamicClient object
        isvc (InferenceService): InferenceService object
        runtime_name (str): ServingRuntime name
        timeout (int): Time to wait for the pod.

    Raises:
        FailedPodsError: If any pod is in failed state

    """
    wait_for_isvc_pods(client=client, isvc=isvc, runtime_name=runtime_name)

    LOGGER.info("Verifying no failed pods")
    for pods in TimeoutSampler(
        wait_timeout=timeout,
        sleep=10,
        func=get_pods_by_isvc_label,
        client=client,
        isvc=isvc,
        runtime_name=runtime_name,
    ):
        ready_pods = 0
        failed_pods: dict[str, Any] = {}

        container_wait_base_errors = ["InvalidImageName"]
        container_terminated_base_errors = [Resource.Status.ERROR]

        # For Model Mesh, if image pulling takes longer, pod may be in CrashLoopBackOff state but recover with retries.
        if (
            deployment_mode := isvc.instance.metadata.annotations.get("serving.kserve.io/deploymentMode")
        ) and deployment_mode != KServeDeploymentType.MODEL_MESH:
            container_wait_base_errors.append(Resource.Status.CRASH_LOOPBACK_OFF)
            container_terminated_base_errors.append(Resource.Status.CRASH_LOOPBACK_OFF)

        if pods:
            for pod in pods:
                for condition in pod.instance.status.conditions:
                    if condition.type == pod.Status.READY and condition.status == pod.Condition.Status.TRUE:
                        ready_pods += 1

            if ready_pods == len(pods):
                return

            for pod in pods:
                pod_status = pod.instance.status

                if pod_status.containerStatuses:
                    for container_status in pod_status.get("containerStatuses", []) + pod_status.get(
                        "initContainerStatuses", []
                    ):
                        is_waiting_pull_back_off = (
                            wait_state := container_status.state.waiting
                        ) and wait_state.reason in container_wait_base_errors

                        is_terminated_error = (
                            terminate_state := container_status.state.terminated
                        ) and terminate_state.reason in container_terminated_base_errors

                        if is_waiting_pull_back_off or is_terminated_error:
                            failed_pods[pod.name] = pod_status

                elif pod_status.phase in (
                    pod.Status.CRASH_LOOPBACK_OFF,
                    pod.Status.FAILED,
                ):
                    failed_pods[pod.name] = pod_status

            if failed_pods:
                raise FailedPodsError(pods=failed_pods)


def check_pod_status_in_time(pod: Pod, status: Set[str], duration: int = Timeout.TIMEOUT_2MIN, wait: int = 1) -> None:
    """
    Checks if a pod status is maintained for a given duration. If not, an AssertionError is raised.

    Args:
        pod (Pod): The pod to check
        status (Set[Pod.Status]): Expected pod status(es)
        duration (int): Maximum time to check for in seconds
        wait (int): Time to wait between checks in seconds

    Raises:
        AssertionError: If pod status is not in the expected set
    """
    LOGGER.info(f"Checking pod status for {pod.name} to be {status} for {duration} seconds")

    sampler = TimeoutSampler(
        wait_timeout=duration,
        sleep=wait,
        func=lambda: pod.instance,
    )

    try:
        for sample in sampler:
            if sample:
                if sample.status.phase not in status:
                    raise AssertionError(f"Pod status is not the expected: {pod.status}")

    except TimeoutExpiredError:
        LOGGER.info(f"Pod status is {pod.status} as expected")


def get_product_version(admin_client: DynamicClient) -> Version:
    """
    Get RHOAI/ODH product version

    Args:
        admin_client (DynamicClient): DynamicClient object

    Returns:
        Version: RHOAI/ODH product version

    Raises:
        MissingResourceError: If product's ClusterServiceVersion not found

    """
    operator_version: str = ""
    for csv in ClusterServiceVersion.get(client=admin_client, namespace=py_config["applications_namespace"]):
        if re.match("rhods|opendatahub", csv.name):
            operator_version = csv.instance.spec.version
            break

    if not operator_version:
        raise MissingResourceError("Operator ClusterServiceVersion not found")

    return Version.parse(version=operator_version)


def get_data_science_cluster(client: DynamicClient, dsc_name: str = "default-dsc") -> DataScienceCluster:
    return DataScienceCluster(client=client, name=dsc_name, ensure_exists=True)


def get_dsci_applications_namespace(client: DynamicClient) -> str:
    """
    Get the namespace where DSCI applications are deployed.
    Args:
        client (DynamicClient): DynamicClient object
    Returns:
        str: Namespace where DSCI applications are deployed.
    Raises:
            ValueError: If DSCI applications namespace not found
            MissingResourceError: If DSCI not found
    """
    dsci_name = py_config["dsci_name"]
    dsci = DSCInitialization(client=client, name=dsci_name)

    if dsci.exists:
        if app_namespace := dsci.instance.spec.get("applicationsNamespace"):
            return app_namespace

        else:
            raise ValueError("DSCI applications namespace not found in {dsci_name}")

    raise MissingResourceError(f"DSCI {dsci_name} not found")


def get_operator_distribution(client: DynamicClient, dsc_name: str = "default-dsc") -> str:
    """
    Get the operator distribution.

    Args:
        client (DynamicClient): DynamicClient object
        dsc_name (str): DSC name

    Returns:
        str: Operator distribution. One of Open Data Hub or OpenShift AI.

    Raises:
            ValueError: If DSC release name not found

    """
    dsc = get_data_science_cluster(client=client, dsc_name=dsc_name)

    if dsc_release_name := dsc.instance.status.get("release", {}).get("name"):
        return dsc_release_name

    else:
        raise ValueError("DSC release name not found in {dsc_name}")


def wait_for_route_timeout(name: str, namespace: str, route_timeout: str) -> None:
    """
    Wait for route to be annotated with timeout value.
    Given that there is a delay between the openshift route timeout annotation being set
    and the timeout being applied to the route, a counter is instituted to wait until the
    annotation is found in the route twice. This allows for the TimeoutSampler sleep time
    to be executed and the route timeout to be successfully applied.

    Args:
        name (str): Name of the route.
        namespace (str): Namespace the route is located in.
        route_timeout (str): The expected value of the openshift route timeout annotation.

    Raises:
        TimeoutExpiredError: If route annotation is not set to the expected value before timeout expires.
    """
    annotation_found_count = 0
    for route in TimeoutSampler(
        wait_timeout=Timeout.TIMEOUT_30SEC,
        sleep=10,
        exceptions_dict={ResourceNotFoundError: []},
        func=Route,
        name=name,
        namespace=namespace,
        ensure_exists=True,
    ):
        if (
            route.instance.metadata.get("annotations", {}).get(Annotations.HaproxyRouterOpenshiftIo.TIMEOUT)
            != route_timeout
        ):
            continue
        annotation_found_count += 1
        if annotation_found_count == 2:
            return


def wait_for_serverless_pods_deletion(resource: Project | Namespace, admin_client: DynamicClient | None) -> None:
    """
    Wait for serverless pods deletion.

    Args:
        resource (Project | Namespace): project or namespace
        admin_client (DynamicClient): admin client.

    Returns:
        bool: True if we should wait for namespace deletion else False

    """
    client = admin_client or get_client()
    for pod in Pod.get(client=client, namespace=resource.name):
        try:
            if (
                pod.exists
                and pod.instance.metadata.annotations.get(Annotations.KserveIo.DEPLOYMENT_MODE)
                == KServeDeploymentType.SERVERLESS
            ):
                LOGGER.info(f"Waiting for {KServeDeploymentType.SERVERLESS} pod {pod.name} to be deleted")
                pod.wait_deleted(timeout=Timeout.TIMEOUT_1MIN)

        except (ResourceNotFoundError, NotFoundError):
            LOGGER.info(f"Pod {pod.name} is deleted")


@retry(
    wait_timeout=Timeout.TIMEOUT_30SEC,
    sleep=1,
    exceptions_dict={ResourceNotFoundError: []},
)
def wait_for_isvc_pods(client: DynamicClient, isvc: InferenceService, runtime_name: str | None = None) -> list[Pod]:
    """
    Wait for ISVC pods.

    Args:
        client (DynamicClient): DynamicClient object
        isvc (InferenceService): InferenceService object
        runtime_name (ServingRuntime): ServingRuntime name

    Returns:
        list[Pod]: A list of all matching pods

    Raises:
        TimeoutExpiredError: If pods do not exist
    """
    LOGGER.info("Waiting for pods to be created")
    return get_pods_by_isvc_label(client=client, isvc=isvc, runtime_name=runtime_name)


def get_rhods_subscription() -> Subscription | None:
    subscriptions = Subscription.get(client=get_client(), namespace=RHOAI_OPERATOR_NAMESPACE)
    if subscriptions:
        for subscription in subscriptions:
            LOGGER.info(f"Checking subscription {subscription.name}")
            if subscription.name.startswith(tuple(["rhods-operator", "rhoai-operator"])):
                return subscription

    LOGGER.warning("No RHOAI subscription found. Potentially ODH cluster")
    return None


def get_rhods_operator_installed_csv() -> ClusterServiceVersion | None:
    subscription = get_rhods_subscription()
    if subscription:
        csv_name = subscription.instance.status.installedCSV
        LOGGER.info(f"Expected CSV: {csv_name}")
        return ClusterServiceVersion(name=csv_name, namespace=RHOAI_OPERATOR_NAMESPACE, ensure_exists=True)
    return None


@retry(
    wait_timeout=120,
    sleep=5,
    exceptions_dict={ResourceNotReadyError: []},
)
def wait_for_dsci_status_ready(dsci_resource: DSCInitialization) -> bool:
    LOGGER.info(f"Wait for DSCI {dsci_resource.name} to be in {dsci_resource.Status.READY} status.")
    if dsci_resource.status == dsci_resource.Status.READY:
        return True

    raise ResourceNotReadyError(
        f"DSCI {dsci_resource.name} is not ready.\nCurrent status: {dsci_resource.instance.status}"
    )


@retry(
    wait_timeout=120,
    sleep=5,
    exceptions_dict={ResourceNotReadyError: []},
)
def wait_for_dsc_status_ready(dsc_resource: DataScienceCluster) -> bool:
    LOGGER.info(f"Wait for DSC {dsc_resource.name} are {dsc_resource.Status.READY}.")
    if dsc_resource.status == dsc_resource.Status.READY:
        return True
    raise ResourceNotReadyError(
        f"DSC {dsc_resource.name} is not ready.\nCurrent status: {dsc_resource.instance.status}"
    )


def verify_cluster_sanity(
    request: FixtureRequest,
    nodes: list[Node],
    dsci_resource: DSCInitialization,
    dsc_resource: DataScienceCluster,
    junitxml_property: Callable[[str, object], None] | None = None,
) -> None:
    """
    Check that cluster resources (Nodes, DSCI, DSC) are healthy and exists pytest execution on failure.

    Args:
        request (FixtureRequest): pytest request
        nodes (list[Node]): list of nodes
        dsci_resource (DSCInitialization): dsci resource
        dsc_resource (DataScienceCluster): dsc resource
        junitxml_property (property): Junitxml property

    """
    skip_cluster_sanity_check = "--cluster-sanity-skip-check"
    skip_rhoai_check = "--cluster-sanity-skip-rhoai-check"

    if request.session.config.getoption(skip_cluster_sanity_check):
        LOGGER.warning(f"Skipping cluster sanity check, got {skip_cluster_sanity_check}")
        return

    try:
        LOGGER.info("Check cluster sanity.")

        assert_nodes_in_healthy_condition(nodes=nodes, healthy_node_condition_type={"KubeletReady": "True"})
        assert_nodes_schedulable(nodes=nodes)

        if request.session.config.getoption(skip_rhoai_check):
            LOGGER.warning(f"Skipping RHOAI resource checks, got {skip_rhoai_check}")

        else:
            wait_for_dsci_status_ready(dsci_resource=dsci_resource)
            wait_for_dsc_status_ready(dsc_resource=dsc_resource)

    except (ResourceNotReadyError, NodeUnschedulableError, NodeNotReadyError) as ex:
        error_msg = f"Cluster sanity check failed: {str(ex)}"
        # return_code set to 99 to not collide with https://docs.pytest.org/en/stable/reference/exit-codes.html
        return_code = 99

        LOGGER.error(error_msg)

        if junitxml_property:
            junitxml_property(name="exit_code", value=return_code)  # type: ignore[call-arg]

        # TODO: Write to file to easily report the failure in jenkins
        pytest.exit(reason=error_msg, returncode=return_code)


def get_machine_platform() -> str:
    os_machine_type = platform.machine()
    return "amd64" if os_machine_type == "x86_64" else os_machine_type


def get_os_system() -> str:
    os_system = platform.system().lower()
    if os_system == "darwin" and platform.mac_ver()[0]:
        os_system = "mac"
    return os_system


def get_oc_console_cli_download_link(admin_client: DynamicClient) -> str:
    oc_console_cli_download = ConsoleCLIDownload(client=admin_client, name="oc-cli-downloads", ensure_exists=True)
    os_system = get_os_system()
    machine_platform = get_machine_platform()
    oc_links = oc_console_cli_download.instance.spec.links
    all_links = [
        link_ref.href
        for link_ref in oc_links
        if link_ref.href.endswith(("oc.tar", "oc.zip"))
        and os_system in link_ref.href
        and machine_platform in link_ref.href
    ]
    LOGGER.info(f"All oc console cli download links: {all_links}")
    if not all_links:
        raise ValueError(f"No oc console cli download link found for {os_system} {machine_platform} in {oc_links}")

    return all_links[0]


def download_oc_console_cli(admin_client: DynamicClient, tmpdir: LocalPath) -> str:
    """
    Download and extract the OpenShift CLI binary.

    Args:
        admin_client (DynamicClient): admin client
        tmpdir (str): Directory to download and extract the binary to

    Returns:
        str: Path to the extracted binary

    Raises:
        ValueError: If multiple files are found in the archive or if no download link is found
    """
    oc_console_cli_download_link = get_oc_console_cli_download_link(admin_client=admin_client)
    LOGGER.info(f"Downloading archive using: url={oc_console_cli_download_link}")
    urllib3.disable_warnings()  # TODO: remove when cert issue is addressed for managed clusters
    local_file_name = os.path.join(tmpdir, oc_console_cli_download_link.split("/")[-1])
    with requests.get(oc_console_cli_download_link, verify=False, stream=True) as created_request:
        created_request.raise_for_status()
        with open(local_file_name, "wb") as file_downloaded:
            for chunk in created_request.iter_content(chunk_size=8192):
                file_downloaded.write(chunk)
    LOGGER.info("Extract the downloaded archive.")
    extracted_filenames = []
    if oc_console_cli_download_link.endswith(".zip"):
        zip_file = zipfile.ZipFile(file=local_file_name)
        zip_file.extractall(path=tmpdir)
        extracted_filenames = zip_file.namelist()
    else:
        with tarfile.open(name=local_file_name, mode="r") as tar_file:
            tar_file.extractall(path=tmpdir)
            extracted_filenames = tar_file.getnames()
    LOGGER.info(f"Downloaded file: {extracted_filenames}")

    if len(extracted_filenames) > 1:
        raise ValueError(f"Multiple files found in {extracted_filenames}")
    # Remove the downloaded file
    if os.path.isfile(local_file_name):
        os.remove(local_file_name)
    binary_path = os.path.join(tmpdir, extracted_filenames[0])
    os.chmod(binary_path, stat.S_IRUSR | stat.S_IXUSR)
    return binary_path


def check_internal_image_registry_available(admin_client: DynamicClient) -> bool:
    """Check if internal image registry is available by checking the imageregistry config managementState"""
    try:
        # Access the imageregistry.operator.openshift.io/v1 Config resource named "cluster"
        config_instance = Config(client=admin_client, name="cluster")

        management_state = config_instance.instance.spec.get("managementState", "").lower()
        is_available = management_state == "managed"

        LOGGER.info(f"Image registry management state: {management_state}, available: {is_available}")
        return is_available
    except (ResourceNotFoundError, Exception) as e:
        LOGGER.warning(f"Failed to check image registry config: {e}")
        return False


def get_cluster_authentication(admin_client: DynamicClient) -> Authentication | None:
    auth = Authentication(client=admin_client, name="cluster")
    if auth.exists:
        return auth
    return None
