import json
import re
import shlex
from contextlib import contextmanager
from functools import cache
from typing import Any, Generator, Optional, Set

import kubernetes
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.catalog_source import CatalogSource
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.dsc_initialization import DSCInitialization
from ocp_resources.exceptions import MissingResourceError
from ocp_resources.inference_service import InferenceService
from ocp_resources.infrastructure import Infrastructure
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.project_project_openshift_io import Project
from ocp_resources.project_request import ProjectRequest
from ocp_resources.resource import ResourceEditor, get_client
from ocp_resources.role import Role
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pyhelper_utils.shell import run_command
from pytest_testconfig import config as py_config
from semver import Version
from simple_logger.logger import get_logger

from utilities.constants import ApiGroups, Labels, Timeout
from utilities.constants import KServeDeploymentType
from utilities.constants import Annotations
from utilities.exceptions import ClusterLoginError, FailedPodsError
from timeout_sampler import TimeoutExpiredError, TimeoutSampler, retry
import utilities.general

LOGGER = get_logger(name=__name__)


@contextmanager
def create_ns(
    name: str | None = None,
    admin_client: DynamicClient | None = None,
    unprivileged_client: DynamicClient | None = None,
    teardown: bool = True,
    delete_timeout: int = Timeout.TIMEOUT_4MIN,
    labels: dict[str, str] | None = None,
    ns_annotations: dict[str, str] | None = None,
    model_mesh_enabled: bool = False,
    add_dashboard_label: bool = False,
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

    Yields:
        Namespace | Project: namespace or project

    """
    if pytest_request:
        name = pytest_request.param.get("name", name)
        ns_annotations = pytest_request.param.get("annotations", ns_annotations)
        model_mesh_enabled = pytest_request.param.get("modelmesh-enabled", model_mesh_enabled)
        add_dashboard_label = pytest_request.param.get("add-dashboard-label", add_dashboard_label)

    namespace_kwargs = {
        "name": name,
        "client": admin_client,
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

    if unprivileged_client:
        with ProjectRequest(name=name, client=unprivileged_client, teardown=teardown):
            project = Project(**namespace_kwargs)
            project.wait_for_status(status=project.Status.ACTIVE, timeout=Timeout.TIMEOUT_2MIN)
            yield project

            wait_for_serverless_pods_deletion(resource=project, admin_client=admin_client)

    else:
        with Namespace(**namespace_kwargs) as ns:
            ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=Timeout.TIMEOUT_2MIN)
            yield ns

            wait_for_serverless_pods_deletion(resource=ns, admin_client=admin_client)


def wait_for_replicas_in_deployment(deployment: Deployment, replicas: int) -> None:
    """
    Wait for replicas in deployment to updated in spec.

    Args:
        deployment (Deployment): Deployment object
        replicas (int): number of replicas to be set in spec.replicas

    Raises:
        TimeoutExpiredError: If replicas are not updated in spec.

    """
    _replicas: int | None = None

    try:
        for sample in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_2MIN,
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
    timeout: int = Timeout.TIMEOUT_5MIN,
) -> list[Deployment]:
    """
    Wait for inference deployment replicas to complete.

    Args:
        client (DynamicClient): Dynamic client.
        isvc (InferenceService): InferenceService object
        runtime_name (str): ServingRuntime name.
        expected_num_deployments (int): Expected number of deployments per InferenceService.
        timeout (int): Time to wait for the model deployment.

    Returns:
        list[Deployment]: List of Deployment objects for InferenceService.

    """
    ns = isvc.namespace
    label_selector = utilities.general.create_isvc_label_selector_str(
        isvc=isvc, resource_type="deployment", runtime_name=runtime_name
    )

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
                # Raw deployment: if min replicas is more than 1, wait for min replicas
                # to be set in deployment spec by HPA
                if (
                    isvc.instance.metadata.annotations.get("serving.kserve.io/deploymentMode")
                    == KServeDeploymentType.RAW_DEPLOYMENT
                ):
                    wait_for_replicas_in_deployment(
                        deployment=deployments[0],
                        replicas=isvc.instance.spec.predictor.get("minReplicas", 1),
                    )

                deployment.wait_for_replicas(timeout=timeout)

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
    secret_kwargs = {"client": admin_client, "name": name, "namespace": namespace}
    secret = Secret(**secret_kwargs)

    if secret.exists:
        LOGGER.info(f"Secret {name} already exists in namespace {namespace}")
        yield secret

    else:
        with Secret(
            annotations={f"{ApiGroups.OPENDATAHUB_IO}/connection-type": "s3"},
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
            **secret_kwargs,
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
    label_selector = utilities.general.create_isvc_label_selector_str(
        isvc=isvc, resource_type="pod", runtime_name=runtime_name
    )

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
                        ) and wait_state.reason in (
                            pod.Status.IMAGE_PULL_BACK_OFF,
                            pod.Status.CRASH_LOOPBACK_OFF,
                            pod.Status.ERR_IMAGE_PULL,
                            "InvalidImageName",
                        )

                        is_terminated_error = (
                            terminate_state := container_status.state.terminated
                        ) and terminate_state.reason in (
                            pod.Status.ERROR,
                            pod.Status.CRASH_LOOPBACK_OFF,
                        )

                        if is_waiting_pull_back_off or is_terminated_error:
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
    for csv in ClusterServiceVersion.get(dyn_client=admin_client, namespace=py_config["applications_namespace"]):
        if re.match("rhods|opendatahub", csv.name):
            operator_version = csv.instance.spec.version
            break

    if not operator_version:
        raise MissingResourceError("Operator ClusterServiceVersion not found")

    return Version.parse(operator_version)


def get_dsci_applications_namespace(client: DynamicClient, dsci_name: str = "default-dsci") -> str:
    """
    Get the namespace where DSCI applications are deployed.

    Args:
        client (DynamicClient): DynamicClient object
        dsci_name (str): DSCI name

    Returns:
        str: Namespace where DSCI applications are deployed.

    Raises:
            ValueError: If DSCI applications namespace not found
            MissingResourceError: If DSCI not found

    """
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
        str: Operator distribution.

    Raises:
            ValueError: If DSC release name not found
            MissingResourceError: If DSC not found

    """
    dsc = DataScienceCluster(client=client, name=dsc_name)

    if dsc.exists:
        if dsc_release_name := dsc.instance.status.get("release", {}).get("name"):
            return dsc_release_name

        else:
            raise ValueError("DSC release name not found in {dsc_name}")

    raise MissingResourceError(f"DSC {dsc_name} not found")


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
    for pod in Pod.get(dyn_client=client, namespace=resource.name):
        if (
            pod.exists
            and pod.instance.metadata.annotations.get(Annotations.KserveIo.DEPLOYMENT_MODE)
            == KServeDeploymentType.SERVERLESS
        ):
            LOGGER.info(f"Waiting for {KServeDeploymentType.SERVERLESS} pod {pod.name} to be deleted")
            pod.wait_deleted(timeout=Timeout.TIMEOUT_1MIN)


@retry(wait_timeout=Timeout.TIMEOUT_30SEC, sleep=1, exceptions_dict={ResourceNotFoundError: []})
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
