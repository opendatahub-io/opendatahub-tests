from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.pod import Pod
from ocp_resources.resource import NamespacedResource
from simple_logger.logger import get_logger

KEYS_TO_VALIDATE = ["runAsGroup", "runAsUser", "allowPrivilegeEscalation", "capabilities"]

LOGGER = get_logger(name=__name__)


def get_uid_from_namespace(namespace_scc: dict[str, str]) -> str:
    return namespace_scc["uid-range"].split("/")[0]


def validate_pod_security_context(
    pod_security_context: dict[str, Any],
    namespace_scc: dict[str, str],
    model_registry_pod: NamespacedResource,
    ns_uid: str,
) -> list[str]:
    """
    Check model registry pod, ensure the security context values are being set by openshift
    """
    errors = []
    pod_selinux_option = pod_security_context.get("seLinuxOptions", {}).get("level")
    fs_group = pod_security_context.get("fsGroup")
    LOGGER.info(f"pod_selinux_option: {pod_selinux_option}")
    if pod_selinux_option != namespace_scc["seLinuxOptions"]:
        errors.append(
            f"selinux option from pod {model_registry_pod.name} {pod_selinux_option},"
            f" namespace: {namespace_scc['seLinuxOptions']}"
        )
    LOGGER.info(f"fs_group: {fs_group}")
    if fs_group != int(ns_uid):
        errors.append(
            f"UID-range from pod {model_registry_pod.name} {pod_security_context.get('fsGroup')}, namespace: {ns_uid}"
        )
    return errors


def validate_containers_pod_security_context(model_registry_pod: Pod, namespace_uid: str) -> list[str]:
    """
    Check all the containers of model registry pod, ensure the security context values are being set by openshift
    """
    errors = []
    containers = model_registry_pod.instance.spec.containers
    for container in containers:
        uid = int(namespace_uid) + 1
        expected_value = {
            "runAsUser": uid if container.args and "sidecar" in container.args else int(namespace_uid),
            "runAsGroup": uid if container.args and "sidecar" in container.args else None,
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        }

        for key in KEYS_TO_VALIDATE:
            LOGGER.info(f"key: {container.securityContext.get(key)}, type: {type(container.securityContext.get(key))}")
            field_value = container.securityContext.get(key)
            if key == "capabilities" and field_value:
                field_value = field_value.to_dict()
            if field_value == expected_value[key]:
                LOGGER.info(
                    f"For container: {container.name}, {key} validation: {expected_value[key]} completed successfully"
                )
            else:
                errors.append(
                    f"For {container.name}, expected key {key} value: {expected_value[key]},"
                    f" actual: {container.securityContext.get(key)}"
                )
    return errors


def get_pod_by_deployment_name(admin_client: DynamicClient, namespace: str, deployment_name: str) -> Pod:
    """
    Get a pod by deployment name. First ensures the deployment exists, then finds its associated pod.

    Args:
        admin_client: The admin client for Kubernetes operations
        namespace: The namespace to search in
        deployment_name: The name of the deployment to find pods for

    Returns:
        Pod: The pod associated with the deployment

    Raises:
        AssertionError: If exactly one pod is not found
    """
    # First ensure the deployment exists
    deployment = Deployment(client=admin_client, name=deployment_name, namespace=namespace, ensure_exists=True)
    deployment_instance = deployment.instance

    # Get pods using the deployment's label selector
    label_selector = ",".join([f"{k}={v}" for k, v in deployment_instance.spec.selector.matchLabels.items()])
    pods = list(
        Pod.get(
            client=admin_client,
            namespace=namespace,
            label_selector=label_selector,
        )
    )
    expected_replicas = deployment_instance.status.replicas
    if len(pods) != expected_replicas:
        raise AssertionError(
            f"Expected {expected_replicas} pod(s) for deployment '{deployment_name}'. "
            f"Found {len(pods)} pods: {[pod.name for pod in pods] if pods else None}"
        )
    return pods[0]


def validate_deployment_scc(deployment: Deployment) -> None:
    """
    Validate that deployment does not set runAsUser/runAsGroup in container security contexts.

    Args:
        deployment: The deployment object to validate

    Raises:
        AssertionError: If any container has runAsUser or runAsGroup set
    """
    LOGGER.info(f"Validating security context for deployment: {deployment.name}")

    error = []
    for container in deployment.instance.spec.template.spec.containers:
        container_security_context = container.securityContext
        LOGGER.info(f"Container security context validation for {container.name}")
        if not container_security_context:
            LOGGER.info(f"No container security context exists for {container.name}")
        else:
            if not all(True for key in ["runAsGroup", "runAsUser"] if not container_security_context.get(key)):
                error.append({container.name: container.securityContext})

    if error:
        raise AssertionError(
            f"{deployment.name} {deployment.kind} containers expected to not set {KEYS_TO_VALIDATE}, actual: {error}"
        )


def validate_pod_scc(pod: Pod, model_registry_scc_namespace: dict[str, str]) -> None:
    """
    Validate that pod gets runAsUser/runAsGroup from openshift and the values match namespace annotations.

    Args:
        pod: The pod object to validate
        model_registry_scc_namespace: Dictionary containing SCC namespace information

    Raises:
        AssertionError: If pod security context validation fails
    """
    LOGGER.info(f"Validating pod security context for pod: {pod.name}")
    ns_uid = get_uid_from_namespace(namespace_scc=model_registry_scc_namespace)
    pod_spec = pod.instance.spec
    errors = validate_pod_security_context(
        pod_security_context=pod_spec.securityContext,
        namespace_scc=model_registry_scc_namespace,
        model_registry_pod=pod,
        ns_uid=ns_uid,
    )
    errors.extend(validate_containers_pod_security_context(model_registry_pod=pod, namespace_uid=ns_uid))
    if errors:
        raise AssertionError(f"{pod.name} {pod.kind} pod security context validation failed with error: {errors}")
