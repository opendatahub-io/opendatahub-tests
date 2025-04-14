from typing import Any
from simple_logger.logger import get_logger

from ocp_resources.pod import Pod
from ocp_resources.resource import NamespacedResource

KEYS_TO_VALIDATE = ["runAsGroup", "runAsUser"]

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
    if pod_selinux_option != namespace_scc["seLinuxOptions"]:
        errors.append(
            f"selinux option from pod {model_registry_pod.name} {pod_selinux_option},"
            f" namespace: {namespace_scc['seLinuxOptions']}"
        )
    if pod_security_context.get("fsGroup") != int(ns_uid):
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
        expected_value = {
            "runAsUser": int(namespace_uid) + 1 if "sidecar" in container.args else int(namespace_uid),
            "runAsGroup": int(namespace_uid) + 1 if "sidecar" in container.args else None,
        }

        for key in KEYS_TO_VALIDATE:
            if container.securityContext.get(key) == expected_value[key]:
                LOGGER.info(
                    f"For container: {container.name}, {key} validation: {expected_value[key]} completed successfully"
                )
            else:
                errors.append(
                    f"For {container.name}, expected key {key} value: {expected_value[key]},"
                    f" actual: {container.securityContext.get(key)}"
                )
    return errors
