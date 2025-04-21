import pytest
from typing import Self
from simple_logger.logger import get_logger
from _pytest.fixtures import FixtureRequest

from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.deployment import Deployment
from tests.model_registry.scc.utils import (
    get_uid_from_namespace,
    validate_pod_security_context,
    KEYS_TO_VALIDATE,
    validate_containers_pod_security_context,
)
from utilities.constants import DscComponents
from tests.model_registry.constants import MR_NAMESPACE, MODEL_DICT, MR_INSTANCE_NAME

from kubernetes.dynamic import DynamicClient
from ocp_utilities.infra import get_pods_by_name_prefix

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def model_registry_scc_namespace(model_registry_namespace: str):
    mr_annotations = Namespace(name=model_registry_namespace).instance.metadata.annotations
    return {
        "seLinuxOptions": mr_annotations.get("openshift.io/sa.scc.mcs"),
        "uid-range": mr_annotations.get("openshift.io/sa.scc.uid-range"),
    }


@pytest.fixture(scope="class")
def model_registry_resource(
    request: FixtureRequest, admin_client: DynamicClient, model_registry_namespace: str
) -> Deployment | Pod:
    if request.param["kind"] == Deployment:
        return Deployment(name=MR_INSTANCE_NAME, namespace=model_registry_namespace, ensure_exists=True)
    elif request.param["kind"] == Pod:
        pods = get_pods_by_name_prefix(
            client=admin_client, pod_prefix=MR_INSTANCE_NAME, namespace=model_registry_namespace
        )
        if len(pods) != 1:
            pytest.fail(
                "Expected one model registry pod. Found: {[{pod.name: pod.status} for pod in pods] if pods else None}"
            )
        return pods[0]
    else:
        raise AssertionError(f"Invalid resource: {request.param['kind']}. Valid options: Deployment and Pod")


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class, registered_model",
    [
        pytest.param(
            {
                "component_patch": {
                    DscComponents.MODELREGISTRY: {
                        "managementState": DscComponents.ManagementState.MANAGED,
                        "registriesNamespace": MR_NAMESPACE,
                    },
                },
            },
            MODEL_DICT,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class", "registered_model")
class TestModelRegistrySecurityContextValidation:
    @pytest.mark.parametrize(
        "model_registry_resource",
        [
            pytest.param({"kind": Deployment}),
        ],
        indirect=["model_registry_resource"],
    )
    def test_model_registry_deployment_security_context_validation(
        self: Self,
        model_registry_resource: Deployment,
    ):
        """
        Validate that model registry deployment does not set runAsUser/runAsGroup
        """
        error = []
        for container in model_registry_resource.instance.spec.template.spec.containers:
            if not all([True for key in KEYS_TO_VALIDATE if not container.get(key)]):
                error.append({container.name: container.securityContext})

        if error:
            pytest.fail(
                f"{model_registry_resource.name} {model_registry_resource.kind} containers expected to not "
                f"set {KEYS_TO_VALIDATE}, actual: {error}"
            )

    @pytest.mark.parametrize(
        "model_registry_resource",
        [
            pytest.param({"kind": Pod}),
        ],
        indirect=["model_registry_resource"],
    )
    def test_model_registry_pod_security_context_validation(
        self: Self,
        model_registry_resource: Pod,
        model_registry_scc_namespace: dict[str, str],
    ):
        """
        Validate that model registry pod gets runAsUser/runAsGroup from openshift and the values matches namespace
        annotations
        """
        ns_uid = get_uid_from_namespace(namespace_scc=model_registry_scc_namespace)
        pod_spec = model_registry_resource.instance.spec
        errors = validate_pod_security_context(
            pod_security_context=pod_spec.securityContext,
            namespace_scc=model_registry_scc_namespace,
            model_registry_pod=model_registry_resource,
            ns_uid=ns_uid,
        )
        errors.extend(
            validate_containers_pod_security_context(model_registry_pod=model_registry_resource, namespace_uid=ns_uid)
        )
        if errors:
            pytest.fail(
                f"{model_registry_resource.name} {model_registry_resource.kind} pod security context validation failed"
                f" with error: {errors}"
            )
