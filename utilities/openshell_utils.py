from kubernetes.dynamic import DynamicClient
from ocp_resources.ingress_config_openshift_io import Ingress
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler

from utilities.constants import Timeout


def get_cluster_apps_domain(admin_client: DynamicClient) -> str:
    ingress = Ingress(client=admin_client, name="cluster", ensure_exists=True)
    return ingress.instance.spec.domain


def wait_for_openshell_gateway_pod(client: DynamicClient, namespace: str, timeout: int = Timeout.TIMEOUT_5MIN) -> None:
    def _get_gateway_pods() -> list[Pod]:
        # Label selector taken from standard Helm chart conventions; verify against the
        # actual openshell-helm-chart templates once available.
        return [
            _pod
            for _pod in Pod.get(
                client=client,
                namespace=namespace,
                label_selector="app.kubernetes.io/name=openshell",
            )
        ]

    sampler = TimeoutSampler(wait_timeout=timeout, sleep=1, func=lambda: bool(_get_gateway_pods()))

    for sample in sampler:
        if sample:
            break

    pods = _get_gateway_pods()
    for pod in pods:
        pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status="True",
        )
