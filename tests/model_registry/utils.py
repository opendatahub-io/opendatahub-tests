from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from ocp_resources.model_registry import ModelRegistry
from kubernetes.dynamic.exceptions import ResourceNotFoundError

from utilities.exceptions import ProtocolNotSupportedError, TooManyServicesError
from utilities.constants import Protocols


ADDRESS_ANNOTATION_PREFIX: str = "routing.opendatahub.io/external-address-"


def get_mr_service_by_label(client: DynamicClient, ns: Namespace, mr_instance: ModelRegistry) -> Service:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        ns (Namespace): Namespace object where to find the Service
        mr_instance (ModelRegistry): Model Registry instance

    Returns:
        Service: The matching Service

    Raises:
        ResourceNotFoundError: if no service is found.
    """
    if svc := [
        svcs
        for svcs in Service.get(
            dyn_client=client,
            namespace=ns.name,
            label_selector=f"app={mr_instance.name},component=model-registry",
        )
    ]:
        if len(svc) == 1:
            return svc[0]
        raise TooManyServicesError(svc)
    raise ResourceNotFoundError(f"{mr_instance.name} has no Service")


def get_endpoint_from_mr_service(client: DynamicClient, svc: Service, protocol: str) -> str:
    if protocol in (Protocols.REST, Protocols.GRPC):
        return svc.instance.metadata.annotations[f"{ADDRESS_ANNOTATION_PREFIX}{protocol}"]
    else:
        raise ProtocolNotSupportedError(protocol)
