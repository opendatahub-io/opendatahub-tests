# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/scripts/resource/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource


class Route(NamespacedResource):
    """
    A route allows developers to expose services through an HTTP(S) aware load balancing and proxy layer via a public
    DNS entry. The route may further specify TLS options and a certificate, or specify a public CNAME that the router
    should also accept for HTTP and HTTPS traffic. An administrator typically configures their router to be visible
    outside the cluster firewall, and may also add additional security, caching, or traffic controls on the service
    content. Routers usually talk directly to the service endpoints.

    Once a route is created, the `host` field may not be changed. Generally, routers use the oldest route with a given
    host when resolving conflicts.

    Routers are subject to additional customization and may support additional controls via the annotations field.

    Because administrators may configure multiple routers, the route status field is used to return information to
    clients about the names and states of the route under each router. If a client chooses a duplicate name, for
    instance, the route status conditions are used to indicate the route cannot be chosen.

    To enable HTTP/2 ALPN on a route it requires a custom (non-wildcard) certificate. This prevents connection
    coalescing by clients, notably web browsers. We do not support HTTP/2 ALPN on routes that use the default
    certificate because of the risk of connection re-use/coalescing. Routes that do not have their own custom
    certificate will not be HTTP/2 ALPN-enabled on either the frontend or the backend.

    Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is
    longer).
    """

    api_group: str = NamespacedResource.ApiGroup.ROUTE_OPENSHIFT_IO

    def __init__(
        self,
        alternate_backends: list[Any] | None = None,
        host: str | None = None,
        http_headers: dict[str, Any] | None = None,
        path: str | None = None,
        port: dict[str, Any] | None = None,
        subdomain: str | None = None,
        tls: dict[str, Any] | None = None,
        to: dict[str, Any] | None = None,
        wildcard_policy: str | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            alternate_backends (list[Any]): alternateBackends allows up to 3 additional backends to be assigned to
              the route. Only the Service kind is allowed, and it will be
              defaulted to Service. Use the weight field in RouteTargetReference
              object to specify relative preference.

            host (str): host is an alias/DNS that points to the service. Optional. If not
              specified a route name will typically be automatically chosen.
              Must follow DNS952 subdomain conventions.

            http_headers (dict[str, Any]): RouteHTTPHeaders defines policy for HTTP headers.

            path (str): path that the router watches for, to route traffic for to the service.
              Optional

            port (dict[str, Any]): RoutePort defines a port mapping from a router to an endpoint in the
              service endpoints.

            subdomain (str): subdomain is a DNS subdomain that is requested within the ingress
              controller's domain (as a subdomain). If host is set this field is
              ignored. An ingress controller may choose to ignore this suggested
              name, in which case the controller will report the assigned name
              in the status.ingress array or refuse to admit the route. If this
              value is set and the server does not support this field host will
              be populated automatically. Otherwise host is left empty. The
              field may have multiple parts separated by a dot, but not all
              ingress controllers may honor the request. This field may not be
              changed after creation except by a user with the update
              routes/custom-host permission.  Example: subdomain `frontend`
              automatically receives the router subdomain `apps.mycluster.com`
              to have a full hostname `frontend.apps.mycluster.com`.

            tls (dict[str, Any]): TLSConfig defines config used to secure a route and provide
              termination

            to (dict[str, Any]): RouteTargetReference specifies the target that resolve into endpoints.
              Only the 'Service' kind is allowed. Use 'weight' field to
              emphasize one over others.

            wildcard_policy (str): Wildcard policy if any for the route. Currently only 'Subdomain' or
              'None' is allowed.

        """
        super().__init__(**kwargs)

        self.alternate_backends = alternate_backends
        self.host = host
        self.http_headers = http_headers
        self.path = path
        self.port = port
        self.subdomain = subdomain
        self.tls = tls
        self.to = to
        self.wildcard_policy = wildcard_policy

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.to is None:
                raise MissingRequiredArgumentError(argument="self.to")

            self.res["spec"] = {}
            _spec = self.res["spec"]

            _spec["to"] = self.to

            if self.alternate_backends is not None:
                _spec["alternateBackends"] = self.alternate_backends

            if self.host is not None:
                _spec["host"] = self.host

            if self.http_headers is not None:
                _spec["httpHeaders"] = self.http_headers

            if self.path is not None:
                _spec["path"] = self.path

            if self.port is not None:
                _spec["port"] = self.port

            if self.subdomain is not None:
                _spec["subdomain"] = self.subdomain

            if self.tls is not None:
                _spec["tls"] = self.tls

            if self.wildcard_policy is not None:
                _spec["wildcardPolicy"] = self.wildcard_policy

    # End of generated code
