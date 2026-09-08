# Generated using https://github.com/RedHatQE/openshift-python-wrapper/blob/main/class_generator/README.md


from typing import Any

from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.resource import NamespacedResource


class ServiceEntry(NamespacedResource):
    """
    No field description from API
    """

    api_group: str = NamespacedResource.ApiGroup.NETWORKING_ISTIO_IO

    def __init__(
        self,
        addresses: list[Any] | None = None,
        endpoints: list[Any] | None = None,
        export_to: list[Any] | None = None,
        hosts: list[Any] | None = None,
        location: str | None = None,
        ports: list[Any] | None = None,
        resolution: str | None = None,
        subject_alt_names: list[Any] | None = None,
        workload_selector: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            addresses (list[Any]): The virtual IP addresses associated with the service.

            endpoints (list[Any]): One or more endpoints associated with the service.

            export_to (list[Any]): A list of namespaces to which this service is exported.

            hosts (list[Any]): The hosts associated with the ServiceEntry.

            location (str): Specify whether the service should be considered external to the mesh
              or part of the mesh.  Valid Options: MESH_EXTERNAL, MESH_INTERNAL

            ports (list[Any]): The ports associated with the external service.

            resolution (str): Service resolution mode for the hosts.  Valid Options: NONE, STATIC,
              DNS, DNS_ROUND_ROBIN, DYNAMIC_DNS

            subject_alt_names (list[Any]): If specified, the proxy will verify that the server certificate's
              subject alternate name matches one of the specified values.

            workload_selector (dict[str, Any]): Applicable only for MESH_INTERNAL services.

        """
        super().__init__(**kwargs)

        self.addresses = addresses
        self.endpoints = endpoints
        self.export_to = export_to
        self.hosts = hosts
        self.location = location
        self.ports = ports
        self.resolution = resolution
        self.subject_alt_names = subject_alt_names
        self.workload_selector = workload_selector

    def to_dict(self) -> None:

        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            if self.hosts is None:
                raise MissingRequiredArgumentError(argument="self.hosts")


            self.res["spec"] = {}
            _spec = self.res["spec"]


            _spec["hosts"] = self.hosts


            if self.addresses is not None:
                _spec["addresses"] = self.addresses

            if self.endpoints is not None:
                _spec["endpoints"] = self.endpoints

            if self.export_to is not None:
                _spec["exportTo"] = self.export_to

            if self.location is not None:
                _spec["location"] = self.location

            if self.ports is not None:
                _spec["ports"] = self.ports

            if self.resolution is not None:
                _spec["resolution"] = self.resolution

            if self.subject_alt_names is not None:
                _spec["subjectAltNames"] = self.subject_alt_names

            if self.workload_selector is not None:
                _spec["workloadSelector"] = self.workload_selector


    # End of generated code
