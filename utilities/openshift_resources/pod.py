"""Pod resource."""

from __future__ import annotations

from typing import Any

from utilities.openshift_resources._sync import _run_sync
from utilities.openshift_resources.namespace_scoped_resource import NamespaceScopedResource
from utilities.openshift_resources.oc import run_oc


class Pod(NamespaceScopedResource):
    """Pod is a collection of containers that can run on a host.

    This resource is created by clients and scheduled onto hosts.
    """

    # -- Async --

    async def _async_status(self) -> str:
        return (await self._async_instance()).status.phase or ""

    async def _async_ip(self) -> str:
        return (await self._async_instance()).status.podIP or ""

    async def _async_node(self) -> str:
        return (await self._async_instance()).spec.nodeName or ""

    # -- Sync --

    @property
    def status(self) -> str:
        return _run_sync(coro=self._async_status())

    def __init__(
        self,
        active_deadline_seconds: int | None = None,
        affinity: dict[str, Any] | None = None,
        automount_service_account_token: bool | None = None,
        containers: list[Any] | None = None,
        dns_config: dict[str, Any] | None = None,
        dns_policy: str | None = None,
        enable_service_links: bool | None = None,
        ephemeral_containers: list[Any] | None = None,
        host_aliases: list[Any] | None = None,
        host_ipc: bool | None = None,
        host_network: bool | None = None,
        host_pid: bool | None = None,
        host_users: bool | None = None,
        hostname: str | None = None,
        image_pull_secrets: list[Any] | None = None,
        init_containers: list[Any] | None = None,
        node_name: str | None = None,
        node_selector: dict[str, str] | None = None,
        os: dict[str, Any] | None = None,
        overhead: dict[str, Any] | None = None,
        preemption_policy: str | None = None,
        priority: int | None = None,
        priority_class_name: str | None = None,
        readiness_gates: list[Any] | None = None,
        resource_claims: list[Any] | None = None,
        resources: dict[str, Any] | None = None,
        restart_policy: str | None = None,
        runtime_class_name: str | None = None,
        scheduler_name: str | None = None,
        scheduling_gates: list[Any] | None = None,
        security_context: dict[str, Any] | None = None,
        service_account: str | None = None,
        service_account_name: str | None = None,
        set_hostname_as_fqdn: bool | None = None,
        share_process_namespace: bool | None = None,
        subdomain: str | None = None,
        termination_grace_period_seconds: int | None = None,
        tolerations: list[Any] | None = None,
        topology_spread_constraints: list[Any] | None = None,
        volumes: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            active_deadline_seconds (int): Optional duration in seconds
                the pod may be active on the node relative to StartTime
                before the system will actively try to mark it failed
                and kill associated containers.
            affinity (dict[str, Any]): If specified, the pod's scheduling
                constraints.
            automount_service_account_token (bool): Indicates whether a
                service account token should be automatically mounted.
            containers (list[Any]) (required): List of containers
                belonging to the pod. Containers cannot currently be
                added or removed. There must be at least one container
                in a Pod.
            dns_config (dict[str, Any]): Specifies the DNS parameters of
                a pod. Parameters specified here will be merged to the
                generated DNS configuration based on DNSPolicy.
            dns_policy (str): Set DNS policy for the pod. Defaults to
                "ClusterFirst". Valid values are
                'ClusterFirstWithHostNet', 'ClusterFirst', 'Default',
                'None'.
            enable_service_links (bool): Indicates whether information
                about services should be injected into pod's environment
                variables, matching the syntax of Docker links.
            ephemeral_containers (list[Any]): List of ephemeral containers
                run in this pod. Ephemeral containers may be run in an
                existing pod to perform user-initiated actions.
            host_aliases (list[Any]): HostAliases is an optional list of
                hosts and IPs that will be injected into the pod's hosts
                file if specified.
            host_ipc (bool): Use the host's ipc namespace. Optional:
                Default to false.
            host_network (bool): Host networking requested for this pod.
                Use the host's network namespace. If this option is set,
                the ports that will be used must be specified.
            host_pid (bool): Use the host's pid namespace. Optional:
                Default to false.
            host_users (bool): Use the host's user namespace. Optional:
                Default to true. If set to true or not present, the pod
                will be run in the host user namespace.
            hostname (str): Specifies the hostname of the Pod. If not
                specified, the pod's hostname will be set to a
                system-defined value.
            image_pull_secrets (list[Any]): Optional list of references
                to secrets in the same namespace to use for pulling any
                of the images used by this pod.
            init_containers (list[Any]): List of initialization containers
                belonging to the pod. Init containers are executed in
                order prior to containers being started.
            node_name (str): NodeName indicates in which node this pod is
                scheduled. If empty, this pod is a candidate for
                scheduling by the scheduler.
            node_selector (dict[str, str]): NodeSelector is a selector
                which must be true for the pod to fit on a node.
                Selector which must match a node's labels for the pod
                to be scheduled on that node.
            os (dict[str, Any]): Specifies the OS of the containers in
                the pod. Some pod and container fields are restricted
                if this is set.
            overhead (dict[str, Any]): Overhead represents the resource
                overhead associated with running a pod for a given
                RuntimeClass. This field will be auto-populated at
                admission time.
            preemption_policy (str): PreemptionPolicy is the Policy for
                preempting pods with lower priority. One of Never,
                PreemptLowerPriority. Defaults to PreemptLowerPriority.
            priority (int): The priority value. Various system components
                use this field to find the priority of the pod. When
                Priority Admission Controller is enabled, it prevents
                users from setting this field.
            priority_class_name (str): If specified, indicates the pod's
                priority. "system-node-critical" and
                "system-cluster-critical" are two special keywords
                which indicate the highest priorities.
            readiness_gates (list[Any]): If specified, all readiness
                gates will be evaluated for pod readiness. A pod is
                ready when all its containers are ready AND all
                conditions specified in the readiness gates have status
                equal to "True".
            resource_claims (list[Any]): ResourceClaims defines which
                ResourceClaims must be allocated and reserved before
                the Pod is allowed to start.
            resources (dict[str, Any]): Resources is the total amount of
                CPU and Memory resources required by all containers in
                the pod. It supports specifying Requests and Limits.
            restart_policy (str): Restart policy for all containers
                within the pod. One of Always, OnFailure, Never. In
                some contexts, only a subset of those values may be
                permitted.
            runtime_class_name (str): RuntimeClassName refers to a
                RuntimeClass object in the node.k8s.io group, which
                should be used to run this pod.
            scheduler_name (str): If specified, the pod will be
                dispatched by specified scheduler. If not specified,
                the pod will be dispatched by default scheduler.
            scheduling_gates (list[Any]): SchedulingGates is an opaque
                list of values that if specified will block scheduling
                the pod.
            security_context (dict[str, Any]): SecurityContext holds
                pod-level security attributes and common container
                settings. Optional: Defaults to empty.
            service_account (str): DeprecatedServiceAccount is a
                deprecated alias for ServiceAccountName. Deprecated:
                Use serviceAccountName instead.
            service_account_name (str): ServiceAccountName is the name
                of the ServiceAccount to use to run this pod.
            set_hostname_as_fqdn (bool): If true the pod's hostname
                will be configured as the pod's FQDN, rather than the
                leaf name (the default).
            share_process_namespace (bool): Share a single process
                namespace between all of the containers in a pod. When
                this is set containers will be able to view and signal
                processes from other containers in the pod.
            subdomain (str): If specified, the fully qualified Pod
                hostname will be
                "<hostname>.<subdomain>.<pod namespace>.svc.<cluster domain>".
            termination_grace_period_seconds (int): Optional duration in
                seconds the pod needs to terminate gracefully. May be
                decreased in delete request. Value must be non-negative.
            tolerations (list[Any]): If specified, the pod's tolerations.
            topology_spread_constraints (list[Any]):
                TopologySpreadConstraints describes how a group of pods
                ought to spread across topology domains. Scheduler will
                schedule pods in a way which abides by the constraints.
            volumes (list[Any]): List of volumes that can be mounted by
                containers belonging to the pod.
        """
        super().__init__(**kwargs)
        self.active_deadline_seconds = active_deadline_seconds
        self.affinity = affinity
        self.automount_service_account_token = automount_service_account_token
        self.containers = containers
        self.dns_config = dns_config
        self.dns_policy = dns_policy
        self.enable_service_links = enable_service_links
        self.ephemeral_containers = ephemeral_containers
        self.host_aliases = host_aliases
        self.host_ipc = host_ipc
        self.host_network = host_network
        self.host_pid = host_pid
        self.host_users = host_users
        self.hostname = hostname
        self.image_pull_secrets = image_pull_secrets
        self.init_containers = init_containers
        self.node_name = node_name
        self.node_selector = node_selector
        self.os = os
        self.overhead = overhead
        self.preemption_policy = preemption_policy
        self.priority = priority
        self.priority_class_name = priority_class_name
        self.readiness_gates = readiness_gates
        self.resource_claims = resource_claims
        self.resources = resources
        self.restart_policy = restart_policy
        self.runtime_class_name = runtime_class_name
        self.scheduler_name = scheduler_name
        self.scheduling_gates = scheduling_gates
        self.security_context = security_context
        self.service_account = service_account
        self.service_account_name = service_account_name
        self.set_hostname_as_fqdn = set_hostname_as_fqdn
        self.share_process_namespace = share_process_namespace
        self.subdomain = subdomain
        self.termination_grace_period_seconds = termination_grace_period_seconds
        self.tolerations = tolerations
        self.topology_spread_constraints = topology_spread_constraints
        self.volumes = volumes

    def _build_dict(self) -> dict[str, Any]:
        resource = super()._build_dict()

        spec: dict[str, Any] = {}
        if self.active_deadline_seconds is not None:
            spec["activeDeadlineSeconds"] = self.active_deadline_seconds
        if self.affinity is not None:
            spec["affinity"] = self.affinity
        if self.automount_service_account_token is not None:
            spec["automountServiceAccountToken"] = self.automount_service_account_token
        spec["containers"] = self.containers
        if self.dns_config is not None:
            spec["dnsConfig"] = self.dns_config
        if self.dns_policy is not None:
            spec["dnsPolicy"] = self.dns_policy
        if self.enable_service_links is not None:
            spec["enableServiceLinks"] = self.enable_service_links
        if self.ephemeral_containers is not None:
            spec["ephemeralContainers"] = self.ephemeral_containers
        if self.host_aliases is not None:
            spec["hostAliases"] = self.host_aliases
        if self.host_ipc is not None:
            spec["hostIPC"] = self.host_ipc
        if self.host_network is not None:
            spec["hostNetwork"] = self.host_network
        if self.host_pid is not None:
            spec["hostPID"] = self.host_pid
        if self.host_users is not None:
            spec["hostUsers"] = self.host_users
        if self.hostname is not None:
            spec["hostname"] = self.hostname
        if self.image_pull_secrets is not None:
            spec["imagePullSecrets"] = self.image_pull_secrets
        if self.init_containers is not None:
            spec["initContainers"] = self.init_containers
        if self.node_name is not None:
            spec["nodeName"] = self.node_name
        if self.node_selector is not None:
            spec["nodeSelector"] = self.node_selector
        if self.os is not None:
            spec["os"] = self.os
        if self.overhead is not None:
            spec["overhead"] = self.overhead
        if self.preemption_policy is not None:
            spec["preemptionPolicy"] = self.preemption_policy
        if self.priority is not None:
            spec["priority"] = self.priority
        if self.priority_class_name is not None:
            spec["priorityClassName"] = self.priority_class_name
        if self.readiness_gates is not None:
            spec["readinessGates"] = self.readiness_gates
        if self.resource_claims is not None:
            spec["resourceClaims"] = self.resource_claims
        if self.resources is not None:
            spec["resources"] = self.resources
        if self.restart_policy is not None:
            spec["restartPolicy"] = self.restart_policy
        if self.runtime_class_name is not None:
            spec["runtimeClassName"] = self.runtime_class_name
        if self.scheduler_name is not None:
            spec["schedulerName"] = self.scheduler_name
        if self.scheduling_gates is not None:
            spec["schedulingGates"] = self.scheduling_gates
        if self.security_context is not None:
            spec["securityContext"] = self.security_context
        if self.service_account is not None:
            spec["serviceAccount"] = self.service_account
        if self.service_account_name is not None:
            spec["serviceAccountName"] = self.service_account_name
        if self.set_hostname_as_fqdn is not None:
            spec["setHostnameAsFQDN"] = self.set_hostname_as_fqdn
        if self.share_process_namespace is not None:
            spec["shareProcessNamespace"] = self.share_process_namespace
        if self.subdomain is not None:
            spec["subdomain"] = self.subdomain
        if self.termination_grace_period_seconds is not None:
            spec["terminationGracePeriodSeconds"] = self.termination_grace_period_seconds
        if self.tolerations is not None:
            spec["tolerations"] = self.tolerations
        if self.topology_spread_constraints is not None:
            spec["topologySpreadConstraints"] = self.topology_spread_constraints
        if self.volumes is not None:
            spec["volumes"] = self.volumes
        if spec:
            resource["spec"] = spec

        return resource

    # -- Async: log / execute --

    async def _async_log(
        self,
        container: str | None = None,
        tail: int | None = None,
        follow: bool = False,
        since: str | None = None,
        since_time: str | None = None,
        timestamps: bool = False,
    ) -> str:
        args = ["logs", self.name, "-n", self.namespace]
        if container:
            args.extend(["-c", container])
        if tail is not None:
            args.extend(["--tail", str(tail)])
        if follow:
            args.append("-f")
        if since:
            args.extend(["--since", since])
        if since_time:
            args.extend(["--since-time", since_time])
        if timestamps:
            args.append("--timestamps")
        return (await run_oc(args=args)).stdout

    async def _async_execute(self, command: list[str], container: str | None = None, ignore_rc: bool = False) -> str:
        args = ["exec", self.name, "-n", self.namespace]
        if container:
            args.extend(["-c", container])
        args.append("--")
        args.extend(command)
        return (await run_oc(args=args, check=not ignore_rc)).stdout

    # -- Sync: ip, node, log, execute --

    def ip(self) -> str:
        return _run_sync(coro=self._async_ip())

    def node(self) -> str:
        return _run_sync(coro=self._async_node())

    def log(self, **kwargs: Any) -> str:
        return _run_sync(coro=self._async_log(**kwargs))

    def execute(self, command: list[str], container: str | None = None, ignore_rc: bool = False) -> str:
        return _run_sync(coro=self._async_execute(command=command, container=container, ignore_rc=ignore_rc))
