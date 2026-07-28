"""ocp_resources — interact with OpenShift clusters via the oc binary."""

import logging

import structlog

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

from utilities.openshift_resources.cluster_scoped_resource import ClusterScopedResource  # noqa: E402
from utilities.openshift_resources.namespace_scoped_resource import NamespaceScopedResource  # noqa: E402

__all__ = ["ClusterScopedResource", "NamespaceScopedResource"]
