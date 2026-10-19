"""Shared helpers for pre/post-upgrade fixture handling.

Pre-upgrade fixtures persist their resources on purpose: post-upgrade tests
consume them, and post-upgrade owns teardown. Nothing cleans up when an
upgrade cycle dies between the phases, so the next pre-upgrade run collides
with leftover fixed-name resources (409 Conflict / AlreadyExists).

`purge_upgrade_leftover` makes pre-upgrade fixtures self-healing without
owning their creation logic:

* post-upgrade mode: no-op (fixtures adopt existing resources).
* pre-upgrade mode, resource absent: no-op (the fixture creates it).
* pre-upgrade mode, resource present: delete it (bounded wait) so the
  fixture re-creates a fresh one. If deletion times out (typically a
  finalizer no controller services, e.g. after an operator swap), skip with
  a clear reason instead of raising a stacktrace wall.

Deletion only ever targets the exact resource the caller declares; there is
no prefix globbing (namespace conventions differ across suites).
"""

from collections.abc import Callable
from typing import Any

import pytest
import structlog
from ocp_resources.resource import Resource

LOGGER = structlog.get_logger(name=__name__)

DEFAULT_DELETE_TIMEOUT = 300


def delete_upgrade_resource(resource: Resource, timeout: int) -> bool:
    """Delete `resource` and verify it is gone within `timeout` seconds.

    Returns True when the resource vanished, False when deletion is blocked
    (resource still exists / stuck Terminating). Never waits unbounded.
    """
    LOGGER.info(f"[upgrade-idem] purging leftover {resource.kind} {resource.namespace}/{resource.name}")
    deleted = resource.delete(wait=True, timeout=timeout)
    return deleted and not resource.exists


def upgrade_resource_skip_reason(
    resource: Resource,
    timeout: int,
    skip_context: str,
) -> str:
    namespace_hint = f" -n {resource.namespace}" if resource.namespace else ""
    return (
        f"{skip_context}: leftover {resource.kind} '{resource.namespace}/{resource.name}' could not be deleted "
        f"within {timeout}s (likely a finalizer no controller is servicing). "
        f"Manual cleanup required: `oc delete {resource.kind.lower()} {resource.name}{namespace_hint} --wait`, "
        f"then re-run pre-upgrade."
    )


def purge_upgrade_leftover(
    pytestconfig: Any,
    resource_factory: Callable[..., Resource],
    delete_timeout: int = DEFAULT_DELETE_TIMEOUT,
    skip_context: str = "pre-upgrade",
) -> None:
    """Purge a leftover fixed-name resource before a pre-upgrade fixture creates it.

    No-op in post-upgrade mode and when the resource does not exist. When the
    resource exists, deletes it with a bounded wait; on a blocked deletion,
    raises `pytest.skip` naming the resource and the manual remedy.

    Call this at the top of the pre-upgrade branch of a fixed-name fixture,
    before the fixture's own creation logic. For namespaced resources whose
    namespace is also fixture-managed, the namespace fixture must purge the
    namespace; CR fixtures purge their CR. A namespace whose deletion is
    blocked by finalized CRs inside it skips at the namespace level.

    Args:
        pytestconfig: pytest config; `option.post_upgrade` selects the mode.
        resource_factory: constructor taking the identity kwargs (client,
            name, namespace, ...) and returning a NOT-yet-created Resource.
        delete_timeout: bounded wait for leftover deletion, seconds.
        skip_context: suite hint for the skip reason, e.g. "MaaS upgrade".
    """
    if pytestconfig.option.post_upgrade:
        return

    resource = resource_factory()
    if not resource.exists:
        return

    if not delete_upgrade_resource(resource=resource, timeout=delete_timeout):
        pytest.skip(
            upgrade_resource_skip_reason(resource=resource, timeout=delete_timeout, skip_context=skip_context)
        )

    LOGGER.info(f"[upgrade-idem] leftover purged, fixture will re-create {resource.kind} {resource.name} fresh")


def upgrade_resource_factory(
    resource_class: type[Resource],
    **identity_kwargs: Any,
) -> Callable[..., Resource]:
    """Bind a resource class to its identity kwargs (client, name, namespace).

    Usage: `upgrade_resource_factory(Namespace, client=admin_client, name="upgrade-maas")`
    returns a zero-arg callable suitable for `purge_upgrade_leftover`.
    """

    def factory(**extra: Any) -> Resource:
        return resource_class(**{**identity_kwargs, **extra})

    return factory
