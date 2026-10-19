"""Behavior-matrix tests for utilities.upgrade_utils against a real (kind) cluster.

Covers: purge-and-recreate on leftovers, skip on finalizer-blocked deletion,
clean-cluster no-op, and post-upgrade no-op. Uses only core resources
(Namespace, Secret) so no OpenShift or MaaS stack is required.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.resource import get_client
from ocp_resources.secret import Secret

from utilities.upgrade_utils import (
    DEFAULT_DELETE_TIMEOUT,
    purge_upgrade_leftover,
    upgrade_resource_factory,
    upgrade_resource_skip_reason,
)

FAST_DELETE_TIMEOUT = 10
SKIP_CONTEXT = "kind upgrade-utils tests"


class FakePytestConfig:
    """Minimal stand-in for pytest.Config carrying only the upgrade mode."""

    def __init__(self, post_upgrade: bool = False) -> None:
        self.option = SimpleNamespace(post_upgrade=post_upgrade)


@pytest.fixture(scope="session")
def admin_client() -> DynamicClient:
    client = get_client(try_refresh_token=False)
    return client  # type: ignore[return-value]


@pytest.fixture
def pre_upgrade_config() -> FakePytestConfig:
    return FakePytestConfig()


@pytest.fixture
def post_upgrade_config() -> FakePytestConfig:
    config = FakePytestConfig()
    config.option.post_upgrade = True
    return config


def make_namespace(admin_client: DynamicClient, name: str) -> Namespace:
    ns = Namespace(client=admin_client, name=name)
    ns.deploy(wait=True)
    return ns


def test_clean_cluster_is_noop(admin_client: DynamicClient, pre_upgrade_config: FakePytestConfig) -> None:
    """Path C: resource absent in pre-upgrade mode -> no error, no skip."""
    purge_upgrade_leftover(
        pytestconfig=pre_upgrade_config,
        resource_factory=upgrade_resource_factory(Namespace, client=admin_client, name="idem-clean-ns"),
        delete_timeout=FAST_DELETE_TIMEOUT,
        skip_context=SKIP_CONTEXT,
    )
    assert not Namespace(client=admin_client, name="idem-clean-ns").exists


def test_purge_and_recreate_leftover(admin_client: DynamicClient, pre_upgrade_config: FakePytestConfig) -> None:
    """Path A: leftover is deleted, so the caller can create it fresh."""
    ns = make_namespace(admin_client, name="idem-leftover-ns")
    secret_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": "idem-leftover-secret",
        "namespace": ns.name,
    }
    Secret(**secret_kwargs, string_data={"k": "v"}).deploy(wait=True)

    purge_upgrade_leftover(
        pytestconfig=pre_upgrade_config,
        resource_factory=upgrade_resource_factory(Secret, **secret_kwargs),
        delete_timeout=FAST_DELETE_TIMEOUT,
        skip_context=SKIP_CONTEXT,
    )
    assert not Secret(**secret_kwargs).exists

    # Caller-side recreate works right away (no 409).
    Secret(**secret_kwargs, string_data={"k": "fresh"}).deploy(wait=True)
    assert Secret(**secret_kwargs).exists


def test_skip_on_finalizer_blocked_delete(admin_client: DynamicClient, pre_upgrade_config: FakePytestConfig) -> None:
    """Path B: deletion blocked by an unserviced finalizer -> skip with a clear reason."""
    ns = make_namespace(admin_client, name="idem-blocked-ns")
    secret_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": "idem-blocked-secret",
        "namespace": ns.name,
    }
    secret = Secret(**secret_kwargs, string_data={"k": "v"})
    secret.deploy(wait=True)
    # Attach a finalizer nothing will ever remove. update() merge-patches, so
    # the body must carry name (and namespace) in metadata to address the object.
    secret.update(resource_dict={"metadata": {"name": secret.name, "finalizers": ["kind.ci/blocked"]}})

    with pytest.raises(pytest.skip.Exception) as excinfo:  # type: ignore[attr-defined]
        purge_upgrade_leftover(
            pytestconfig=pre_upgrade_config,
            resource_factory=upgrade_resource_factory(Secret, **secret_kwargs),
            delete_timeout=FAST_DELETE_TIMEOUT,
            skip_context=SKIP_CONTEXT,
        )
    reason = str(excinfo.value)
    assert "idem-blocked-secret" in reason
    assert "finalizer" in reason
    assert "oc delete" in reason


def test_post_upgrade_mode_never_deletes(admin_client: DynamicClient, post_upgrade_config: FakePytestConfig) -> None:
    """Path D: post-upgrade mode must not touch an existing resource."""
    ns = make_namespace(admin_client, name="idem-adopt-ns")
    secret_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": "idem-adopt-secret",
        "namespace": ns.name,
    }
    secret = Secret(**secret_kwargs, string_data={"k": "keep-me"})
    secret.deploy(wait=True)
    uid_before = secret.instance.metadata.uid

    purge_upgrade_leftover(
        pytestconfig=post_upgrade_config,
        resource_factory=upgrade_resource_factory(Secret, **secret_kwargs),
        delete_timeout=FAST_DELETE_TIMEOUT,
        skip_context=SKIP_CONTEXT,
    )
    assert secret.exists
    assert secret.instance.metadata.uid == uid_before


def test_skip_reason_names_resource_and_remedy(admin_client: DynamicClient) -> None:
    """The skip reason is the debugging signal: resource identity + manual remedy."""
    ns = Namespace(client=admin_client, name="reason-ns")
    reason = upgrade_resource_skip_reason(resource=ns, timeout=DEFAULT_DELETE_TIMEOUT, skip_context="MaaS upgrade")
    assert "MaaS upgrade" in reason
    assert "Namespace" in reason
    assert "reason-ns" in reason
    assert "oc delete namespace reason-ns --wait" in reason
