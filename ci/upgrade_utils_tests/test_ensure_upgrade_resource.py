"""Behavior-matrix tests for utilities.upgrade_utils against a real (kind) cluster.

Covers: purge-and-recreate on leftovers, skip on finalizer-blocked deletion,
clean-cluster no-op, and post-upgrade no-op. Uses only core resources
(Namespace, Secret) so no OpenShift or MaaS stack is required.
"""

from collections.abc import Generator
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
def kind_admin_client() -> DynamicClient:
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


def make_namespace(kind_admin_client: DynamicClient, name: str) -> Namespace:
    ns = Namespace(client=kind_admin_client, name=name)
    ns.deploy(wait=True)
    return ns


@pytest.fixture
def leftover_namespace(kind_admin_client: DynamicClient) -> Generator[str, Any, Any]:
    """Create a fixed-name namespace and remove it (and its contents) after the test.

    Ensures reruns against a reused kind cluster start clean even when the test
    fails mid-way: the same fixed-name collision this PR cures for the suites.
    """
    name = "idem-test-ns"
    purge_upgrade_leftover(
        pytestconfig=FakePytestConfig(),
        resource_factory=upgrade_resource_factory(Namespace, client=kind_admin_client, name=name),
        delete_timeout=FAST_DELETE_TIMEOUT,
        skip_context=SKIP_CONTEXT,
    )
    make_namespace(kind_admin_client=kind_admin_client, name=name)
    yield name
    Namespace(client=kind_admin_client, name=name).clean_up(wait=True, timeout=FAST_DELETE_TIMEOUT)


def test_clean_cluster_is_noop(kind_admin_client: DynamicClient, pre_upgrade_config: FakePytestConfig) -> None:
    """Path C: resource absent in pre-upgrade mode -> no error, no skip."""
    purge_upgrade_leftover(
        pytestconfig=pre_upgrade_config,
        resource_factory=upgrade_resource_factory(Namespace, client=kind_admin_client, name="idem-clean-ns"),
        delete_timeout=FAST_DELETE_TIMEOUT,
        skip_context=SKIP_CONTEXT,
    )
    assert not Namespace(client=kind_admin_client, name="idem-clean-ns").exists


def test_purge_and_recreate_leftover(
    kind_admin_client: DynamicClient,
    leftover_namespace: str,
    pre_upgrade_config: FakePytestConfig,
) -> None:
    """Path A: leftover is deleted, so the caller can create it fresh."""
    secret_kwargs: dict[str, Any] = {
        "client": kind_admin_client,
        "name": "idem-leftover-secret",
        "namespace": leftover_namespace,
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


def test_skip_on_finalizer_blocked_delete(
    kind_admin_client: DynamicClient,
    leftover_namespace: str,
    pre_upgrade_config: FakePytestConfig,
) -> None:
    """Path B: deletion blocked by an unserviced finalizer -> skip with a clear reason."""
    secret_kwargs: dict[str, Any] = {
        "client": kind_admin_client,
        "name": "idem-blocked-secret",
        "namespace": leftover_namespace,
    }
    secret = Secret(**secret_kwargs, string_data={"k": "v"})
    secret.deploy(wait=True)
    # Attach a finalizer nothing will ever remove. update() merge-patches, so
    # the body must carry name in metadata to address the object.
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
    assert "controller" in reason

    # Unblock the secret so fixture teardown (namespace clean_up) does not hang
    # on the finalizer this test attached.
    secret.update(resource_dict={"metadata": {"name": secret.name, "finalizers": None}})


def test_post_upgrade_mode_never_deletes(
    kind_admin_client: DynamicClient,
    leftover_namespace: str,
    post_upgrade_config: FakePytestConfig,
) -> None:
    """Path D: post-upgrade mode must not touch an existing resource."""
    secret_kwargs: dict[str, Any] = {
        "client": kind_admin_client,
        "name": "idem-adopt-secret",
        "namespace": leftover_namespace,
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


def test_skip_reason_names_resource_and_remedy(kind_admin_client: DynamicClient) -> None:
    """The skip reason is the debugging signal: resource identity + manual remedy."""
    ns = Namespace(client=kind_admin_client, name="reason-ns")
    reason = upgrade_resource_skip_reason(resource=ns, timeout=DEFAULT_DELETE_TIMEOUT, skip_context="MaaS upgrade")
    assert "MaaS upgrade" in reason
    assert "Namespace" in reason
    assert "reason-ns" in reason
    assert "oc get namespace reason-ns -o yaml" in reason
    assert "finalizer" in reason
