#
# Copyright Skodjob authors.
# License: Apache License 2.0 (see the file LICENSE or http://apache.org/licenses/LICENSE-2.0.html).
#
from __future__ import annotations

from typing import Generator

from kubernetes.dynamic import DynamicClient

from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim

import pytest

from utilities import constants
from utilities.infra import create_ns


@pytest.fixture(scope="class")
def users_namespace(
    request: pytest.FixtureRequest, unprivileged_client: DynamicClient
) -> Generator[Namespace, None, None]:
    with create_ns(
        unprivileged_client=unprivileged_client,
        name=request.param["name"],
        labels={constants.Labels.OpenDataHub.DASHBOARD: "true"},
        annotations={constants.Annotations.OpenDataHub.SERVICE_MESH: "false"},
    ) as ns:
        yield ns


@pytest.fixture(scope="function")
def users_persistent_volume_claim(
    request: pytest.FixtureRequest, users_namespace: Namespace, unprivileged_client: DynamicClient
) -> Generator[PersistentVolumeClaim, None, None]:
    with PersistentVolumeClaim(
        client=unprivileged_client,
        name=request.param["name"],
        namespace=users_namespace.name,
        label={constants.Labels.OpenDataHub.DASHBOARD: "true"},
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        size="10Gi",
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
    ) as pvc:
        yield pvc
