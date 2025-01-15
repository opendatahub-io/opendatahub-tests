#
# Copyright Skodjob authors.
# License: Apache License 2.0 (see the file LICENSE or http://apache.org/licenses/LICENSE-2.0.html).
#
from __future__ import annotations

from ocp_resources.resource import NamespacedResource


class Notebook(NamespacedResource):
    api_group: str = "kubeflow.org"
