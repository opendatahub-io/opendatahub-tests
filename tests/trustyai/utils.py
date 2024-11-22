from contextlib import contextmanager
from typing import Dict, Any

from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor


@contextmanager
def update_configmap_data(configmap: ConfigMap, data: Dict[str, Any]) -> ResourceEditor:
    if configmap.data == data:
        yield configmap
    else:
        with ResourceEditor(patches={configmap: {"data": data}}) as update:
            yield update
