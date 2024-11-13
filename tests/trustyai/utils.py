from typing import Dict, Any, Union

from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor


def update_configmap_data(configmap: ConfigMap, data: Dict[str, Any]) -> Union[ResourceEditor, None]:
    updated_cm = None
    if configmap.data != data:
        updated_cm = ResourceEditor(patches={configmap: {"data": data}})
        updated_cm.update(backup_resources=True)

    return updated_cm
