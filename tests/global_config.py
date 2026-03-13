from utilities.constants import RHOAI_OPERATOR_NAMESPACE

global config  # type:ignore[unused-ignore]  # noqa: PLW0604

dsc_name: str = "default-dsc"
must_gather_base_dir: str = "must-gather-base-dir"
dsci_name: str = "default-dsci"
dependent_operators: str = "authorino-operator"
use_unprivileged_client: bool = True
# overwrite the followings in conftest.py, in updated_global_config() if distribution is upstream
distribution: str = "downstream"
applications_namespace: str = "redhat-ods-applications"
model_registry_namespace: str = "rhoai-model-registries"
operator_namespace: str = RHOAI_OPERATOR_NAMESPACE

for _dir in dir():
    val = locals()[_dir]
    if type(val) not in [bool, list, dict, str, int]:
        continue

    if _dir in ["encoding", "py_file"]:
        continue

    config[_dir] = locals()[_dir]  # type:ignore[name-defined]  # noqa: F821
