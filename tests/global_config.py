global config  # type:ignore[unused-ignore]

distribution: str = "downstream"
applications_namespace: str = "redhat-ods-applications"  # overwritten in conftest.py if distribution is upstream
dsc_name: str = "default-dsc"
dsci_name: str = "default-dsci"
dependent_operators: str = "servicemeshoperator,authorino-operator,serverless-operator"

use_unprivileged_client: bool = True

for _dir in dir():
    val = locals()[_dir]
    if type(val) not in [bool, list, dict, str, int]:
        continue

    if _dir in ["encoding", "py_file"]:
        continue

    config[_dir] = locals()[_dir]  # type:ignore[name-defined]  # noqa: F821
