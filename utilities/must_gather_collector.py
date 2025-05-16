import os
import shlex

from pytest_testconfig import config as py_config
from pytest import Item
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger
from utilities.exceptions import InvalidArgumentsError
from utilities.infra import get_rhods_csv_version, get_oc_image_info, generate_openshift_pull_secret_file

BASE_DIRECTORY_NAME = "must-gather-collected"
BASE_RESULTS_DIR = "/home/odh/opendatahub-tests/"
LOGGER = get_logger(name=__name__)


def get_base_dir() -> str:
    if os.path.exists(BASE_RESULTS_DIR):
        # we are running from jenkins.
        return os.path.join(BASE_RESULTS_DIR, "results")
    else:
        # this is local run
        return ""


def set_must_gather_collector_values() -> dict[str, str]:
    py_config["must_gather_collector"] = {
        "must_gather_base_directory": os.path.join(get_base_dir(), BASE_DIRECTORY_NAME),
    }
    return py_config["must_gather_collector"]


def set_must_gather_collector_directory(item: Item, directory_path: str) -> None:
    must_gather_dict = py_config["must_gather_collector"]
    must_gather_dict["collector_directory"] = prepare_pytest_item_data_dir(item=item, output_dir=directory_path)


def get_must_gather_collector_dir() -> str:
    must_gather_dict = py_config["must_gather_collector"]
    return must_gather_dict.get(
        "collector_directory",
        must_gather_dict["must_gather_base_directory"],
    )


def prepare_pytest_item_data_dir(item: Item, output_dir: str) -> str:
    """
    Prepare output directory for pytest item

    Example:
        item.fspath= "/home/user/git/<tests-repo>/tests/<test_dir>/test_something.py"
        must-gather-base-dir = "must-gather-base-dir"
        item.name = "test1"
        item_dir_log = "must-gather-base-dir/test_dir/test_something/test1"
    """
    item_cls_name = item.cls.__name__ if item.cls else ""
    tests_path = item.session.config.inicfg.get("testpaths")
    assert tests_path, "pytest.ini must include testpaths"

    fspath_split_str = "/" if tests_path != os.path.split(item.fspath.dirname)[1] else ""
    item_dir_log = os.path.join(
        output_dir,
        item.fspath.dirname.split(f"/{tests_path}{fspath_split_str}")[-1],
        item.fspath.basename.partition(".py")[0],
        item_cls_name,
        item.name,
    )
    os.makedirs(item_dir_log, exist_ok=True)
    return item_dir_log


def get_must_gather_output_dir(must_gather_path: str) -> str:
    for item in os.listdir(must_gather_path):
        new_path = os.path.join(must_gather_path, item)
        if os.path.isdir(new_path):
            return new_path
    raise FileNotFoundError(f"No log directory was created in '{must_gather_path}'")


def run_must_gather(
    image_url: str = "",
    target_dir: str = "",
    since: str = "1m",
    component_name: str = "",
    namespaces_dict: dict[str, str] | None = None,
) -> str:
    """
    Process the arguments to build must-gather command and run the same

    Args:
         image_url (str): must-gather image url
         target_dir (str): must-gather target directory
         since (str): duration in seconds for must-gather log collection
         component_name (str): must-gather component name
         namespaces_dict (dict[str, str] | None): namespaces dict for extra data collection from different component
            namespaces

    Returns:
        str: must-gather output
    """
    if component_name and namespaces_dict:
        raise InvalidArgumentsError("component name and namespaces can't be passed together")

    must_gather_command = "oc adm must-gather"
    if target_dir:
        must_gather_command += f" --dest-dir={target_dir}"
    if since:
        must_gather_command += f" --since={since}"
    if image_url:
        must_gather_command += f" --image={image_url}"
    if component_name:
        must_gather_command += f" -- 'export COMPONENT={component_name}; /usr/bin/gather' "
    elif namespaces_dict:
        namespace_str = ""
        if namespaces_dict.get("operator"):
            namespace_str += f"export OPERATOR_NAMESPACE={shlex.quote(namespaces_dict['operator'])};"
        if namespaces_dict.get("notebooks"):
            namespace_str += f"export NOTEBOOKS_NAMESPACE={shlex.quote(namespaces_dict['notebooks'])};"
        if namespaces_dict.get("monitoring"):
            namespace_str += f"export MONITORING_NAMESPACE={shlex.quote(namespaces_dict['monitoring'])};"
        if namespaces_dict.get("application"):
            namespace_str += f"export APPLICATIONS_NAMESPACE={shlex.quote(namespaces_dict['application'])};"
        if namespaces_dict.get("model_registries"):
            namespace_str += f"export MODEL_REGISTRIES_NAMESPACE={shlex.quote(namespaces_dict['model_registries'])};"
        if namespaces_dict.get("ossm"):
            namespace_str += f"export OSSM_NS={shlex.quote(namespaces_dict['ossm'])};"
        if namespaces_dict.get("knative"):
            namespace_str += f"export KNATIVE_NS={shlex.quote(namespaces_dict['knative'])};"
        if namespaces_dict.get("auth"):
            namespace_str += f"export AUTH_NS={shlex.quote(namespaces_dict['auth'])};"
        must_gather_command += f" -- '{namespace_str} /usr/bin/gather'"

    return run_command(command=shlex.split(must_gather_command), check=False)[1]


def get_must_gather_image_info(architecture: str = "linux/amd64") -> str:
    try:
        csv_version = get_rhods_csv_version()
        if csv_version:
            must_gather_image_manifest = f"quay.io/modh/must-gather:rhoai-{csv_version.major}.{csv_version.minor}"
            pull_secret = generate_openshift_pull_secret_file()
            image_info = get_oc_image_info(
                image=must_gather_image_manifest, architecture=architecture, pull_secret=pull_secret
            )
            return f"quay.io/modh/must-gather@{image_info['digest']}"
        else:
            LOGGER.warning(
                "No RHAOI CSV found. Potentially ODH cluster and must-gather collection is not "
                "relevant for this cluster"
            )
            return ""
    except Exception as exec:
        raise RuntimeError(f"Failed to retrieve must-gather image info: {str(exec)}") from exec


def collect_rhoai_must_gather(
    target_dir: str, since: int, save_collection_output: bool = True, architecture: str = "linux/amd64"
) -> str:
    """
    Collect must-gather data for RHOAI cluster.

    Args:
        target_dir (str): Directory to store the must-gather output
        since (int): Time in seconds to collect logs from
        save_collection_output (bool, optional): Whether to save must-gather command output. Defaults to True.
        architecture (str, optional): Target architecture for must-gather image. Defaults to "linux/amd64".

    Returns:
        str: Path to the must-gather output directory, or empty string if collection is skipped
    """
    must_gather_image = get_must_gather_image_info(architecture=architecture)
    if must_gather_image:
        output = run_must_gather(image_url=must_gather_image, target_dir=target_dir, since=f"{since}s")
        if save_collection_output:
            with open(os.path.join(target_dir, "output.log"), "w") as _file:
                _file.write(output)
        return get_must_gather_output_dir(must_gather_path=target_dir)
    else:
        LOGGER.warning("Must-gather collection would be skipped.")
        return ""
