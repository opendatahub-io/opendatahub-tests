import logging
import os
import pathlib
import shutil
import datetime
import traceback

import shortuuid
from _pytest.runner import CallInfo
from _pytest.reports import TestReport
from pytest import (
    Parser,
    Session,
    FixtureRequest,
    FixtureDef,
    Item,
    Collector,
    Config,
    CollectReport,
)
from _pytest.terminal import TerminalReporter
from typing import Optional, Any
from pytest_testconfig import config as py_config

from utilities.constants import KServeDeploymentType
from utilities.database import Database
from utilities.logger import separator, setup_logging
from utilities.must_gather_collector import (
    set_must_gather_collector_directory,
    set_must_gather_collector_values,
    get_must_gather_collector_dir,
    collect_rhoai_must_gather,
    get_base_dir,
)

LOGGER = logging.getLogger(name=__name__)
BASIC_LOGGER = logging.getLogger(name="basic")


def pytest_addoption(parser: Parser) -> None:
    aws_group = parser.getgroup(name="AWS")
    buckets_group = parser.getgroup(name="Buckets")
    runtime_group = parser.getgroup(name="Runtime details")
    upgrade_group = parser.getgroup(name="Upgrade options")
    must_gather_group = parser.getgroup(name="MustGather")
    cluster_sanity_group = parser.getgroup(name="ClusterSanity")

    # AWS config and credentials options
    aws_group.addoption(
        "--aws-secret-access-key",
        default=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        help="AWS secret access key",
    )
    aws_group.addoption(
        "--aws-access-key-id",
        default=os.environ.get("AWS_ACCESS_KEY_ID"),
        help="AWS access key id",
    )

    # Buckets options
    buckets_group.addoption(
        "--ci-s3-bucket-name",
        default=os.environ.get("CI_S3_BUCKET_NAME"),
        help="Ci S3 bucket name",
    )
    buckets_group.addoption(
        "--ci-s3-bucket-region",
        default=os.environ.get("CI_S3_BUCKET_REGION"),
        help="Ci S3 bucket region",
    )

    buckets_group.addoption(
        "--ci-s3-bucket-endpoint",
        default=os.environ.get("CI_S3_BUCKET_ENDPOINT"),
        help="Ci S3 bucket endpoint",
    )

    buckets_group.addoption(
        "--models-s3-bucket-name",
        default=os.environ.get("MODELS_S3_BUCKET_NAME"),
        help="Models S3 bucket name",
    )
    buckets_group.addoption(
        "--models-s3-bucket-region",
        default=os.environ.get("MODELS_S3_BUCKET_REGION"),
        help="Models S3 bucket region",
    )

    buckets_group.addoption(
        "--models-s3-bucket-endpoint",
        default=os.environ.get("MODELS_S3_BUCKET_ENDPOINT"),
        help="Models S3 bucket endpoint",
    )
    # Runtime options
    runtime_group.addoption(
        "--supported-accelerator-type",
        default=os.environ.get("SUPPORTED_ACCLERATOR_TYPE"),
        help="Supported accelerator type : Nvidia,AMD,Gaudi",
    )
    runtime_group.addoption(
        "--vllm-runtime-image",
        default=os.environ.get("VLLM_RUNTIME_IMAGE"),
        help="Specify the runtime image to use for the tests",
    )

    # Upgrade options
    upgrade_group.addoption(
        "--pre-upgrade",
        action="store_true",
        help="Run pre-upgrade tests",
    )
    upgrade_group.addoption(
        "--post-upgrade",
        action="store_true",
        help="Run post-upgrade tests",
    )
    upgrade_group.addoption(
        "--delete-pre-upgrade-resources",
        action="store_true",
        help="Delete pre-upgrade resources; useful when debugging pre-upgrade tests",
    )
    upgrade_group.addoption(
        "--upgrade-deployment-modes",
        help="Coma-separated str; specify inference service deployment modes tests to run in upgrade tests. "
        "If not set, all will be tested.",
    )
    must_gather_group.addoption(
        "--collect-must-gather",
        help="Indicate if must-gather should be collected on failure.",
        action="store_true",
        default=False,
    )

    # Cluster sanity options
    cluster_sanity_group.addoption(
        "--cluster-sanity-skip-check",
        help="Skip cluster_sanity check",
        action="store_true",
    )
    cluster_sanity_group.addoption(
        "--cluster-sanity-skip-rhoai-check",
        help="Skip RHOAI/ODH-related resources (DSCI and DSC) checks",
        action="store_true",
    )


def pytest_cmdline_main(config: Any) -> None:
    config.option.basetemp = py_config["tmp_base_dir"] = f"{config.option.basetemp}-{shortuuid.uuid()}"


def pytest_collection_modifyitems(session: Session, config: Config, items: list[Item]) -> None:
    """
    Pytest fixture to filter or re-order the items in-place.

    Filters upgrade tests based on '--pre-upgrade' / '--post-upgrade' option and marker.
    If `--upgrade-deployment-modes` option is set, only tests with the specified deployment modes will be added.
    """

    def _add_upgrade_test(_item: Item, _upgrade_deployment_modes: list[str]) -> bool:
        """
        Add upgrade test to the list of tests to run.

        Args:
            _item (Item): The test item.
            _upgrade_deployment_modes (list[str]): The deployment modes to test.

        Returns:
            True if the test should be added, False otherwise.

        """
        if not _upgrade_deployment_modes:
            return True

        return any([keyword for keyword in _item.keywords if keyword in _upgrade_deployment_modes])

    pre_upgrade_tests: list[Item] = []
    post_upgrade_tests: list[Item] = []
    non_upgrade_tests: list[Item] = []
    upgrade_deployment_modes: list[str] = []

    run_pre_upgrade_tests: str | None = config.getoption(name="pre_upgrade")
    run_post_upgrade_tests: str | None = config.getoption(name="post_upgrade")
    if config_upgrade_deployment_modes := config.getoption(name="upgrade_deployment_modes"):
        upgrade_deployment_modes = config_upgrade_deployment_modes.split(",")

    for item in items:
        if "pre_upgrade" in item.keywords and _add_upgrade_test(
            _item=item, _upgrade_deployment_modes=upgrade_deployment_modes
        ):
            pre_upgrade_tests.append(item)

        elif "post_upgrade" in item.keywords and _add_upgrade_test(
            _item=item, _upgrade_deployment_modes=upgrade_deployment_modes
        ):
            post_upgrade_tests.append(item)

        else:
            non_upgrade_tests.append(item)

    upgrade_tests = pre_upgrade_tests + post_upgrade_tests

    if run_pre_upgrade_tests and run_post_upgrade_tests:
        items[:] = upgrade_tests
        config.hook.pytest_deselected(items=non_upgrade_tests)

    elif run_pre_upgrade_tests:
        items[:] = pre_upgrade_tests
        config.hook.pytest_deselected(items=post_upgrade_tests + non_upgrade_tests)

    elif run_post_upgrade_tests:
        items[:] = post_upgrade_tests
        config.hook.pytest_deselected(items=pre_upgrade_tests + non_upgrade_tests)

    else:
        items[:] = non_upgrade_tests
        config.hook.pytest_deselected(items=upgrade_tests)


def pytest_sessionstart(session: Session) -> None:
    log_file = session.config.getoption("log_file") or "pytest-tests.log"
    tests_log_file = os.path.join(get_base_dir(), log_file)
    LOGGER.info(f"Writing tests log to {tests_log_file}")
    if os.path.exists(tests_log_file):
        pathlib.Path(tests_log_file).unlink()
    if session.config.getoption("--collect-must-gather"):
        session.config.option.must_gather_db = Database()
    session.config.option.log_listener = setup_logging(
        log_file=tests_log_file,
        log_level=session.config.getoption("log_cli_level") or logging.INFO,
    )
    must_gather_dict = set_must_gather_collector_values()
    shutil.rmtree(
        path=must_gather_dict["must_gather_base_directory"],
        ignore_errors=True,
    )


def pytest_fixture_setup(fixturedef: FixtureDef[Any], request: FixtureRequest) -> None:
    LOGGER.info(f"Executing {fixturedef.scope} fixture: {fixturedef.argname}")


def pytest_runtest_setup(item: Item) -> None:
    """
    Performs the following actions:
    1. Updates global config (`updated_global_config`)
    2. Adds `fail_if_missing_dependent_operators` fixture for Serverless tests.
    3. Adds fixtures to enable KServe/model mesh in DSC for model server tests.
    """
    BASIC_LOGGER.info(f"\n{separator(symbol_='-', val=item.name)}")
    BASIC_LOGGER.info(f"{separator(symbol_='-', val='SETUP')}")
    if item.config.getoption("--collect-must-gather"):
        # set must-gather collection directory:
        set_must_gather_collector_directory(item=item, directory_path=get_must_gather_collector_dir())

        # At the begining of setup work, insert current epoch time into the database to indicate test
        # start time

        try:
            db = item.config.option.must_gather_db
            db.insert_test_start_time(
                test_name=f"{item.fspath}::{item.name}",
                start_time=int(datetime.datetime.now().timestamp()),
            )
        except Exception as db_exception:
            LOGGER.error(f"Database error: {db_exception}. Must-gather collection may not be accurate")

    if KServeDeploymentType.SERVERLESS.lower() in item.keywords:
        item.fixturenames.insert(0, "fail_if_missing_dependent_operators")

    if KServeDeploymentType.SERVERLESS.lower() in item.keywords:
        item.fixturenames.insert(0, "enabled_kserve_in_dsc")

    elif KServeDeploymentType.RAW_DEPLOYMENT.lower() in item.keywords:
        item.fixturenames.insert(0, "enabled_kserve_in_dsc")

    elif KServeDeploymentType.MODEL_MESH.lower() in item.keywords:
        item.fixturenames.insert(0, "enabled_modelmesh_in_dsc")

    # The above fixtures require the global config to be updated before being called
    item.fixturenames.insert(0, "updated_global_config")


def pytest_runtest_call(item: Item) -> None:
    BASIC_LOGGER.info(f"{separator(symbol_='-', val='CALL')}")


def pytest_runtest_teardown(item: Item) -> None:
    BASIC_LOGGER.info(f"{separator(symbol_='-', val='TEARDOWN')}")
    # reset must-gather collector after each tests
    py_config["must_gather_collector"]["collector_directory"] = py_config["must_gather_collector"][
        "must_gather_base_directory"
    ]


def pytest_report_teststatus(report: CollectReport, config: Config) -> None:
    test_name = report.head_line
    when = report.when
    call_str = "call"
    if report.passed:
        if when == call_str:
            BASIC_LOGGER.info(f"\nTEST: {test_name} STATUS: \033[0;32mPASSED\033[0m")

    elif report.skipped:
        BASIC_LOGGER.info(f"\nTEST: {test_name} STATUS: \033[1;33mSKIPPED\033[0m")

    elif report.failed:
        if when != call_str:
            BASIC_LOGGER.info(f"\nTEST: {test_name} [{when}] STATUS: \033[0;31mERROR\033[0m")
        else:
            BASIC_LOGGER.info(f"\nTEST: {test_name} STATUS: \033[0;31mFAILED\033[0m")


def pytest_sessionfinish(session: Session, exitstatus: int) -> None:
    session.config.option.log_listener.stop()
    if session.config.option.setupplan or session.config.option.collectonly:
        return
    if session.config.getoption("--collect-must-gather"):
        db = session.config.option.must_gather_db
        file_path = db.database_file_path
        LOGGER.info(f"Removing database file path {file_path}")
        if os.path.exists(file_path):
            os.remove(file_path)
        # clean up the empty folders
    collector_directory = py_config["must_gather_collector"]["must_gather_base_directory"]
    if os.path.exists(collector_directory):
        for root, dirs, files in os.walk(collector_directory, topdown=False):
            for _dir in dirs:
                dir_path = os.path.join(root, _dir)
                if not os.listdir(dir_path):
                    shutil.rmtree(path=dir_path, ignore_errors=True)
    LOGGER.info(f"Deleting pytest base dir {session.config.option.basetemp}")
    shutil.rmtree(path=session.config.option.basetemp, ignore_errors=True)

    reporter: Optional[TerminalReporter] = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.summary_stats()


def calculate_must_gather_timer(test_start_time: int) -> int:
    default_duration = 300
    if test_start_time > 0:
        duration = int(datetime.datetime.now().timestamp()) - test_start_time
        return duration if duration > 60 else default_duration
    else:
        LOGGER.warning(f"Could not get start time of test. Collecting must-gather for last {default_duration}s")
        return default_duration


def pytest_exception_interact(node: Item | Collector, call: CallInfo[Any], report: TestReport | CollectReport) -> None:
    LOGGER.error(report.longreprtext)
    if node.config.getoption("--collect-must-gather"):
        test_name = f"{node.fspath}::{node.name}"
        LOGGER.info(f"Must-gather collection is enabled for {test_name}.")

        try:
            db = node.config.option.must_gather_db
            test_start_time = db.get_test_start_time(test_name=test_name)
        except Exception as db_exception:
            test_start_time = 0
            LOGGER.warning(f"Error: {db_exception} in accessing database.")

        try:
            collect_rhoai_must_gather(
                since=calculate_must_gather_timer(test_start_time=test_start_time),
                target_dir=os.path.join(get_must_gather_collector_dir(), "pytest_exception_interact"),
            )

        except Exception as current_exception:
            LOGGER.warning(f"Failed to collect logs: {test_name}: {current_exception} {traceback.format_exc()}")
