from __future__ import annotations

import logging
import os
import pathlib
import shutil

import shortuuid
from pytest import Parser, Session, FixtureRequest, FixtureDef, Item, Config, CollectReport
from _pytest.terminal import TerminalReporter
from typing import Optional, Any
from pytest_testconfig import config as py_config

from utilities.constants import KServeDeploymentType
from utilities.logger import separator, setup_logging


LOGGER = logging.getLogger(name=__name__)
BASIC_LOGGER = logging.getLogger(name="basic")


def pytest_addoption(parser: Parser) -> None:
    aws_group = parser.getgroup(name="AWS")
    buckets_group = parser.getgroup(name="Buckets")
    runtime_group = parser.getgroup(name="Runtime details")
    upgrade_group = parser.getgroup(name="Upgrade options")

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
        "--ci-s3-bucket-name", default=os.environ.get("CI_S3_BUCKET_NAME"), help="Ci S3 bucket name"
    )
    buckets_group.addoption(
        "--ci-s3-bucket-region", default=os.environ.get("CI_S3_BUCKET_REGION"), help="Ci S3 bucket region"
    )

    buckets_group.addoption(
        "--ci-s3-bucket-endpoint", default=os.environ.get("CI_S3_BUCKET_ENDPOINT"), help="Ci S3 bucket endpoint"
    )

    buckets_group.addoption(
        "--models-s3-bucket-name", default=os.environ.get("MODELS_S3_BUCKET_NAME"), help="Models S3 bucket name"
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


def pytest_cmdline_main(config: Any) -> None:
    config.option.basetemp = py_config["tmp_base_dir"] = f"{config.option.basetemp}-{shortuuid.uuid()}"


def pytest_collection_modifyitems(session: Session, config: Config, items: list[Item]) -> None:
    """
    Pytest fixture to filter or re-order the items in-place.

    Filters upgrade tests based on '--pre-upgrade' / '--post-upgrade' option and marker.
    """
    pre_upgrade_tests: list[Item] = []
    post_upgrade_tests: list[Item] = []
    non_upgrade_tests: list[Item] = []

    run_pre_upgrade_tests: str | None = config.getoption(name="pre_upgrade")
    run_post_upgrade_tests: str | None = config.getoption(name="post_upgrade")

    for item in items:
        if "pre_upgrade" in item.keywords:
            pre_upgrade_tests.append(item)

        elif "post_upgrade" in item.keywords:
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
    tests_log_file = session.config.getoption("log_file") or "pytest-tests.log"
    if os.path.exists(tests_log_file):
        pathlib.Path(tests_log_file).unlink()

    session.config.option.log_listener = setup_logging(
        log_file=tests_log_file,
        log_level=session.config.getoption("log_cli_level") or logging.INFO,
    )

    if py_config.get("distribution") == "upstream":
        py_config["applications_namespace"] = "opendatahub"


def pytest_fixture_setup(fixturedef: FixtureDef[Any], request: FixtureRequest) -> None:
    LOGGER.info(f"Executing {fixturedef.scope} fixture: {fixturedef.argname}")


def pytest_runtest_setup(item: Item) -> None:
    """
    Performs the following actions:
    1. Adds skip fixture for kserve if serverless or authorino operators are not installed.
    2. Adds skip fixture for serverless if authorino/serverless/service mesh are not deployed.
    """

    BASIC_LOGGER.info(f"\n{separator(symbol_='-', val=item.name)}")
    BASIC_LOGGER.info(f"{separator(symbol_='-', val='SETUP')}")

    if KServeDeploymentType.SERVERLESS.lower() in item.keywords:
        item.fixturenames.insert(0, "skip_if_no_deployed_redhat_authorino_operator")
        item.fixturenames.insert(0, "skip_if_no_deployed_openshift_serverless")
        item.fixturenames.insert(0, "skip_if_no_deployed_openshift_service_mesh")
        item.fixturenames.insert(0, "enabled_kserve_in_dsc")

    elif KServeDeploymentType.RAW_DEPLOYMENT.lower() in item.keywords:
        item.fixturenames.insert(0, "enabled_kserve_in_dsc")

    elif KServeDeploymentType.MODEL_MESH.lower() in item.keywords:
        item.fixturenames.insert(0, "enabled_modelmesh_in_dsc")


def pytest_runtest_call(item: Item) -> None:
    BASIC_LOGGER.info(f"{separator(symbol_='-', val='CALL')}")


def pytest_runtest_teardown(item: Item) -> None:
    BASIC_LOGGER.info(f"{separator(symbol_='-', val='TEARDOWN')}")


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
    if session.config.option.setupplan or session.config.option.collectonly:
        return

    base_dir = py_config["tmp_base_dir"]
    LOGGER.info(f"Deleting pytest base dir {base_dir}")
    shutil.rmtree(path=base_dir, ignore_errors=True)

    reporter: Optional[TerminalReporter] = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.summary_stats()
