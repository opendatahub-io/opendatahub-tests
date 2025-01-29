import logging
import os
import pathlib
import shutil

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
    runtime_group = parser.getgroup(name="Runtime Details")

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
        item.fixturenames.insert(0, "skip_if_no_redhat_authorino_operator")
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
    shutil.rmtree(path=session.config.option.basetemp, ignore_errors=True)

    reporter: Optional[TerminalReporter] = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.summary_stats()
