import logging
import os
import pathlib
import shutil

from utilities.logger import separator, setup_logging


LOGGER = logging.getLogger(__name__)
BASIC_LOGGER = logging.getLogger("basic")


def pytest_addoption(parser):
    aws_group = parser.getgroup(name="AWS")
    buckets_group = parser.getgroup(name="Buckets")

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


def pytest_sessionstart(session):
    tests_log_file = session.config.getoption("log_file") or "pytest-tests.log"
    if os.path.exists(tests_log_file):
        pathlib.Path(tests_log_file).unlink()

    session.config.option.log_listener = setup_logging(
        log_file=tests_log_file,
        log_level=session.config.getoption("log_cli_level") or logging.INFO,
    )


def pytest_fixture_setup(fixturedef, request):
    LOGGER.info(f"Executing {fixturedef.scope} fixture: {fixturedef.argname}")


def pytest_runtest_setup(item):
    BASIC_LOGGER.info(f"\n{separator(symbol_='-', val=item.name)}")
    BASIC_LOGGER.info(f"{separator(symbol_='-', val='SETUP')}")


def pytest_runtest_call(item):
    BASIC_LOGGER.info(f"{separator(symbol_='-', val='CALL')}")


def pytest_runtest_teardown(item):
    BASIC_LOGGER.info(f"{separator(symbol_='-', val='TEARDOWN')}")


def pytest_report_teststatus(report, config):
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


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(path=session.config.option.basetemp, ignore_errors=True)

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    reporter.summary_stats()
