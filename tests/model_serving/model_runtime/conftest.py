import pytest
from ocp_resources.cluster_service_version import ClusterServiceVersion

from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger
from pytest_testconfig import config as py_config

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="package")
def fail_if_missing_dependent_operators(admin_client: DynamicClient) -> None:
    if dependent_operators := py_config.get("dependent_operators"):
        missing_operators: list[str] = []

        for operator_name in dependent_operators.split(","):
            csvs = list(
                ClusterServiceVersion.get(
                    dyn_client=admin_client,
                    namespace=py_config["applications_namespace"],
                )
            )

            LOGGER.info(f"Verifying if {operator_name} is installed")
            for csv in csvs:
                if csv.name.startswith(operator_name):
                    if csv.status == csv.Status.SUCCEEDED:
                        break

                    else:
                        missing_operators.append(
                            f"Operator {operator_name} is installed but CSV is not in {csv.Status.SUCCEEDED} state"
                        )

            else:
                missing_operators.append(f"{operator_name} is not installed")

        if missing_operators:
            pytest.fail(str(missing_operators))

    else:
        LOGGER.info("No dependent operators to verify")


@pytest.fixture(scope="session")
def skip_if_no_supported_accelerator_type(supported_accelerator_type: str) -> None:
    if not supported_accelerator_type:
        pytest.skip("Accelartor type is not provided,vLLM test can not be run on CPU")
