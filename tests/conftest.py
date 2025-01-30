from __future__ import annotations

import base64
import os
import shutil
from typing import List, Tuple, Any, Generator

import pytest
import yaml
from _pytest.tmpdir import TempPathFactory
from ocp_resources.secret import Secret
from pyhelper_utils.shell import run_command
from pytest import FixtureRequest, Config
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.namespace import Namespace
from ocp_resources.resource import get_client
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from utilities.data_science_cluster_utils import update_components_in_dsc
from utilities.infra import create_ns, login_with_user_password, get_openshift_token
from utilities.constants import AcceleratorType


LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session")
def admin_client() -> DynamicClient:
    return get_client()


@pytest.fixture(scope="session", autouse=True)
def tests_tmp_dir(request: FixtureRequest, tmp_path_factory: TempPathFactory) -> None:
    base_path = os.path.join(request.config.option.basetemp, "tests")
    tests_tmp_path = tmp_path_factory.mktemp(basename=base_path)
    py_config["tmp_base_dir"] = str(tests_tmp_path)

    yield

    shutil.rmtree(path=str(tests_tmp_path), ignore_errors=True)


@pytest.fixture(scope="session")
def current_client_token(admin_client: DynamicClient) -> str:
    return get_openshift_token()


@pytest.fixture(scope="class")
def model_namespace(request: FixtureRequest, admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    ns_kwargs = {"admin_client": admin_client, "name": request.param["name"]}

    if request.param.get("modelmesh-enabled"):
        request.getfixturevalue(argname="enabled_modelmesh_in_dsc")
        ns_kwargs["labels"] = {"modelmesh-enabled": "true"}

    with create_ns(**ns_kwargs) as ns:
        yield ns


@pytest.fixture(scope="session")
def aws_access_key_id(pytestconfig: Config) -> str:
    access_key = pytestconfig.option.aws_access_key_id
    if not access_key:
        raise ValueError(
            "AWS access key id is not set. "
            "Either pass with `--aws-access-key-id` or set `AWS_ACCESS_KEY_ID` environment variable"
        )
    return access_key


@pytest.fixture(scope="session")
def aws_secret_access_key(pytestconfig: Config) -> str:
    secret_access_key = pytestconfig.option.aws_secret_access_key
    if not secret_access_key:
        raise ValueError(
            "AWS secret access key is not set. "
            "Either pass with `--aws-secret-access-key` or set `AWS_SECRET_ACCESS_KEY` environment variable"
        )
    return secret_access_key


@pytest.fixture(scope="session")
def valid_aws_config(aws_access_key_id: str, aws_secret_access_key: str) -> Tuple[str, str]:
    return aws_access_key_id, aws_secret_access_key


@pytest.fixture(scope="session")
def ci_s3_bucket_name(pytestconfig: Config) -> str:
    bucket_name = pytestconfig.option.ci_s3_bucket_name
    if not bucket_name:
        raise ValueError(
            "CI S3 bucket name is not set. "
            "Either pass with `--ci-s3-bucket-name` or set `CI_S3_BUCKET_NAME` environment variable"
        )
    return bucket_name


@pytest.fixture(scope="session")
def ci_s3_bucket_region(pytestconfig: pytest.Config) -> str:
    ci_bucket_region = pytestconfig.option.ci_s3_bucket_region
    if not ci_bucket_region:
        raise ValueError(
            "Region for the ci s3 bucket is not defined."
            "Either pass with `--ci-s3-bucket-region` or set `CI_S3_BUCKET_REGION` environment variable"
        )
    return ci_bucket_region


@pytest.fixture(scope="session")
def ci_s3_bucket_endpoint(pytestconfig: pytest.Config) -> str:
    ci_bucket_endpoint = pytestconfig.option.ci_s3_bucket_endpoint
    if not ci_bucket_endpoint:
        raise ValueError(
            "Endpoint for the ci s3 bucket is not defined."
            "Either pass with `--ci-s3-bucket-endpoint` or set `CI_S3_BUCKET_ENDPOINT` environment variable"
        )
    return ci_bucket_endpoint


@pytest.fixture(scope="session")
def models_s3_bucket_name(pytestconfig: pytest.Config) -> str:
    models_bucket = pytestconfig.option.models_s3_bucket_name
    if not models_bucket:
        raise ValueError(
            "Bucket name for the models bucket is not defined."
            "Either pass with `--models-s3-bucket-name` or set `MODELS_S3_BUCKET_NAME` environment variable"
        )
    return models_bucket


@pytest.fixture(scope="session")
def models_s3_bucket_region(pytestconfig: pytest.Config) -> str:
    models_bucket_region = pytestconfig.option.models_s3_bucket_region
    if not models_bucket_region:
        raise ValueError(
            "region for the models bucket is not defined."
            "Either pass with `--models-s3-bucket-region` or set `MODELS_S3_BUCKET_REGION` environment variable"
        )
    return models_bucket_region


@pytest.fixture(scope="session")
def models_s3_bucket_endpoint(pytestconfig: pytest.Config) -> str:
    models_bucket_endpoint = pytestconfig.option.models_s3_bucket_endpoint
    if not models_bucket_endpoint:
        raise ValueError(
            "endpoint for the models bucket is not defined."
            "Either pass with `--models-s3-bucket-endpoint` or set `MODELS_S3_BUCKET_ENDPOINT` environment variable"
        )
    return models_bucket_endpoint


@pytest.fixture(scope="session")
def supported_accelerator_type(pytestconfig: pytest.Config) -> str | None:
    accelerator_type = pytestconfig.option.supported_accelerator_type
    if not accelerator_type:
        return None
    if accelerator_type.lower() not in AcceleratorType.SUPPORTED_LISTS:
        raise ValueError(
            "accelerator type is not defined."
            "Either pass with `--supported-accelerator-type` or set `SUPPORTED_ACCLERATOR_TYPE` environment variable"
        )
    return accelerator_type


@pytest.fixture(scope="session")
def vllm_runtime_image(pytestconfig: pytest.Config) -> str | None:
    runtime_image = pytestconfig.option.vllm_runtime_image
    if not runtime_image:
        return None
    return runtime_image


@pytest.fixture(scope="session")
def non_admin_user_password(admin_client: DynamicClient) -> Tuple[str, str] | None:
    def _decode_split_data(_data: str) -> List[str]:
        return base64.b64decode(_data).decode().split(",")

    if ldap_Secret := list(
        Secret.get(
            dyn_client=admin_client,
            name="openldap",
            namespace="openldap",
        )
    ):
        data = ldap_Secret[0].instance.data
        users = _decode_split_data(_data=data.users)
        passwords = _decode_split_data(_data=data.passwords)
        first_user_index = next(index for index, user in enumerate(users) if "user" in user)

        return users[first_user_index], passwords[first_user_index]

    LOGGER.error("ldap secret not found")
    return None


@pytest.fixture(scope="session")
def kubconfig_filepath() -> str:
    kubeconfig_path = os.path.join(os.path.expanduser("~"), ".kube/config")
    kubeconfig_path_from_env = os.getenv("KUBECONFIG", "")

    if os.path.isfile(kubeconfig_path_from_env):
        return kubeconfig_path_from_env

    return kubeconfig_path


@pytest.fixture(scope="session")
def unprivileged_client(
    admin_client: DynamicClient,
    kubconfig_filepath: str,
    non_admin_user_password: Tuple[str, str],
) -> Generator[DynamicClient, Any, Any]:
    """
    Provides none privileged API client. If non_admin_user_password is None, then it will yield admin_client.
    """
    if non_admin_user_password is None:
        yield admin_client

    else:
        current_user = run_command(command=["oc", "whoami"])[1].strip()

        if login_with_user_password(
            api_address=admin_client.configuration.host,
            user=non_admin_user_password[0],
            password=non_admin_user_password[1],
        ):
            with open(kubconfig_filepath) as fd:
                kubeconfig_content = yaml.safe_load(fd)

            unprivileged_context = kubeconfig_content["current-context"]

            # Get back to admin account
            login_with_user_password(
                api_address=admin_client.configuration.host,
                user=current_user.strip(),
            )
            yield get_client(config_file=kubconfig_filepath, context=unprivileged_context)

        else:
            yield admin_client


@pytest.fixture(scope="session")
def dsc_resource(admin_client: DynamicClient):
    name = py_config["dsc_name"]
    for dsc in DataScienceCluster.get(dyn_client=admin_client, name=name):
        return dsc
    raise ResourceNotFoundError(f"DSC resource {name} not found")


@pytest.fixture(scope="module")
def updated_dsc_component_state(
    request: FixtureRequest,
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    with update_components_in_dsc(
        dsc=dsc_resource,
        components={request.param["component_name"]: request.param["desired_state"]},
    ) as dsc:
        yield dsc
