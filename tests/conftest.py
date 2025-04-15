import base64
import os
import shutil
from typing import Any, Callable, Generator

import pytest
import shortuuid
import yaml
from _pytest.tmpdir import TempPathFactory
from ocp_resources.config_map import ConfigMap
from ocp_resources.dsc_initialization import DSCInitialization
from ocp_resources.node import Node
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from pyhelper_utils.shell import run_command
from pytest import FixtureRequest, Config
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.namespace import Namespace
from ocp_resources.resource import get_client
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from utilities.data_science_cluster_utils import update_components_in_dsc
from utilities.exceptions import ClusterLoginError
from utilities.general import get_s3_secret_dict
from utilities.infra import (
    verify_cluster_sanity,
    create_ns,
    get_dsci_applications_namespace,
    get_operator_distribution,
    login_with_user_password,
    get_openshift_token,
)
from utilities.constants import (
    AcceleratorType,
    ApiGroups,
    DscComponents,
    Labels,
    MinIo,
    Protocols,
)
from utilities.infra import update_configmap_data


LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session")
def admin_client() -> DynamicClient:
    return get_client()


@pytest.fixture(scope="session", autouse=True)
def tests_tmp_dir(request: FixtureRequest, tmp_path_factory: TempPathFactory) -> Generator[None, None, None]:
    base_path = os.path.join(request.config.option.basetemp, "tests")
    tests_tmp_path = tmp_path_factory.mktemp(basename=base_path)
    py_config["tmp_base_dir"] = str(tests_tmp_path)

    yield

    shutil.rmtree(path=str(tests_tmp_path), ignore_errors=True)


@pytest.fixture(scope="session")
def updated_global_config(request: FixtureRequest, admin_client: DynamicClient) -> None:
    if get_operator_distribution(client=admin_client) == "Open Data Hub":
        py_config["distribution"] = "upstream"

    else:
        py_config["distribution"] = "downstream"

    if applications_namespace := request.config.getoption("applications_namespace"):
        py_config["applications_namespace"] = applications_namespace

    else:
        py_config["applications_namespace"] = get_dsci_applications_namespace(client=admin_client)


@pytest.fixture(scope="session")
def current_client_token(admin_client: DynamicClient) -> str:
    return get_openshift_token()


@pytest.fixture(scope="session")
def teardown_resources(pytestconfig: pytest.Config) -> bool:
    if delete_pre_upgrade_resources := pytestconfig.option.delete_pre_upgrade_resources:
        LOGGER.warning("Resources will be deleted")

    return delete_pre_upgrade_resources


@pytest.fixture(scope="class")
def model_namespace(
    request: FixtureRequest, pytestconfig: pytest.Config, admin_client: DynamicClient, teardown_resources: bool
) -> Generator[Namespace, Any, Any]:
    if request.param.get("modelmesh-enabled"):
        request.getfixturevalue(argname="enabled_modelmesh_in_dsc")

    ns = Namespace(client=admin_client, name=request.param["name"])

    if pytestconfig.option.post_upgrade:
        yield ns
        ns.clean_up()
    else:
        with create_ns(admin_client=admin_client, pytest_request=request, teardown=teardown_resources) as ns:
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
def valid_aws_config(aws_access_key_id: str, aws_secret_access_key: str) -> tuple[str, str]:
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
def non_admin_user_password(admin_client: DynamicClient) -> tuple[str, str] | None:
    def _decode_split_data(_data: str) -> list[str]:
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
    non_admin_user_password: tuple[str, str],
) -> Generator[DynamicClient, Any, Any]:
    """
    Provides none privileged API client. If non_admin_user_password is None, then it will raise.
    """
    if non_admin_user_password is None:
        raise ValueError("Unprivileged user not provisioned")

    else:
        current_user = run_command(command=["oc", "whoami"])[1].strip()
        non_admin_user_name = non_admin_user_password[0]

        if login_with_user_password(
            api_address=admin_client.configuration.host,
            user=non_admin_user_name,
            password=non_admin_user_password[1],
        ):
            with open(kubconfig_filepath) as fd:
                kubeconfig_content = yaml.safe_load(fd)

            unprivileged_context = kubeconfig_content["current-context"]

            unprivileged_client = get_client(config_file=kubconfig_filepath, context=unprivileged_context)

            # Get back to admin account
            login_with_user_password(
                api_address=admin_client.configuration.host,
                user=current_user.strip(),
            )
            yield unprivileged_client

        else:
            raise ClusterLoginError(user=non_admin_user_name)


@pytest.fixture(scope="session")
def dsci_resource(admin_client: DynamicClient) -> DSCInitialization:
    return DSCInitialization(client=admin_client, name=py_config["dsci_name"], ensure_exists=True)


@pytest.fixture(scope="session")
def dsc_resource(admin_client: DynamicClient) -> DataScienceCluster:
    return DataScienceCluster(client=admin_client, name=py_config["dsc_name"], ensure_exists=True)


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


@pytest.fixture(scope="package")
def enabled_modelmesh_in_dsc(
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    with update_components_in_dsc(
        dsc=dsc_resource,
        components={DscComponents.MODELMESHSERVING: DscComponents.ManagementState.MANAGED},
    ) as dsc:
        yield dsc


@pytest.fixture(scope="package")
def enabled_kserve_in_dsc(
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    with update_components_in_dsc(
        dsc=dsc_resource,
        components={DscComponents.KSERVE: DscComponents.ManagementState.MANAGED},
    ) as dsc:
        yield dsc


@pytest.fixture(scope="session")
def cluster_monitoring_config(
    admin_client: DynamicClient,
) -> Generator[ConfigMap, Any, Any]:
    data = {"config.yaml": yaml.dump({"enableUserWorkload": True})}

    with update_configmap_data(
        client=admin_client,
        name="cluster-monitoring-config",
        namespace="openshift-monitoring",
        data=data,
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def unprivileged_model_namespace(
    request: FixtureRequest, unprivileged_client: DynamicClient
) -> Generator[Namespace, Any, Any]:
    if request.param.get("modelmesh-enabled"):
        request.getfixturevalue(argname="enabled_modelmesh_in_dsc")

    with create_ns(unprivileged_client=unprivileged_client, pytest_request=request) as ns:
        yield ns


# MinIo
@pytest.fixture(scope="class")
def minio_namespace(admin_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    with create_ns(
        name=f"{MinIo.Metadata.NAME}-{shortuuid.uuid().lower()}",
        admin_client=admin_client,
    ) as ns:
        yield ns


@pytest.fixture(scope="class")
def minio_pod(
    request: FixtureRequest,
    admin_client: DynamicClient,
    minio_namespace: Namespace,
) -> Generator[Pod, Any, Any]:
    pod_labels = {Labels.Openshift.APP: MinIo.Metadata.NAME}

    if labels := request.param.get("labels"):
        pod_labels.update(labels)

    with Pod(
        client=admin_client,
        name=MinIo.Metadata.NAME,
        namespace=minio_namespace.name,
        containers=[
            {
                "args": request.param.get("args"),
                "env": [
                    {
                        "name": MinIo.Credentials.ACCESS_KEY_NAME,
                        "value": MinIo.Credentials.ACCESS_KEY_VALUE,
                    },
                    {
                        "name": MinIo.Credentials.SECRET_KEY_NAME,
                        "value": MinIo.Credentials.SECRET_KEY_VALUE,
                    },
                ],
                "image": request.param.get("image"),
                "name": MinIo.Metadata.NAME,
            }
        ],
        label=pod_labels,
        annotations=request.param.get("annotations"),
    ) as minio_pod:
        yield minio_pod


@pytest.fixture(scope="class")
def minio_service(admin_client: DynamicClient, minio_namespace: Namespace) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        name=MinIo.Metadata.NAME,
        namespace=minio_namespace.name,
        ports=[
            {
                "name": f"{MinIo.Metadata.NAME}-client-port",
                "port": MinIo.Metadata.DEFAULT_PORT,
                "protocol": Protocols.TCP,
                "targetPort": MinIo.Metadata.DEFAULT_PORT,
            }
        ],
        selector={
            Labels.Openshift.APP: MinIo.Metadata.NAME,
        },
    ) as minio_service:
        yield minio_service


@pytest.fixture(scope="class")
def minio_data_connection(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_service: Service,
) -> Generator[Secret, Any, Any]:
    data_dict = get_s3_secret_dict(
        aws_access_key=MinIo.Credentials.ACCESS_KEY_VALUE,
        aws_secret_access_key=MinIo.Credentials.SECRET_KEY_VALUE,  # pragma: allowlist secret
        aws_s3_bucket=request.param["bucket"],
        aws_s3_endpoint=f"{Protocols.HTTP}://{minio_service.instance.spec.clusterIP}:{str(MinIo.Metadata.DEFAULT_PORT)}",  # noqa: E501
        aws_s3_region="us-south",
    )

    with Secret(
        client=admin_client,
        name="aws-connection-minio-data-connection",
        namespace=model_namespace.name,
        data_dict=data_dict,
        label={
            Labels.OpenDataHub.DASHBOARD: "true",
            Labels.OpenDataHubIo.MANAGED: "true",
        },
        annotations={
            f"{ApiGroups.OPENDATAHUB_IO}/connection-type": "s3",
            "openshift.io/display-name": "Minio Data Connection",
        },
    ) as minio_secret:
        yield minio_secret


@pytest.fixture(scope="session")
def nodes(admin_client: DynamicClient) -> Generator[list[Node], Any, Any]:
    yield list(Node.get(dyn_client=admin_client))


@pytest.fixture(scope="session")
def junitxml_plugin(
    request: FixtureRequest, record_testsuite_property: Callable[[str, object], None]
) -> Callable[[str, object], None] | None:
    return record_testsuite_property if request.config.pluginmanager.has_plugin("junitxml") else None


@pytest.fixture(scope="session", autouse=True)
@pytest.mark.early(order=0)
def cluster_sanity_scope_session(
    request: FixtureRequest,
    nodes: list[Node],
    dsci_resource: DSCInitialization,
    dsc_resource: DataScienceCluster,
    junitxml_plugin: Callable[[str, object], None],
) -> None:
    verify_cluster_sanity(
        request=request,
        nodes=nodes,
        dsc_resource=dsc_resource,
        dsci_resource=dsci_resource,
        junitxml_property=junitxml_plugin,
    )
