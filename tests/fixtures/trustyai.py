import json
import tempfile
from pathlib import Path

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment

from typing import Generator

from ocp_resources.resource import ResourceEditor
from pytest_testconfig import py_config

from tests.llama_stack.constants import LlamaStackProviders
from utilities.constants import TRUSTYAI_SERVICE_NAME
from utilities.infra import get_data_science_cluster


@pytest.fixture(scope="class")
def trustyai_operator_deployment(admin_client: DynamicClient) -> Deployment:
    return Deployment(
        client=admin_client,
        name=f"{TRUSTYAI_SERVICE_NAME}-operator-controller-manager",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def patched_dsc_lmeval_allow_all(
    admin_client, trustyai_operator_deployment: Deployment
) -> Generator[DataScienceCluster, None, None]:
    """Enable LMEval PermitOnline and PermitCodeExecution flags in the Datascience cluster."""
    dsc = get_data_science_cluster(client=admin_client)
    with ResourceEditor(
        patches={
            dsc: {
                "spec": {
                    "components": {
                        "trustyai": {
                            "eval": {
                                "lmeval": {
                                    "permitCodeExecution": "allow",
                                    "permitOnline": "allow",
                                }
                            }
                        }
                    }
                }
            }
        }
    ):
        num_replicas: int = trustyai_operator_deployment.replicas
        trustyai_operator_deployment.scale_replicas(replica_count=0)
        trustyai_operator_deployment.scale_replicas(replica_count=num_replicas)
        trustyai_operator_deployment.wait_for_replicas()
        yield dsc


@pytest.fixture
def custom_benchmark_config(custom_dataset):
    return {
        "benchmark_id": "trustyai-lmeval-custom",
        "dataset_id": "custom_eval_ds",
        "scoring_functions": ["string"],
        "provider_id": LlamaStackProviders.Eval.TRUSTYAI_LMEVAL,
        "provider_benchmark_id": "string",
        "metadata": {
            "tokenized_requests": False,
            "dataset_location": custom_dataset,
            "dataset_type": "custom",
            "tokenizer": "google/flan-t5-small",
        },
    }


@pytest.fixture(scope="class")
def custom_dataset():
    """
    Fixture that sets up a temporary custom JSONL dataset for LlamaStack benchmark tests.
    Provides the dataset path for use in tests and other fixtures.
    """

    temp_dir = tempfile.mkdtemp(prefix="custom_dataset_")

    dataset_file = Path(temp_dir) / "custom_data.jsonl"

    data = [
        {
            "user_input": "what is the meaning of verifying the identity of a person or an entity ",
            "reference": "It means to use methods to ensure that the information in an identification document or from other informational sources matches the information that the person or entity provided.",
            "response": "Verifying identity is the process of obtaining, recording, and maintaining information to confirm a person or entity's identity. This information can include name, address, date of birth, or other details, depending on the specific method used. The purpose of identity verification is to ensure that financial institutions can confirm that the person they are dealing with is who they claim to be.",
        },
        {
            "user_input": "Why is it important to verify identify?",
            "reference": "Verifying identify is a foundational element of Canada's anti-money laundering and anti-terrorist financing_x000D_\n          regime and a key component of an Reporting Entity's relationship with clients. It helps you to know_x000D_\n          your clients and to understand_x000D_\n          and assess any risk that be associated to their transactions or activities.",
            "response": "Verifying identity is a critical step in maintaining security and preventing fraud. It's required by Mexican law and is used to protect you, your data, and your financial accounts.",
        },
    ]

    # Write each record as a line in JSONL format
    with open(dataset_file, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")

    yield temp_dir
