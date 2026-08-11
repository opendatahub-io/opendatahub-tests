"""End-to-end model deployment lifecycle via rhoai-mcp MCP tools.

Deploys a lightweight MNIST ModelCar model (OCI image, OpenVINO format,
OVMS runtime) entirely through MCP tool calls — no direct K8s API usage
for the model-serving operations.  Analogous to the OCI ModelCar test
in tests/model_serving/ but driven by the rhoai-mcp server.
"""

import json

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from ocp_resources.namespace import Namespace
from tenacity import retry as tenacity_retry
from tenacity import retry_if_not_result, stop_after_delay, wait_exponential

from tests.rhoai_mcp.constants import (
    RHOAI_MCP_MODEL_DEPLOY_FORMAT,
    RHOAI_MCP_MODEL_DEPLOY_NAME,
    RHOAI_MCP_MODEL_DEPLOY_RUNTIME_TEMPLATE,
)
from utilities.image_constants import SharedImages

STORAGE_URI: str = SharedImages.MODELCAR_MNIST_8_1


def _parse_tool_result(result: object) -> dict:
    """Parse the JSON payload from a call_tool response."""
    return json.loads(result.content[0].text)


@tenacity_retry(
    stop=stop_after_delay(300),
    wait=wait_exponential(min=5, max=30),
    retry=retry_if_not_result(lambda data: data.get("status") == "Ready"),
)
async def _wait_for_model_ready(client: Client, name: str, namespace: str) -> dict:
    """Poll get_inference_service until the model reports Ready or timeout."""
    result = await client.call_tool(
        name="get_inference_service",
        arguments={"name": name, "namespace": namespace},
    )
    return _parse_tool_result(result=result)


@pytest.mark.asyncio
@pytest.mark.tier1
@pytest.mark.usefixtures("mcp_model_deploy_namespace")
class TestRhoaiMcpModelDeployment:
    """Validate model deployment lifecycle using rhoai-mcp MCP tools.

    Steps:
        1. Discover serving runtimes via list_serving_runtimes.
        2. Run pre-flight checks via check_deployment_prerequisites.
        3. Instantiate an OVMS serving runtime via create_serving_runtime.
        4. Deploy a MNIST OCI ModelCar model via deploy_model.
        5. Wait for the model to reach Ready status.
        6. Verify the model endpoint is accessible.
        7. Verify the model appears in namespace listings.
    """

    @pytest.mark.dependency(name="list_runtimes")
    async def test_list_serving_runtimes(
        self,
        mcp_model_deployer_transport: StreamableHttpTransport,
        mcp_model_deploy_namespace: Namespace,
    ) -> None:
        """Given a namespace with no existing runtimes
        When list_serving_runtimes is called with include_templates=True
        Then the OVMS kserve template is discoverable with openvino_ir support
        """
        async with Client(mcp_model_deployer_transport) as client:
            result = await client.call_tool(
                name="list_serving_runtimes",
                arguments={
                    "namespace": mcp_model_deploy_namespace.name,
                    "include_templates": True,
                },
            )
        data = _parse_tool_result(result=result)

        runtimes = data["result"]
        assert runtimes, "No serving runtimes found (including templates)"

        ovms_runtime = next(
            (r for r in runtimes if RHOAI_MCP_MODEL_DEPLOY_FORMAT in r.get("supported_formats", [])),
            None,
        )
        assert ovms_runtime is not None, (
            f"No runtime supports '{RHOAI_MCP_MODEL_DEPLOY_FORMAT}'; available: {[r['name'] for r in runtimes]}"
        )

    @pytest.mark.dependency(name="check_prereqs", depends=["list_runtimes"])
    async def test_check_deployment_prerequisites(
        self,
        mcp_model_deployer_transport: StreamableHttpTransport,
        mcp_model_deploy_namespace: Namespace,
    ) -> None:
        """Given the target namespace, model format, and OCI storage URI
        When check_deployment_prerequisites is called
        Then all pre-flight checks pass (namespace, runtime, storage)
        """
        async with Client(mcp_model_deployer_transport) as client:
            result = await client.call_tool(
                name="check_deployment_prerequisites",
                arguments={
                    "namespace": mcp_model_deploy_namespace.name,
                    "model_format": RHOAI_MCP_MODEL_DEPLOY_FORMAT,
                    "storage_uri": STORAGE_URI,
                },
            )
        data = _parse_tool_result(result=result)

        checks = {c["name"]: c for c in data["checks"]}

        assert checks["Namespace"]["passed"], f"Namespace check failed: {checks['Namespace']['message']}"
        assert checks["Serving runtime"]["passed"], f"Runtime check failed: {checks['Serving runtime']['message']}"
        assert checks["Storage"]["passed"], f"Storage check failed: {checks['Storage']['message']}"

    @pytest.mark.dependency(name="create_runtime", depends=["check_prereqs"])
    async def test_create_serving_runtime(
        self,
        mcp_model_deployer_transport: StreamableHttpTransport,
        mcp_model_deploy_namespace: Namespace,
    ) -> None:
        """Given the kserve-ovms template exists in the platform namespace
        When create_serving_runtime is called for the model deployment namespace
        Then a ServingRuntime is created that supports openvino_ir
        """
        async with Client(mcp_model_deployer_transport) as client:
            result = await client.call_tool(
                name="create_serving_runtime",
                arguments={
                    "namespace": mcp_model_deploy_namespace.name,
                    "template_name": RHOAI_MCP_MODEL_DEPLOY_RUNTIME_TEMPLATE,
                },
            )
        data = _parse_tool_result(result=result)

        assert data.get("success") is True, f"create_serving_runtime failed: {data}"
        assert RHOAI_MCP_MODEL_DEPLOY_FORMAT in data.get("supported_formats", []), (
            f"Created runtime does not support {RHOAI_MCP_MODEL_DEPLOY_FORMAT}: {data.get('supported_formats')}"
        )

    @pytest.mark.dependency(name="deploy_model", depends=["create_runtime"])
    async def test_deploy_model(
        self,
        mcp_model_deployer_transport: StreamableHttpTransport,
        mcp_model_deploy_namespace: Namespace,
    ) -> None:
        """Given a namespace with an OVMS serving runtime
        When deploy_model is called with the MNIST OCI ModelCar image
        Then an InferenceService is created successfully
        """
        async with Client(mcp_model_deployer_transport) as client:
            # Discover the runtime name that was created from the template
            rt_result = await client.call_tool(
                name="list_serving_runtimes",
                arguments={
                    "namespace": mcp_model_deploy_namespace.name,
                    "include_templates": False,
                },
            )
            rt_data = _parse_tool_result(result=rt_result)
            existing_runtimes = rt_data["result"]
            assert existing_runtimes, "No serving runtime found in namespace after creation"
            runtime_name = existing_runtimes[0]["name"]

            result = await client.call_tool(
                name="deploy_model",
                arguments={
                    "name": RHOAI_MCP_MODEL_DEPLOY_NAME,
                    "namespace": mcp_model_deploy_namespace.name,
                    "runtime": runtime_name,
                    "model_format": RHOAI_MCP_MODEL_DEPLOY_FORMAT,
                    "storage_uri": STORAGE_URI,
                },
            )
        data = _parse_tool_result(result=result)

        assert data["name"] == RHOAI_MCP_MODEL_DEPLOY_NAME
        assert data["namespace"] == mcp_model_deploy_namespace.name

    @pytest.mark.dependency(name="model_ready", depends=["deploy_model"])
    async def test_model_reaches_ready(
        self,
        mcp_model_deployer_transport: StreamableHttpTransport,
        mcp_model_deploy_namespace: Namespace,
    ) -> None:
        """Given a newly deployed InferenceService
        When get_inference_service is polled over time
        Then the model eventually reports status Ready
        """
        async with Client(mcp_model_deployer_transport) as client:
            data = await _wait_for_model_ready(
                client=client,
                name=RHOAI_MCP_MODEL_DEPLOY_NAME,
                namespace=mcp_model_deploy_namespace.name,
            )

        assert data["status"] == "Ready"
        assert data.get("model_format") == RHOAI_MCP_MODEL_DEPLOY_FORMAT
        assert data.get("storage_uri") == STORAGE_URI

    @pytest.mark.dependency(name="endpoint_accessible", depends=["model_ready"])
    async def test_model_endpoint_accessible(
        self,
        mcp_model_deployer_transport: StreamableHttpTransport,
        mcp_model_deploy_namespace: Namespace,
    ) -> None:
        """Given a model that has reached Ready status
        When get_model_endpoint and test_model_endpoint are called
        Then the endpoint URL is populated and the model is accessible
        """
        async with Client(mcp_model_deployer_transport) as client:
            endpoint_result = await client.call_tool(
                name="get_model_endpoint",
                arguments={
                    "name": RHOAI_MCP_MODEL_DEPLOY_NAME,
                    "namespace": mcp_model_deploy_namespace.name,
                },
            )
            endpoint_data = _parse_tool_result(result=endpoint_result)

            assert endpoint_data["status"] == "Ready"
            assert endpoint_data.get("url"), "Model endpoint URL is empty"

            test_result = await client.call_tool(
                name="test_model_endpoint",
                arguments={
                    "name": RHOAI_MCP_MODEL_DEPLOY_NAME,
                    "namespace": mcp_model_deploy_namespace.name,
                },
            )
            test_data = _parse_tool_result(result=test_result)

        assert test_data["accessible"] is True, f"Model endpoint not accessible: {test_data.get('issues')}"

    @pytest.mark.dependency(depends=["model_ready"])
    async def test_model_appears_in_listing(
        self,
        mcp_model_deployer_transport: StreamableHttpTransport,
        mcp_model_deploy_namespace: Namespace,
    ) -> None:
        """Given a deployed and ready model
        When list_inference_services is called for the namespace
        Then the model appears in the listing with Ready status
        """
        async with Client(mcp_model_deployer_transport) as client:
            result = await client.call_tool(
                name="list_inference_services",
                arguments={"namespace": mcp_model_deploy_namespace.name},
            )
        data = _parse_tool_result(result=result)

        items = data.get("items", [])
        names = [item["name"] for item in items]
        assert RHOAI_MCP_MODEL_DEPLOY_NAME in names, (
            f"Model '{RHOAI_MCP_MODEL_DEPLOY_NAME}' not found in listing: {names}"
        )

        model_entry = next(i for i in items if i["name"] == RHOAI_MCP_MODEL_DEPLOY_NAME)
        assert model_entry["status"] == "Ready"

    # Namespace teardown handles cleanup for now; will enable this test once
    # owned operations are allowed in the rhoai-mcp codebase.
    @pytest.mark.skip(reason="Requires owned operations are allowed in the rhoai-mcp, to be enabled later")
    @pytest.mark.dependency(depends=["model_ready"])
    async def test_delete_inference_service(
        self,
        mcp_model_deployer_transport: StreamableHttpTransport,
        mcp_model_deploy_namespace: Namespace,
    ) -> None:
        """Given a deployed model
        When delete_inference_service is called with confirm=True
        Then the model is deleted successfully
        """
        async with Client(mcp_model_deployer_transport) as client:
            result = await client.call_tool(
                name="delete_inference_service",
                arguments={
                    "name": RHOAI_MCP_MODEL_DEPLOY_NAME,
                    "namespace": mcp_model_deploy_namespace.name,
                    "confirm": True,
                },
            )
        data = _parse_tool_result(result=result)

        assert data["deleted"] is True
        assert data["name"] == RHOAI_MCP_MODEL_DEPLOY_NAME
