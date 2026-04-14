import math
import os
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from llama_stack_client import APIConnectionError, InternalServerError, LlamaStackClient
from llama_stack_client.types.file import File
from llama_stack_client.types.vector_store import VectorStore
from llama_stack_client.types.vector_stores.vector_store_file import VectorStoreFile
from ocp_resources.pod import Pod
from openai import OpenAI
from ragas import SingleTurnSample
from ragas.llms import llm_factory
from timeout_sampler import retry

from tests.llama_stack.constants import (
    LLS_CORE_POD_FILTER,
    RAGAS_MAX_SAMPLES,
    ModelInfo,
)
from tests.llama_stack.datasets import Dataset
from utilities.exceptions import UnexpectedResourceCountError
from utilities.path_utils import resolve_repo_path
from utilities.resources.llama_stack_distribution import LlamaStackDistribution

LOGGER = structlog.get_logger(name=__name__)


def _assert_file_uploaded(uploaded_file: File, expected_purpose: str) -> None:
    """Validate that the Files API response indicates a successful upload."""
    assert uploaded_file.id, f"Uploaded file has no id: {uploaded_file}"
    assert uploaded_file.bytes > 0, f"Uploaded file reports 0 bytes: {uploaded_file}"
    assert uploaded_file.filename, f"Uploaded file has no filename: {uploaded_file}"
    assert uploaded_file.purpose == expected_purpose, (
        f"Expected purpose '{expected_purpose}', got '{uploaded_file.purpose}'"
    )
    LOGGER.info(
        f"File uploaded successfully: id={uploaded_file.id}, "
        f"filename={uploaded_file.filename}, bytes={uploaded_file.bytes}"
    )


def _assert_vector_store_file_attached(
    filename: str,
    vs_file: VectorStoreFile,
    vector_store_id: str,
    *,
    attributes: dict[str, str | int | float | bool] | None = None,
) -> None:
    """Validate that the vector store file was attached successfully.

    Per the OpenAI Vector Store Files API, status may be in_progress (processing)
    or completed (ready for use). We require that the file is not failed or cancelled.
    """
    assert vs_file.id, f"Vector store file has no id: {vs_file}"
    assert vs_file.vector_store_id == vector_store_id, (
        f"Vector store file vector_store_id {vs_file.vector_store_id!r} does not match expected {vector_store_id!r}"
    )
    assert vs_file.status != "failed", (
        f"Vector store file is failed: filename={filename} id={vs_file.id}, last_error={vs_file.last_error!r}"
    )
    assert vs_file.status != "cancelled", f"Vector store file was cancelled: filename={filename} id={vs_file.id}"
    if attributes:
        assert vs_file.attributes, f"Expected attributes on vector store file {vs_file.id} but got none"
        for key, expected in attributes.items():
            actual = vs_file.attributes.get(key)
            assert actual == expected, (
                f"Attribute mismatch on file {vs_file.id}: {key!r} expected {expected!r}, got {actual!r}"
            )
    LOGGER.info(
        f"File attached to vector store: filename={filename} id={vs_file.id}, "
        f"vector_store_id={vs_file.vector_store_id}, status={vs_file.status}"
    )


def vector_store_create_and_poll(
    llama_stack_client: LlamaStackClient,
    vector_store_id: str,
    file_id: str,
    *,
    attributes: dict[str, str | int | float | bool] | None = None,
    poll_interval_sec: float = 5.0,
    wait_timeout: float = 300.0,
) -> VectorStoreFile:
    """Attach a file to a vector store and poll until processing finishes.

    Mirrors the OpenAI Python SDK create_and_poll pattern: create the vector store
    file, then repeatedly retrieve until status is completed, failed, or cancelled.

    Args:
        llama_stack_client: The configured LlamaStackClient.
        vector_store_id: The vector store to attach the file to.
        file_id: The file ID (from files.create) to attach.
        attributes: Optional attributes to associate with the file.
        poll_interval_sec: Seconds to wait between poll attempts.
        wait_timeout: Total seconds to wait for a terminal status before raising.

    Returns:
        The final VectorStoreFile (caller should check status and last_error).

    Raises:
        TimeoutError: If wait_timeout is reached while status is still in_progress.
    """
    start = time.monotonic()
    request_timeout = max(1, int(wait_timeout - (time.monotonic() - start)))
    vs_file = llama_stack_client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=file_id,
        timeout=request_timeout,
        attributes=dict(attributes) if attributes else attributes,
    )
    terminal_statuses = ("completed", "failed", "cancelled")
    deadline = start + wait_timeout

    while vs_file.status == "in_progress":
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Vector store file {vs_file.id} still in_progress after {wait_timeout}s")
        time.sleep(poll_interval_sec)
        vs_file = llama_stack_client.vector_stores.files.retrieve(file_id=vs_file.id, vector_store_id=vector_store_id)

    if vs_file.status not in terminal_statuses:
        LOGGER.warning(f"Unexpected vector store file status {vs_file.status!r}, treating as terminal")
    return vs_file


@contextmanager
def create_llama_stack_distribution(
    client: DynamicClient,
    name: str,
    namespace: str,
    replicas: int,
    server: dict[str, Any],
    teardown: bool = True,
) -> Generator[LlamaStackDistribution, Any, Any]:
    """
    Context manager to create and optionally delete a LLama Stack Distribution
    """

    # Starting with RHOAI 3.3, pods in the 'openshift-ingress' namespace must be allowed
    # to access the llama-stack-service. This is required for the llama_stack_test_route
    # to function properly.
    network: dict[str, Any] = {
        "allowedFrom": {
            "namespaces": ["openshift-ingress"],
        },
    }

    with LlamaStackDistribution(
        client=client,
        name=name,
        namespace=namespace,
        replicas=replicas,
        network=network,
        server=server,
        wait_for_resource=True,
        teardown=teardown,
    ) as llama_stack_distribution:
        yield llama_stack_distribution


@retry(
    wait_timeout=240,
    sleep=5,
    exceptions_dict={ResourceNotFoundError: [], UnexpectedResourceCountError: []},
)
def wait_for_unique_llama_stack_pod(client: DynamicClient, namespace: str) -> Pod:
    """Wait until exactly one LlamaStackDistribution pod is found in the
    namespace (multiple pods may indicate known bug RHAIENG-1819)."""
    pods = list(
        Pod.get(
            client=client,
            namespace=namespace,
            label_selector=LLS_CORE_POD_FILTER,
        )
    )
    if not pods:
        raise ResourceNotFoundError(f"No pods found with label selector {LLS_CORE_POD_FILTER} in namespace {namespace}")
    if len(pods) != 1:
        raise UnexpectedResourceCountError(
            f"Expected exactly 1 pod with label selector {LLS_CORE_POD_FILTER} "
            f"in namespace {namespace}, found {len(pods)}. "
            f"(possibly due to known bug RHAIENG-1819)"
        )
    return pods[0]


@retry(wait_timeout=90, sleep=5)
def wait_for_llama_stack_client_ready(client: LlamaStackClient) -> bool:
    """Wait for LlamaStack client to be ready by checking health, version, and database access."""
    try:
        client.inspect.health()
        version = client.inspect.version()
        models = client.models.list()
        vector_stores = client.vector_stores.list()
        files = client.files.list()
        LOGGER.info(
            f"Llama Stack server is available! "
            f"(version:{version.version} "
            f"models:{len(models)} "
            f"vector_stores:{len(vector_stores.data)} "
            f"files:{len(files.data)})"
        )

    except (APIConnectionError, InternalServerError) as error:
        LOGGER.debug(f"Llama Stack server not ready yet: {error}")
        LOGGER.debug(f"Base URL: {client.base_url}, Error type: {type(error)}, Error details: {error!s}")
        return False

    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"Unexpected error checking Llama Stack readiness: {e}")
        return False

    else:
        return True


def vector_store_create_file_from_url(
    url: str, llama_stack_client: LlamaStackClient, vector_store: Any
) -> VectorStoreFile:
    """
    Downloads a file from URL to a temporally file and uploads it to the files provider (files.create)
    and to the vector_store (vector_stores.files.create)

    Args:
        url: The URL to download the file from
        llama_stack_client: The configured LlamaStackClient
        vector_store: The vector store to upload the file to

    Returns:
        The vector store file after processing completes. Raises on failure.
    """
    temp_file_path = None
    try:
        LOGGER.info(f"Downloading remote file (url={url})")
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        path_part = url.split("/")[-1].split("?")[0]

        if content_type == "application/pdf" or path_part.lower().endswith(".pdf"):
            file_suffix = ".pdf"
        elif path_part.lower().endswith(".rst"):
            file_suffix = "_" + path_part.replace(".rst", ".txt")
        else:
            file_suffix = "_" + (path_part or "document.txt")

        with tempfile.NamedTemporaryFile(mode="wb", suffix=file_suffix, delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = Path(temp_file.name)  # noqa: FCN001
            LOGGER.info(f"Stored remote file (url={url}) into temporal file (temp_file_path={temp_file_path})")
            return vector_store_create_file_from_path(
                file_path=temp_file_path, llama_stack_client=llama_stack_client, vector_store=vector_store
            )

    except (requests.exceptions.RequestException, Exception) as e:
        LOGGER.warning(f"Failed to download remote file (url={url}) and attach it to vector store: {e}")
        raise
    finally:
        if temp_file_path is not None:
            os.unlink(temp_file_path)


def vector_store_create_file_from_path(
    file_path: Path,
    llama_stack_client: LlamaStackClient,
    vector_store: Any,
    *,
    attributes: dict[str, str | int | float | bool] | None = None,
) -> VectorStoreFile:
    """
    Uploads a local file to the files provider (files.create) and adds it to the
    vector store (vector_stores.files.create).

    Args:
        file_path: Path to the local file to upload
        llama_stack_client: The configured LlamaStackClient
        vector_store: The vector store to add the file to
        attributes: Optional attributes to associate with the vector-store file.

    Returns:
        The vector store file after processing completes.

    Raises:
        FileNotFoundError: If the file does not exist
        Exception: If the upload fails
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    LOGGER.info(f"Uploading local file {file_path.name} to the llama-stack files provider")
    with open(file_path, "rb") as file_to_upload:
        uploaded_file = llama_stack_client.files.create(file=file_to_upload, purpose="assistants")
        _assert_file_uploaded(uploaded_file=uploaded_file, expected_purpose="assistants")
    LOGGER.info(f"Uploaded {file_path.name} (file_id={uploaded_file.id}) to the llama-stack files provider")

    LOGGER.info(f"Adding uploaded file (filename{uploaded_file.filename} to vector store {vector_store.id}")
    vs_file = vector_store_create_and_poll(
        llama_stack_client=llama_stack_client,
        vector_store_id=vector_store.id,
        file_id=uploaded_file.id,
        attributes=attributes,
    )
    _assert_vector_store_file_attached(
        filename=uploaded_file.filename, vs_file=vs_file, vector_store_id=vector_store.id, attributes=attributes
    )
    LOGGER.info(f"Added uploaded file (filename{uploaded_file.filename} to vector store {vector_store.id}")
    return vs_file


def vector_store_upload_doc_sources(
    doc_sources: list[str],
    llama_stack_client: LlamaStackClient,
    vector_store: Any,
    vector_io_provider: str,
) -> None:
    """Upload document sources (URLs and repo-local paths) to a vector store.

    Resolves each local path via ``resolve_repo_path`` and re-resolves directory entries
    to avoid symlink escape outside the repository.

    Args:
        doc_sources: List of URL or path strings (repo-relative or absolute under repo root).
        llama_stack_client: Client used for file and vector store APIs.
        vector_store: Target vector store (must expose ``id``).
        vector_io_provider: Provider id for log context only.

    Raises:
        ValueError: If a local path resolves outside the repo root.
        FileNotFoundError: If a file or non-empty directory source is missing.
    """
    LOGGER.info(
        f"Uploading doc_sources to vector_store (provider_id={vector_io_provider}, id={vector_store.id}): {doc_sources}"
    )
    for source in doc_sources:
        if source.startswith(("http://", "https://")):
            vector_store_create_file_from_url(
                url=source,
                llama_stack_client=llama_stack_client,
                vector_store=vector_store,
            )
            continue
        source_path = resolve_repo_path(source=source)

        if source_path.is_dir():
            files = sorted(source_path.iterdir())
            if not files:
                raise FileNotFoundError(f"No files found in directory: {source_path}")
            for file_path in files:
                file_path_resolved = resolve_repo_path(source=file_path)
                if not file_path_resolved.is_file():
                    continue
                vector_store_create_file_from_path(
                    file_path=file_path_resolved,
                    llama_stack_client=llama_stack_client,
                    vector_store=vector_store,
                )
        elif source_path.is_file():
            vector_store_create_file_from_path(
                file_path=source_path,
                llama_stack_client=llama_stack_client,
                vector_store=vector_store,
            )
        else:
            raise FileNotFoundError(f"Document source not found: {source_path}")


def vector_store_upload_dataset(
    dataset: Dataset,
    llama_stack_client: LlamaStackClient,
    vector_store: Any,
) -> None:
    """Upload all documents from a ``Dataset`` to a vector store.

    Each ``DatasetDocument`` is uploaded via the Files API and attached to
    the vector store.  When a document carries non-empty ``attributes``,
    they are set on the resulting vector-store file.

    Args:
        dataset: Dataset whose ``documents`` will be uploaded.
        llama_stack_client: Client used for file and vector store APIs.
        vector_store: Target vector store (must expose ``id``).
    """
    LOGGER.info(f"Uploading dataset ({len(dataset.documents)} document(s)) to vector_store (id={vector_store.id})")
    for doc in dataset.documents:
        source_path = resolve_repo_path(source=doc.path)
        vector_store_create_file_from_path(
            file_path=source_path,
            llama_stack_client=llama_stack_client,
            vector_store=vector_store,
            attributes=doc.attributes,
        )


def extract_retrieved_contexts(response: Any) -> list[str]:
    """
    Extract retrieved contexts from a LlamaStack Responses API output.

    Args:
        response: Response object from client.responses.create()

    Returns:
        List of retrieved context strings
    """
    retrieved_contexts = []

    for output_item in response.output:
        if (
            hasattr(output_item, "type")
            and output_item.type == "file_search_call"
            and hasattr(output_item, "results")
            and output_item.results
        ):
            for result in output_item.results:
                if hasattr(result, "text") and result.text:
                    retrieved_contexts.append(result.text)

    return retrieved_contexts


def mean_ragas_score(scores: list[float | None]) -> float:
    """Compute mean of RAGAS per-sample scores, filtering out NaN values."""
    valid = [s for s in scores if s is not None and not math.isnan(s)]
    return sum(valid) / len(valid) if valid else 0.0


@pytest.fixture(scope="class")
def ragas_evaluator_llm(
    unprivileged_llama_stack_client: LlamaStackClient,
    llama_stack_models: ModelInfo,
) -> Generator[Any, Any, Any]:
    """Create a RAGAS evaluator LLM backed by the Llama Stack OpenAI-compatible endpoint."""
    base_url = str(unprivileged_llama_stack_client.base_url).rstrip("/")
    verify_ssl = os.getenv("LLS_CLIENT_VERIFY_SSL", "false").lower() == "true"

    http_client = httpx.Client(verify=verify_ssl, timeout=httpx.Timeout(240.0))
    try:
        openai_client = OpenAI(
            api_key=os.getenv("LLS_CORE_VLLM_API_TOKEN", ""),
            base_url=f"{base_url}/v1",
            http_client=http_client,
        )

        evaluator_llm = llm_factory(
            model=llama_stack_models.model_id,
            provider="openai",
            client=openai_client,
        )
        evaluator_llm.model_args["max_tokens"] = 4096

        yield evaluator_llm
    finally:
        http_client.close()


@pytest.fixture(scope="class")
def ragas_evaluator_embeddings(
    unprivileged_llama_stack_client: LlamaStackClient,
    llama_stack_models: ModelInfo,
) -> Generator[Any, Any, Any]:
    """Create RAGAS embeddings backed by the Llama Stack sentence-transformers provider.

    Uses langchain_openai.OpenAIEmbeddings because the old ragas.metrics.AnswerRelevancy
    calls embed_query()/embed_documents() (LangChain interface), which the newer
    ragas.embeddings.OpenAIEmbeddings does not implement.
    """
    base_url = str(unprivileged_llama_stack_client.base_url).rstrip("/")
    verify_ssl = os.getenv("LLS_CLIENT_VERIFY_SSL", "false").lower() == "true"

    from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings

    http_client = httpx.Client(verify=verify_ssl, timeout=httpx.Timeout(120.0))
    try:
        embeddings = LangchainOpenAIEmbeddings(
            openai_api_key="false",  # pragma: allowlist secret
            openai_api_base=f"{base_url}/v1",
            model=llama_stack_models.embedding_model.id,
            http_client=http_client,
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )

        yield embeddings
    finally:
        http_client.close()


@pytest.fixture(scope="class")
def ragas_samples(
    unprivileged_llama_stack_client: LlamaStackClient,
    llama_stack_models: ModelInfo,
    vector_store: VectorStore,
    dataset: Dataset,
) -> list[SingleTurnSample]:
    """Build RAGAS evaluation samples by querying the RAG pipeline for each ground-truth QA pair.

    Uses the Responses API with the file_search tool against the vector store,
    mirroring a real-world RAG scenario.  The number of questions sent to the
    LLM is capped by ``RAGAS_MAX_SAMPLES`` (env var, default 5).
    """
    qa_records = dataset.load_qa(retrieval_mode="vector")[:RAGAS_MAX_SAMPLES]
    samples: list[SingleTurnSample] = []

    for i, record in enumerate(qa_records):
        LOGGER.info(f"[{i + 1}/{len(qa_records)}] {record.question[:80]}...")

        try:
            resp = unprivileged_llama_stack_client.responses.create(
                model=llama_stack_models.model_id,
                instructions=(
                    "/no_think\n"
                    "You are a helpful assistant with access to data via the file_search tool.\n\n"
                    "When asked questions, use available tools to find the answer. Follow these rules:\n"
                    "1. Use tools immediately without asking for confirmation\n"
                    "2. Chain tool calls as needed\n"
                    "3. Do not narrate your process\n"
                    "4. Only provide the final answer\n"
                    "5. If the answer is not found in the context, respond with 'I don't know'"
                ),
                tools=[{"type": "file_search", "vector_store_ids": [vector_store.id]}],
                stream=False,
                input=record.question,
            )

            rag_answer = resp.output_text.strip()
            retrieved_contexts = extract_retrieved_contexts(response=resp)

        except Exception:
            LOGGER.exception(f"RAG call failed for question: {record.question!r}, skipping sample")
            continue

        if not rag_answer:
            LOGGER.warning(f"Empty RAG response for question: {record.question!r}, skipping sample")
            continue
        if not retrieved_contexts:
            LOGGER.warning(f"No retrieved contexts for question: {record.question!r}, skipping sample")
            continue

        samples.append(
            SingleTurnSample(
                user_input=record.question,
                retrieved_contexts=retrieved_contexts,
                response=rag_answer,
                reference=record.ground_truth,
            )
        )

        LOGGER.info(f"  Answer: {rag_answer[:120]}...")
        LOGGER.info(f"  Retrieved {len(retrieved_contexts)} context(s)")

    assert samples, "Failed to build any RAGAS samples from the dataset QA records"
    LOGGER.info(f"Built {len(samples)} RAGAS evaluation samples from {len(qa_records)} QA records")
    return samples
