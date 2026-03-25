# Llama Stack test fixtures (internal)

These files are for **internal Open Data Hub / OpenShift AI integration tests** only. We use them to hit **[Llama Stack](https://github.com/meta-llama/llama-stack) vector store APIs**---think ingest, indexing, search, and the plumbing around that---not as a shipped dataset or for model training.

## Folder layout

```text
dataset/
  corpus/
    <subject>/                    # one folder per subject (e.g. "finance")
      *.pdf                       # source documents
      documents.json              # manifest: per-file metadata & attributes
    pdf-testing/                  # special: edge-case PDFs (encrypted, signed, etc.)
  ground_truth/
    <subject>_qa.jsonl            # QA pairs for the corresponding subject
```

### Current contents

| Path | Description |
| ---- | ----------- |
| `corpus/finance/` | IBM quarterly earnings press releases (Q1–Q4 2025, 4 PDFs) |
| `corpus/finance/documents.json` | Manifest with per-file attributes (entity, period, doc type, etc.) |
| `corpus/pdf-testing/` | Edge-case PDFs: password-encrypted, PAdES-signed |
| `ground_truth/finance_qa.jsonl` | 60 QA pairs across 4 IBM quarters, 3 retrieval modes each |

## Using `datasets.py`

The module `tests/llama_stack/datasets.py` provides a lightweight, dependency-free loader for the files in this directory. It exposes typed dataclasses and pre-built `Dataset` instances that tests and fixtures can import directly.

### Core types

| Class | Purpose |
| ----- | ------- |
| `DatasetDocument` | A single document: resolved `path`, `document_id`, and optional `attributes` dict |
| `DatasetDocumentQA` | A single QA record: `id`, `question`, `ground_truth`, `retrieval_mode`, `document_id` |
| `Dataset` | Groups a `qa_path` and a tuple of `DatasetDocument`s; provides `load_qa()` |

### Pre-built dataset instances

| Instance | Documents | Description |
| -------- | --------- | ----------- |
| `FINANCE_DATASET` | All 4 IBM quarterly PDFs | Full corpus; documents loaded from `corpus/finance/documents.json` |
| `IBM_2025_Q4_EARNINGS` | 1 (Q4 unencrypted) | Single-document subset for fast smoke tests |
| `IBM_2025_Q4_EARNINGS_ENCRYPTED` | 1 (Q4 encrypted) | Same content as Q4 but password-encrypted; for ingestion edge-case tests |

Subsets like `IBM_2025_Q4_EARNINGS` share the same `finance_qa.jsonl` as their `qa_path`. The `load_qa()` method automatically restricts returned QA records to the `document_id`s present in the subset's `documents`, so only relevant questions are returned.

### Loading QA records

```python
from tests.llama_stack.datasets import FINANCE_DATASET, IBM_2025_Q4_EARNINGS

# All QA records for the full corpus (60 records)
all_qa = FINANCE_DATASET.load_qa()

# Only "hybrid" retrieval-mode records
hybrid_qa = FINANCE_DATASET.load_qa(retrieval_mode="hybrid")

# QA records for a single-document subset (only Q4 questions)
q4_qa = IBM_2025_Q4_EARNINGS.load_qa()

# Combine filters: hybrid mode + specific document
q4_hybrid = IBM_2025_Q4_EARNINGS.load_qa(retrieval_mode="hybrid")
```

### Accessing documents and attributes

```python
from tests.llama_stack.datasets import FINANCE_DATASET

for doc in FINANCE_DATASET.documents:
    print(doc.path)          # "tests/llama_stack/dataset/corpus/finance/ibm-1q25-..."
    print(doc.document_id)   # "ibm_1q25_earnings_pr"
    print(doc.attributes)    # {"entity_symbol": "IBM", "period_year": 2025, ...}
```

### Using datasets in test parametrization

Tests receive a `Dataset` via indirect parametrize and a `dataset` fixture:

```python
from tests.llama_stack.datasets import FINANCE_DATASET, IBM_2025_Q4_EARNINGS, Dataset

@pytest.mark.parametrize(
    "vector_store, dataset",
    [
        pytest.param(
            {"vector_io_provider": "milvus", "dataset": IBM_2025_Q4_EARNINGS},
            IBM_2025_Q4_EARNINGS,
            id="milvus-single-doc",
        ),
        pytest.param(
            {"vector_io_provider": "faiss", "dataset": FINANCE_DATASET},
            FINANCE_DATASET,
            id="faiss-full-corpus",
        ),
    ],
    indirect=True,
)
class TestExample:
    def test_search(self, vector_store, dataset: Dataset) -> None:
        """Given: A populated vector store.
        When: QA queries are executed.
        Then: Results are returned for each retrieval mode.
        """
        for record in dataset.load_qa():
            # record.question, record.ground_truth, record.retrieval_mode
            ...
```

The `vector_store` fixture reads the `"dataset"` key from its param to upload the right documents; the `dataset` fixture hands the same `Dataset` instance to the test body for QA queries.

## Adding a new subject

1. Create `corpus/<subject>/` and drop in the source documents (PDFs, or any other format supported by the Llama Stack Files and Vector IO providers).
2. Create `corpus/<subject>/documents.json` (see schema below).
3. Create `ground_truth/<subject>_qa.jsonl` (see schema below).
4. Add a new `Dataset` instance in `tests/llama_stack/datasets.py` (follow the `FINANCE_DATASET` pattern).

## `documents.json` manifest

A JSON array where each entry maps a corpus file to uploadable attributes:

```json
[
  {
    "filename": "report-q1.pdf",
    "document_id": "report_q1",
    "attributes": {
      "entity_symbol": "ACME",
      "period_label": "2025-Q1",
      "period_year": 2025,
      "document_type": "earnings_press_release",
      "language": "en",
      "publication_date": 1745366400
    }
  }
]
```

- `filename`: file in the same directory as the manifest.
- `document_id`: stable identifier used to link QA records to their source.
- `attributes`: key-value pairs set on the vector-store file after upload (used for attribute filtering in search).

## `<subject>_qa.jsonl` schema

One JSON object per line:

```json
{
  "id": "acme-q1-hybrid-001",
  "question": "What was ACME's Q1 revenue?",
  "ground_truth": "Q1 revenue was $1.2 billion.",
  "retrieval_mode": "hybrid",
  "document_id": "report_q1"
}
```

- `id`: unique record identifier (`<entity>-<period>-<mode>-<seq>`).
- `question`: the query posed to the RAG system.
- `ground_truth`: reference answer (RAGAS convention for the expected answer).
- `retrieval_mode`: one of `vector`, `keyword`, or `hybrid`.
- `document_id`: links to `document_id` in `documents.json` for the source document.

## IBM finance PDFs (`corpus/finance/`)

The PDFs here are IBM **quarterly earnings press releases** (the same material IBM posts for investors). If you need to replace or refresh them, download the official PDFs from IBM's site:

[Quarterly earnings announcements](https://www.ibm.com/investor/financial-reporting/quarterly-earnings) (choose year and quarter, then open the press release PDF).

## PDF edge cases (`corpus/pdf-testing/`)

This folder is for **weird PDFs on purpose**: password-protected files, digitally signed ones (e.g. PAdES), and similar cases so we can test how ingestion and parsers behave when the file is not a plain "print to PDF" document.

## Small print

Not for external distribution as a "dataset." PDFs stay under their publishers' terms; don't reuse them outside this test context without checking those terms.
