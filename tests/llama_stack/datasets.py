"""Lightweight dataset loader for Llama Stack test fixtures.

Uses the standard ``json`` module to avoid heavy dependencies like
``datasets`` / ``pyarrow``.  Functions are generic path-based.
Well-known dataset instances (e.g. ``FINANCE_DATASET``) are defined
at the bottom of this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetDocumentQA:
    """A single question-answer pair tied to a source document."""

    id: str
    question: str
    ground_truth: str
    retrieval_mode: str
    document_id: str


@dataclass(frozen=True)
class DatasetDocument:
    """A document in a dataset with its resolved file path and optional attributes."""

    path: str
    document_id: str
    attributes: dict[str, str | int | float | bool] = field(default_factory=dict)


def _load_document_qa(
    path: str | Path,
    *,
    retrieval_mode: str | None = None,
    document_ids: list[str] | None = None,
) -> list[DatasetDocumentQA]:
    """Load QA records from a JSONL file.

    Args:
        path: Path to the ``.jsonl`` file (repo-relative or absolute).
        retrieval_mode: If given, only records matching this mode are returned.
        document_ids: If given, only records whose ``document_id`` is in this
            list are returned.

    Returns:
        List of ``DatasetDocumentQA`` instances.
    """
    allowed_ids = set(document_ids) if document_ids is not None else None
    records: list[DatasetDocumentQA] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if retrieval_mode is not None and raw.get("retrieval_mode") != retrieval_mode:
                continue
            if allowed_ids is not None and raw.get("document_id") not in allowed_ids:
                continue
            records.append(
                DatasetDocumentQA(
                    id=raw["id"],
                    question=raw["question"],
                    ground_truth=raw["ground_truth"],
                    retrieval_mode=raw["retrieval_mode"],
                    document_id=raw["document_id"],
                )
            )
    return records


def _load_documents_from_manifest(manifest_path: str) -> tuple[DatasetDocument, ...]:
    """Load documents from a JSON manifest, resolving paths relative to the manifest directory."""
    manifest_dir = str(Path(manifest_path).parent)
    raw_list = json.loads(Path(manifest_path).read_text())
    return tuple(
        DatasetDocument(
            path=f"{manifest_dir}/{entry['filename']}",
            document_id=entry["document_id"],
            attributes=entry.get("attributes", {}),
        )
        for entry in raw_list
    )


@dataclass(frozen=True)
class Dataset:
    """Corpus definition with documents and a QA ground-truth file.

    Each ``DatasetDocument`` in ``documents`` carries its resolved file path
    and optional per-file attributes.  Subsets (e.g. ``IBM_2025_Q4_EARNINGS``)
    are simply ``Dataset`` instances with fewer documents.
    """

    qa_path: str
    documents: tuple[DatasetDocument, ...]

    def _effective_document_ids(self, document_ids: list[str] | None) -> list[str] | None:
        own_ids = {d.document_id for d in self.documents}
        if document_ids is None:
            return list(own_ids) if self.documents else None
        return [d for d in document_ids if d in own_ids]

    def load_qa(
        self,
        *,
        retrieval_mode: str | None = None,
        document_ids: list[str] | None = None,
    ) -> list[DatasetDocumentQA]:
        """Load QA records, filtered by retrieval mode and/or document IDs.

        QA records are always restricted to the document IDs present in
        ``self.documents``; an explicit ``document_ids`` further narrows
        that set.
        """
        return _load_document_qa(
            path=self.qa_path,
            retrieval_mode=retrieval_mode,
            document_ids=self._effective_document_ids(document_ids),
        )


FINANCE_DATASET = Dataset(
    qa_path="tests/llama_stack/dataset/ground_truth/finance_qa.jsonl",
    documents=_load_documents_from_manifest("tests/llama_stack/dataset/corpus/finance/documents.json"),
)

# Subsets below reuse the shared finance_qa.jsonl as their qa_path.  This works
# because Dataset.load_qa() automatically restricts QA records to the document_ids
# present in self.documents, so only the questions relevant to the subset's
# documents are returned even though the JSONL contains entries for all quarters.
IBM_2025_Q4_EARNINGS = Dataset(
    qa_path="tests/llama_stack/dataset/ground_truth/finance_qa.jsonl",
    documents=(
        DatasetDocument(
            path="tests/llama_stack/dataset/corpus/finance/ibm-4q25-earnings-press-release-unencrypted.pdf",
            document_id="ibm_4q25_earnings_pr",
            attributes={
                "entity_symbol": "IBM",
                "period_label": "2025-Q4",
                "period_year": 2025,
                "document_type": "earnings_press_release",
                "language": "en",
                "publication_date": 1738022400,
            },
        ),
    ),
)

IBM_2025_Q4_EARNINGS_ENCRYPTED = Dataset(
    qa_path="tests/llama_stack/dataset/ground_truth/finance_qa.jsonl",
    documents=(
        DatasetDocument(
            path="tests/llama_stack/dataset/corpus/pdf-testing/ibm-4q25-press-release-encrypted.pdf",
            document_id="ibm_4q25_earnings_pr",
            attributes={
                "entity_symbol": "IBM",
                "period_label": "2025-Q4",
                "period_year": 2025,
                "document_type": "earnings_press_release",
                "language": "en",
                "publication_date": 1738022400,
            },
        ),
    ),
)
