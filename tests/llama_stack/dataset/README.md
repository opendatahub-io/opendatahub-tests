# Llama Stack test fixtures (internal)

These files are for **internal Open Data Hub / OpenShift AI integration tests** only. We use them to hit **[Llama Stack](https://github.com/meta-llama/llama-stack) vector store APIs**—think ingest, indexing, search, and the plumbing around that—not as a shipped dataset or for model training.

## IBM finance PDFs (`corpus/finance/`)

The PDFs here are IBM **quarterly earnings press releases** (the same material IBM posts for investors). If you need to replace or refresh them, download the official PDFs from IBM’s site:

[Quarterly earnings announcements](https://www.ibm.com/investor/financial-reporting/quarterly-earnings) (choose year and quarter, then open the press release PDF).

## PDF edge cases (`corpus/pdf-testing/`)

This folder is for **weird PDFs on purpose**: password-protected files, digitally signed ones (e.g. PAdES), and similar cases so we can test how ingestion and parsers behave when the file is not a plain “print to PDF” document.

## Small print

Not for external distribution as a “dataset.” PDFs stay under their publishers’ terms; don’t reuse them outside this test context without checking those terms.
