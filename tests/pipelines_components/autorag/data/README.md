# AutoRAG startup data

The smoke fixture uploads `sample.md` and a matching, generated benchmark JSON
record to the test DSPA's MinIO. Its benchmark document key follows the configured
MinIO prefix. These inputs are sufficient to submit the managed AutoRAG pipeline;
the test only verifies registration and initial workflow progression.

The required connection Secrets are created for pipeline submission. The smoke
test does not validate model inference, document processing, vector storage, or
completed RAG optimization.

The upgrade test uses the same inputs and submission logic. It saves the run and
workflow identities before upgrade, then checks that those records persist afterward.
The upgrade test likewise does not require live model or vector database services.
