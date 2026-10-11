# AutoRAG startup data

The smoke fixture uploads `sample.md` and a matching, generated benchmark JSON
record to the test DSPA's MinIO. Its benchmark document key follows the configured
MinIO prefix. These inputs are sufficient to submit the managed AutoRAG pipeline;
the test only verifies registration and initial workflow progression.

The required connection Secrets are created for pipeline submission. The smoke
test does not validate model inference, document processing, vector storage, or
completed RAG optimization.
