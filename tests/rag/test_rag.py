class TestRag:
    def test_rag_deployment(self, llama_stack_distribution_deployment):
        """
        Test that the Llama stack distribution deployment for
        RAG was created and it has a working pod.

        This verifies that:
        1. The Llama stack operator is up.
        2. It is possible to create a Llama stack distribution.
        3. A pod for the Llama stack distribution starts correctly.
        """
        llama_stack_distribution_deployment.wait_for_replicas()
