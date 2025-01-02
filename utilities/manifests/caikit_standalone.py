CAIKIT_STANDALONE_INFERENCE_CONFIG = {
        "check_regex_response": True,
        "default_query_model": {
            "query_input": "At what temperature does Nitrogen boil?",
            "query_output": r'{"result": \{.*?\}, "producer_id": {"name": "EmbeddingModule", "version": "\d.\d.\d"}, "input_token_count": \d+}',
        },
        "embedding": {
            "http": {
                "endpoint": "api/v1/task/embedding",
                "header": "Content-type:application/json",
                "body": '{"model_id": "$model_name","inputs": "$query_input"}',
                "response_fields_map": {
                    "response_output": "output",
                },
            },
        },
        "rerank": {
            "http": {
                "endpoint": "api/v1/task/rerank",
                "header": "Content-type:application/json",
                "body": '{"model_id": "$model_name","inputs": "$query_input"}',
                "response_fields_map": {"response_output": "output"},
            },
        },
        "sentence-similarity": {
        "http": {
            "endpoint": "api/v1/task/sentence-similarity",
            "header": "Content-type:application/json",
            "body": '{"model_id": "$model_name","inputs": "$query_input"}',
            "response_fields_map": {"response_output": "output"},
        },
    },
    }
