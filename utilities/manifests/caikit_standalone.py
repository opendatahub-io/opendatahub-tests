CAIKIT_STANDALONE_INFERENCE_CONFIG = {
        "default_query_model": {
            "query_input": "At what temperature does Nitrogen boil?",
            "query_output": "74 degrees F",
        },
        "embedding": {
            "http": {
                "endpoint": "api/v1/task/embedding",
                "header": "Content-type:application/json",
                "body": '{"model_id": "$model_name","inputs": "$query_input"}',
                "response_fields_map": {
                    "response_output": "result",
                },
            },
        },
        "rerank": {
            "http": {
                "endpoint": "api/v1/task/rerank",
                "header": "Content-type:application/json",
                "body": '{"model_id": "$model_name","inputs": "$query_input"}',
                "response_fields_map": {"response_output": "result"},
            },
        },
        "sentence-similarity": {
        "http": {
            "endpoint": "api/v1/task/sentence-similarity",
            "header": "Content-type:application/json",
            "body": '{"model_id": "$model_name","inputs": "$query_input"}',
            "response_fields_map": {"response_output": "result"},
        },
    },
    }
