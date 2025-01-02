CAIKIT_STANDALONE_INFERENCE_CONFIG = {
    "support_multi_default_queries": True,
    "default_query_model": {
        "embedding": {
            "query_input": "At what temperature does Nitrogen boil?",
            "query_output": r'{"result": \{.*\}, "producer_id": {"name": "EmbeddingModule", "version": "\d.\d.\d"}, "input_token_count": \d+}',
            "use_regex": True,
        },
        "rerank": {
            "query_input": '{"documents": [{"text": "At what temperature does Nitrogen boil?", "title": "Nitrogen Boil"}, {"text": "Cooling Temperature for Nitrogen is different", "more": "There are other features on Nitrogen"}, {"text": "What elements could be used Nitrogen, Helium", "meta": {"date": "today", "i": 999, "f": 12.34}}], "query": "At what temperature does liquid Nitrogen boil?"},"parameters": {"top_n":293}',
            "query_output": r'{"result": \{.*\}, "producer_id": {"name": "EmbeddingModule", "version": "\d.\d.\d"}, "input_token_count": \d+}',
            "use_regex": True,
        },
        "sentence-similarity": {
            "query_input": '{"source_sentence": "At what temperature does liquid Nitrogen boil?", "sentences": ["At what temperature does liquid Nitrogen boil", "Hydrogen boils and cools at temperatures"]}',
            "query_output": r'{"result": \{.*\}, "producer_id": {"name": "EmbeddingModule", "version": "\d.\d.\d"}, "input_token_count": \d+}',
            "use_regex": True,
        },
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
            "body": '{"model_id": "$model_name","inputs": $query_input}',
            "response_fields_map": {"response_output": "output"},
        },
    },
    "sentence-similarity": {
        "http": {
            "endpoint": "api/v1/task/sentence-similarity",
            "header": "Content-type:application/json",
            "body": '{"model_id": "$model_name","inputs": $query_input}',
            "response_fields_map": {"response_output": "output"},
        },
    },
}
