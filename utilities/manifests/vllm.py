VLLM_INFERENCE_CONFIG = {
        "default_query_model": {
            "query_input": '"prompt": "At what temperature does Nitrogen boil?", "max_tokens": 100, "temperature": 0',
            "query_output": r'{"id":"cmpl-[a-z0-9]+","object":"text_completion","created":\d+,"model":"$model_name","choices":\[{"index":0,"text":".*Theboilingpointofnitrogenis77.4.*","logprobs":null,"finish_reason":"length","stop_reason":null.*"prompt_logprobs":null.*}\].*"usage":{"prompt_tokens":10,"total_tokens":110,"completion_tokens":100,"prompt_tokens_details":null}.*}',
            "use_regex": True
        },
        "completions": {
            "http": {
                "endpoint": "v1/completions",
                "header": "Content-type:application/json",
                "body": '{"model": "$model_name",$query_input}',
                "response_fields_map": {
                    "response_output": "output",
                },
            },
        },
    }
