VLLM_INFERENCE_CONFIG = {
        "default_query_model": {
            "query_input": '"prompt": "At what temperature does Nitrogen boil?", "max_tokens": 100, "temperature": 0',
            "query_output": {
                "response_output": '{"model_name":"$model_name","model_version":"1",'
                '"outputs":[{"name":"Plus214_Output_0","shape":[1,10],'
                r'"datatype":"FP32","data":\[.*\]}]}',
            },
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
