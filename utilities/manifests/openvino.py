DEFAULT_INPUT = [
            {
                "name": "Func/StatefulPartitionedCall/input/_0:0",
                "shape": [1, 30],
                "datatype": "FP32",
                "data": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
            }
        ]
DEFAULT_HTTP_QUERY = {
    "endpoint": "v2/models/$model_name/infer",
    "header": "Content-type:application/json",
    "body": '{"inputs": $query_input}',
    "response_fields_map": {
        "response_output": "output",
    },
}

OPENVINO_INFERENCE_CONFIG = {
    "support_multi_default_queries": True,
    "default_query_model": {
        "infer": {
            "query_input": DEFAULT_INPUT,
            "query_output": {
                "response_output": r'{"model_name":"${model_name}__isvc-[0-9a-z]+","model_version":"1",'
                r'"outputs":\[{"name":"Func/StatefulPartitionedCall/output/_13:0","datatype":"FP32",'
                r'"shape":\[1,1\],"data":\[0\]}]}',
            },
            "use_regex": True
        },
        "infer-vehicle-detection": {
            "query_input": "@utilities/manifests/openvino/vehicle-detection-inputs.txt",
            "query_output": r'{"model_name":"${model_name}__isvc-[0-9a-z]+","model_version":"1","outputs":\[{"name":"detection_out","datatype":"FP32","shape":\[1,1,200,7\],"data":\[.*\]}\]}',
            "use_regex": True
        },
    },
    "infer": {
        "http": DEFAULT_HTTP_QUERY,
    },
    "infer-vehicle-detection":
        {
            "http": {
                "endpoint": "v2/models/$model_name/infer",
                "header": "application/x-www-form-urlencoded",
                "body": '$query_input',
                "response_fields_map": {
                    "response_output": "output",
                },
            },
        }
}

OPENVINO_KSERVE_INFERENCE_CONFIG = {
    "default_query_model": {
        "query_input": DEFAULT_INPUT,
        "query_output": {
            "response_output": '{"model_name":"$model_name","model_version":"1",'
            '"outputs":[{"name":"Func/StatefulPartitionedCall/output/_13:0","shape":[1,1],'
            '"datatype":"FP32","data":[0.0]}]}',
        },
    },
    "infer": {
        "http": DEFAULT_HTTP_QUERY,
    },
}
