DEFAULT_INPUT = [
            {
                "name": "Func/StatefulPartitionedCall/input/_0:0",
                "shape": [1, 30],
                "datatype": "FP32",
                "data": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
            }
        ]

INFER_ENDPOINT = "v2/models/$model_name/infer"
RESPONSE_FIELD_MAP = {
        "response_output": "output",
    }
DEFAULT_HTTP_QUERY = {
    "endpoint": INFER_ENDPOINT,
    "header": "Content-type:application/json",
    "body": '{"inputs": $query_input}',
    "response_fields_map": RESPONSE_FIELD_MAP,
}
DEFAULT_FORM_QUERY = {
                "endpoint": INFER_ENDPOINT,
                "header": "application/x-www-form-urlencoded",
                "body": '$query_input',
                "response_fields_map": RESPONSE_FIELD_MAP,
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
        "infer-mnist": {
            "query_input": "@utilities/manifests/openvino/mnist-input.json",
            "query_output": r'{"model_name":"${model_name}","model_version":"1","outputs":\[{"name":"Plus214_Output_0","shape":\[1,10\],"datatype":"FP32","data":\[.*\]}\]}',
            "use_regex": True
        },
    },
    "infer": {
        "http": DEFAULT_HTTP_QUERY,
    },
    "infer-vehicle-detection":
        {
            "http": DEFAULT_FORM_QUERY,
        },
    "infer-mnist": {
            "http": DEFAULT_FORM_QUERY,
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
