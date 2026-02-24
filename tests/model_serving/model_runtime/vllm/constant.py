from typing import Any

from utilities.constants import AcceleratorType, KServeDeploymentType, Labels, RuntimeTemplates

OPENAI_ENDPOINT_NAME: str = "openai"
TGIS_ENDPOINT_NAME: str = "tgis"
# Quantization
VLLM_SUPPORTED_QUANTIZATION: list[str] = ["marlin", "awq"]
# Configurations
ACCELERATOR_IDENTIFIER: dict[str, str] = {
    AcceleratorType.NVIDIA: Labels.Nvidia.NVIDIA_COM_GPU,
    AcceleratorType.AMD: "amd.com/gpu",
    AcceleratorType.GAUDI: "habana.ai/gaudi",
}

TEMPLATE_MAP: dict[str, str] = {
    AcceleratorType.NVIDIA: RuntimeTemplates.VLLM_CUDA,
    AcceleratorType.AMD: RuntimeTemplates.VLLM_ROCM,
    AcceleratorType.GAUDI: RuntimeTemplates.VLLM_GAUDI,
}

PREDICT_RESOURCES: dict[str, list[dict[str, str | dict[str, str]]] | dict[str, dict[str, str]]] = {
    "volumes": [
        {"name": "shared-memory", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
        {"name": "tmp", "emptyDir": {}},
        {"name": "home", "emptyDir": {}},
    ],
    "volume_mounts": [
        {"name": "shared-memory", "mountPath": "/dev/shm"},
        {"name": "tmp", "mountPath": "/tmp"},
        {"name": "home", "mountPath": "/home/vllm"},
    ],
    "resources": {"requests": {"cpu": "2", "memory": "15Gi"}, "limits": {"cpu": "3", "memory": "16Gi"}},
}

COMPLETION_QUERY: list[dict[str, str]] = [
    {
        "text": "List the top five breeds of dogs and their characteristics.",
    },
    {
        "text": "Translate the following English sentence into Japanese, French, and Swahili: 'The early bird catches "
        "the worm.'"
    },
    {"text": "Write a short story about a robot that dreams for the first time."},
    {
        "text": "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in "
        "Western versus Eastern societies."
    },
    {
        "text": "Compare and contrast artificial intelligence with human intelligence in terms of "
        "processing information."
    },
    {"text": "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020."},
]

CHAT_QUERY: list[list[dict[str, str]]] = [
    [{"role": "user", "content": "Write python code to find even number"}],
    [
        {
            "role": "system",
            "content": "Given a target sentence, construct the underlying meaning representation of the input "
            "sentence as a single function with attributes and attribute values.",
        },
        {
            "role": "user",
            "content": "SpellForce 3 is a pretty bad game. The developer Grimlore Games is "
            "clearly a bunch of no-talent hacks, and 2017 was a terrible year for games anyway.",
        },
    ],
]

IMAGE_URL_SCENERY: str = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"  # noqa: E501
IMAGE_URL_DUCK: str = (
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
)
IMAGE_URL_LION: str = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"  # noqa: E501

MULTI_IMAGE_QUERIES: list[list[dict[Any, Any]]] = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": IMAGE_URL_SCENERY},
                },
            ],
        },
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are the animals in these images?"},
                {"type": "image_url", "image_url": {"url": IMAGE_URL_LION}},
                {"type": "image_url", "image_url": {"url": IMAGE_URL_DUCK}},
            ],
        }
    ],
]

THREE_IMAGE_QUERY: list[list[dict[Any, Any]]] = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Explain these images?"},
                {"type": "image_url", "image_url": {"url": IMAGE_URL_LION}},
                {"type": "image_url", "image_url": {"url": IMAGE_URL_DUCK}},
                {
                    "type": "image_url",
                    "image_url": {"url": IMAGE_URL_SCENERY},
                },
            ],
        }
    ],
]


LIGHTSPEED_TOOL_QUERY: list[list[dict[Any, Any]]] = [
    [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant with access to the following\nfunction calls. Your task is to produce a list of function calls\nnecessary to generate response to the user utterance. Use the following\nfunction calls as required.",  # noqa: E501
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "What pods are in the namespace openshift-lightspeed?"}]},
    ],
]

WEATHER_TOOL_QUERY: list[list[dict[Any, Any]]] = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather like in Boston today in celsius?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather like in Japan today in celsius?"},
    ],
]

LIGHTSPEED_TOOL: list[dict[Any, Any]] = [
    {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_object_namespace_list",
                    "description": "Get the list of all objects in a namespace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "str", "description": "the type of object"},
                            "namespace": {"type": "str", "description": "the name of the namespace"},
                        },
                        "required": ["kind", "namespace"],
                    },
                },
            }
        ],
    },
]

WEATHER_TOOL: list[dict[Any, Any]] = [
    {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
    }
]

MATH_CHAT_QUERY: list[list[dict[str, str]]] = [
    [{"role": "user", "content": "what is the sum of numbers between 1..10"}],
    [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is the sum of numbers between 1 and 123 using the formula n(n+1)/2? Explain it using chain-of-thought and solve with code",  # noqa: E501
        },
    ],
]


BASE_RAW_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
    "runtime_argument": None,
    "min-replicas": 1,
}


COMPLETION_QUERY_JAPANESE: list[dict[str, str]] = [
    {
        "text": "日本で一番高い山をjson形式で教えて。",
    },
    {
        "text": "graphvizで、AからB、BからC、CからAに有向エッジが生えているようなグラフを書きたいです。Markdown形式でコードを教えて"
    },
    {
        "text": "小説に登場させる魔法使いのキャラクターを考えています。主人公の師となるようなキャラクターの案を背景を含めて考えてください。"
    },
    {
        "text": "日本国内で観光に行きたいと思っています。東京、名古屋、大阪、京都、福岡の特徴を表にまとめてください。列名は「都道府県」「おすすめスポット」「おすすめグルメ」にしてください。"  # noqa: E501
    },
]

CHAT_QUERY_JAPANESE: list[list[dict[str, str]]] = [
    [
        {
            "role": "user",
            "content": "ランダムな10個の要素からなるリストを作成してソートするコードをPythonで書いてください。",
        }
    ],
    [
        {
            "role": "system",
            "content": "Given a target sentence, construct the underlying meaning representation of the input "
            "sentence as a single function with attributes and attribute values.",
        },
        {
            "role": "user",
            "content": "ルービックキューブをセンター試験の会場で、休憩時間に回そうと思っています。このような行動をしたときに周囲の人たちが感じるであろう感情について、3パターン程度述べてください。",  # noqa: E501
        },
    ],
]
