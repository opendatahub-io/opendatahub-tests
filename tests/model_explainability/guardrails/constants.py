from typing import Dict, Any

QWEN_ISVC_NAME = "qwen-isvc"

BUILTIN_DETECTOR_CONFIG: Dict[str, Any] = {
    "regex": {
        "type": "text_contents",
        "service": {
            "hostname": "127.0.0.1",
            "port": 8080,
        },
        "chunker_id": "whole_doc_chunker",
        "default_threshold": 0.5,
    }
}

CHAT_GENERATION_CONFIG: Dict[str, Any] = {
    "service": {
        "hostname": f"{QWEN_ISVC_NAME}-predictor",
        "port": 8032,
    }
}

EXAMPLE_EMAIL_ADDRESS: str = "myemail@domain.com"
PROMPT_WITH_PII: str = f"This is my email address: {EXAMPLE_EMAIL_ADDRESS}, just answer ACK."
