import base64


def base64_encode_str(text: str) -> str:
    return base64.b64encode(text.encode()).decode()
