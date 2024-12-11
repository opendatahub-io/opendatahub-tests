class ProtocolNotSupported(Exception):
    def __init__(self, protocol: str):
        self.protocol = protocol

    def __str__(self) -> str:
        return f"Protocol {self.protocol} is not supported"
