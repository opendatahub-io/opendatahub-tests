import logging
import grpc
import socket
import ssl
import sys
from model_serving_tests.endpoint_utility.utils import generation_pb2_grpc

logger = logging.getLogger(__name__)


class TGISGRPCPlugin:
    def __init__(self, host: str, model_name: str, streaming: bool = False, use_tls: bool = False):
        """
        Initialize the TGISGRPCPlugin with necessary parameters.

        Args:
            model_name (str): The model name to use.
            host (str): The gRPC server host.
            streaming (bool): Whether to use streaming.
            use_tls (bool): Whether to use TLS for the connection.
        """
        if not all([model_name, host]):
            raise ValueError("Model name and host are required arguments.")

        self.model_name = model_name
        self.host = host
        self.streaming = streaming
        self.use_tls = use_tls
        self.request_func = self.make_grpc_request_stream if streaming else self.make_grpc_request

    def _get_server_certificate(self, host: str, port: int) -> str:
        if sys.version_info >= (3, 10):
            return ssl.get_server_certificate((host, port))
        context = ssl.SSLContext()
        with socket.create_connection((host, port)) as sock, context.wrap_socket(sock, server_hostname=host) as ssock:
            cert_der = ssock.getpeercert(binary_form=True)
        return ssl.DER_cert_to_PEM_cert(cert_der)

    def _channel_credentials(self) -> grpc.ChannelCredentials:
        if self.use_tls:
            cert = self._get_server_certificate(self.host, 443).encode()
            return grpc.ssl_channel_credentials(root_certificates=cert)
        return None

    def _create_channel(self) -> grpc.Channel:
        credentials = self._channel_credentials()
        return grpc.secure_channel(self.host, credentials) if credentials else grpc.insecure_channel(self.host)

    def make_grpc_request(self, query: dict):
        channel = self._create_channel()
        stub = generation_pb2_grpc.GenerationServiceStub(channel)

        request = generation_pb2_grpc.generation__pb2.BatchedGenerationRequest(
            model_id=self.model_name,
            requests=[
                generation_pb2_grpc.generation__pb2.GenerationRequest(text=query.get("text"))
            ],
            params=generation_pb2_grpc.generation__pb2.Parameters(
                method=generation_pb2_grpc.generation__pb2.GREEDY,
                sampling=generation_pb2_grpc.generation__pb2.SamplingParameters(seed=1037),
            ),
        )

        try:
            response = stub.Generate(request=request)
            response = response.responses[0]
            return {
                "input_tokens": response.input_token_count,
                "stop_reason": response.stop_reason,
                "output_text": response.text,
                "output_tokens": response.generated_token_count,
            }
        except grpc.RpcError as err:
            logger.error("gRPC Error: %s", err.details())
            return None

    def make_grpc_request_stream(self, query: dict):
        channel = self._create_channel()
        stub = generation_pb2_grpc.GenerationServiceStub(channel)

        tokens = []
        request = generation_pb2_grpc.generation__pb2.SingleGenerationRequest(
            model_id=self.model_name,
            request=generation_pb2_grpc.generation__pb2.GenerationRequest(text=query.get("text")),
            params=generation_pb2_grpc.generation__pb2.Parameters(
                method=generation_pb2_grpc.generation__pb2.GREEDY,
                sampling=generation_pb2_grpc.generation__pb2.SamplingParameters(seed=1037),
                response=generation_pb2_grpc.generation__pb2.ResponseOptions(generated_tokens=True)
            ),
        )

        try:
            resp_stream = stub.GenerateStream(request=request)
            for resp in resp_stream:
                if resp.tokens:
                    tokens.append(resp.text)
                    if resp.stop_reason:
                        return {
                            "input_tokens": resp.input_token_count,
                            "stop_reason": resp.stop_reason,
                            "output_text": "".join(tokens),
                            "output_tokens": resp.generated_token_count,
                        }
        except grpc.RpcError as err:
            logger.error("gRPC Error: %s", err.details())
            return None

        return {
            "input_tokens": query.get("input_tokens"),
            "stop_reason": None,
            "output_text": "".join(tokens),
            "output_tokens": len(tokens),
        }

    def get_model_info(self):
        channel = self._create_channel()
        stub = generation_pb2_grpc.GenerationServiceStub(channel)

        request = generation_pb2_grpc.generation__pb2.ModelInfoRequest()

        try:
            response = stub.ModelInfo(request=request)
            return response
        except grpc.RpcError as err:
            logger.error("gRPC Error: %s", err.details())
            return None
