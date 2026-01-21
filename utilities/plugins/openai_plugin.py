import json
import requests
import urllib3
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any, Optional
from urllib3.exceptions import InsecureRequestWarning
from utilities.plugins.constant import OpenAIEnpoints, RestHeader
from simple_logger.logger import get_logger

urllib3.disable_warnings(category=InsecureRequestWarning)
requests.packages
LOGGER = get_logger(name=__name__)

MAX_RETRIES = 5


class OpenAIClient:
    """
    A client for interacting with the OpenAI API.

    Attributes:
        host (str): The base URL for the API.
        streaming (bool): Flag to indicate if streaming requests should be used.
        model_name (str, optional): The name of the model to use.
        request_func (Callable): The function to use for making requests.
    """

    def __init__(self, host: Any, streaming: bool = False, model_name: Any = None) -> None:
        """
        Initializes the OpenAIClient.

        Args:
            host (str): The base URL for the API.
            streaming (bool, optional): If True, use streaming requests. Defaults to False.
            model_name (str, optional): The name of the model to use. Defaults to None.
        """
        self.host = host
        self.streaming = streaming
        self.model_name = model_name
        self.request_func = self.streaming_request_http if streaming else self.request_http

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(min=1, max=6))
    def request_http(self, endpoint: str, query: dict[str, str], extra_param: Optional[dict[str, Any]] = None) -> Any:
        """
        Sends a HTTP POST request to the specified endpoint and processes the response.

        Args:
            endpoint (str): The API endpoint to send the request to.
            query (dict): The query parameters to include in the request.
            extra_param (dict, optional): Additional parameters to include in the request.

        Returns:
            Any: The parsed response from the API.

        Raises:
            requests.exceptions.RequestException: If there is a request error.
            json.JSONDecodeError: If there is a JSON decoding error.
        """
        headers = RestHeader.HEADERS
        data = self._construct_request_data(endpoint, query, extra_param)
        try:
            url = f"{self.host}{endpoint}"
            response = requests.post(url, headers=headers, json=data, verify=False)
            LOGGER.info(response)
            response.raise_for_status()
            message = response.json()
            return self._parse_response(endpoint, message)
        except (requests.exceptions.RequestException, json.JSONDecodeError) as err:
            LOGGER.error(f"Test failed due to an unexpected exception: {err}")
            raise

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(min=1, max=6))
    def streaming_request_http(
        self, endpoint: str, query: dict[str, Any], extra_param: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Sends a streaming HTTP POST request to the specified endpoint and processes the streamed response.

        Args:
            endpoint (str): The API endpoint to send the request to.
            query (dict): The query parameters to include in the request.
            extra_param (dict, optional): Additional parameters to include in the request.

        Returns:
            str: The concatenated streaming response.

        Raises:
            requests.exceptions.RequestException: If there is a request error.
            json.JSONDecodeError: If there is a JSON decoding error.
        """
        headers = RestHeader.HEADERS
        data = self._construct_request_data(endpoint, query, extra_param, streaming=True)
        tokens = []
        try:
            url = f"{self.host}{endpoint}"
            response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
            LOGGER.info(response)
            response.raise_for_status()
            for line in response.iter_lines():
                _, found, data = line.partition(b"data: ")
                if found and data != b"[DONE]":
                    message = json.loads(data)
                    token = self._parse_streaming_response(endpoint, message)
                    tokens.append(token)
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            LOGGER.error("Streaming request error")
            raise
        return "".join(tokens)

    @staticmethod
    def get_request_http(host: str, endpoint: str) -> Any:
        """
        Sends a HTTP GET request to the specified endpoint and returns the response data.

        Args:
            host (str): The base URL for the API.
            endpoint (str): The API endpoint to send the request to.

        Returns:
            dict: The data from the response.

        Raises:
            requests.exceptions.RequestException: If there is a request error.
            json.JSONDecodeError: If there is a JSON decoding error.
        """
        headers = RestHeader.HEADERS
        url = f"{host}{endpoint}"
        try:
            response = requests.get(url, headers=headers, verify=False)
            LOGGER.info(response)
            response.raise_for_status()
            message = response.json()
            data = message.get("data", [])
            keys_to_remove = ["created", "id"]
            if data:
                data = OpenAIClient._remove_keys(data, keys_to_remove)
            return data
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            LOGGER.exception("Request error")

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(min=1, max=6))
    def request_audio(
        self, endpoint: str, audio_file_path: str, model_name: str, filename: str = "harvard.wav", language: str = "en"
    ) -> Any:
        """
        Sends a HTTP POST request to the specified endpoint with an audio file.

        Args:
            endpoint (str): The API endpoint to send the request to.
            audio_file_path (str): The path to the audio file to be sent.
            model_name (str): The name of the model to use for processing the audio.
            filename (str, optional): The name of the audio file. Defaults to "harvard.wav".

        Returns:
            Any: The parsed response from the API.
        Raises:
            requests.exceptions.RequestException: If there is a request error.
            json.JSONDecodeError: If there is a JSON decoding error.
        """
        headers = {k: v for k, v in RestHeader.HEADERS.items() if k != "Content-Type"}
        data = {
            "model": model_name,
            "response_format": "json",
            "language": language,
        }
        try:
            url = f"{self.host}{endpoint}"
            with open(audio_file_path, "rb") as audio_file:
                files = {"file": (filename, audio_file, "audio/wav")}
                response = requests.post(url, headers=headers, files=files, data=data, verify=False)
            LOGGER.info(response)
            response.raise_for_status()
            message = response.json()
            return self._parse_response(endpoint, message)
        except (requests.exceptions.RequestException, json.JSONDecodeError) as err:
            LOGGER.error(f"Test failed due to an unexpected exception: {err}")
            raise

    def _construct_request_data(
        self,
        endpoint: str,
        query: dict[str, Any],
        extra_param: Optional[dict[str, Any]] = None,
        streaming: bool = False,
    ) -> dict[str, Any]:
        """
        Constructs the request data based on the endpoint and query parameters.

        Args:
            endpoint (str): The API endpoint to send the request to.
            query (dict): The query parameters to include in the request.
            extra_param (dict, optional): Additional parameters to include in the request.
            streaming (bool, optional): If True, include streaming parameters. Defaults to False.

        Returns:
            dict: The constructed request data.
        """
        data = {}
        if OpenAIEnpoints.CHAT_COMPLETIONS in endpoint:
            data = {"messages": query, "temperature": 0, "seed": 1037, "stream": streaming}
        elif OpenAIEnpoints.EMBEDDINGS in endpoint:
            data = {
                "input": query["text"],
            }
        else:
            data = {"prompt": query["text"], "temperature": 0, "top_p": 0.9, "seed": 1037, "stream": streaming}

        if self.model_name:
            data["model"] = self.model_name

        if extra_param:
            data.update(extra_param)  # Add the extra parameters if provided

        return data

    def _parse_response(self, endpoint: str, message: dict[str, Any]) -> Any:
        """
        Parses the response message based on the endpoint.

        Args:
            endpoint (str): The API endpoint that was queried.
            message (dict): The JSON response message.

        Returns:
            Any: The parsed response data.
        """
        if OpenAIEnpoints.CHAT_COMPLETIONS in endpoint:
            LOGGER.info(message["choices"][0])
            return message["choices"][0]
        elif OpenAIEnpoints.EMBEDDINGS in endpoint:
            LOGGER.info(message["data"][0])
            return message["data"][0]
        elif OpenAIEnpoints.AUDIO_TRANSCRIPTION in endpoint:
            LOGGER.info(message["text"])
            return message["text"]
        else:
            LOGGER.info(message["choices"][0])
            return message["choices"][0]

    def _parse_streaming_response(self, endpoint: str, message: dict[str, Any]) -> Any:
        """
        Parses a streaming response message based on the endpoint.

        Args:
            endpoint (str): The API endpoint that was queried.
            message (dict): The JSON response message.

        Returns:
            str: The parsed streaming response data.
        """
        if OpenAIEnpoints.CHAT_COMPLETIONS in endpoint and not message["choices"][0]["delta"].get("content"):
            message["choices"][0]["delta"]["content"] = ""
        if message.get("error"):
            return message.get("error")
        return (
            message["choices"][0].get("delta", {}).get("content", "")
            if OpenAIEnpoints.CHAT_COMPLETIONS in endpoint
            else message["choices"][0].get("text", "")
        )

    @staticmethod
    def _remove_keys(data: list[dict[str, Any]], keys_to_remove: list[str]) -> list[dict[str, Any]]:
        """Remove specific keys from a list of dictionaries."""
        for item in data:
            item.pop("created", None)  # only delete created timestamp
            if "permission" in item:
                for permission in item["permission"]:
                    for key in keys_to_remove:
                        permission.pop(key, None)
        return data
