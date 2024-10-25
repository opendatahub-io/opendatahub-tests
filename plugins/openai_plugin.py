import json
import logging
import time
from typing import Any, Optional
import pytest
import requests
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    A client for interacting with the OpenAI API.

    Attributes:
        host (str): The base URL for the API.
        streaming (bool): Flag to indicate if streaming requests should be used.
        model_name (str, optional): The name of the model to use.
        request_func (Callable): The function to use for making requests.
    """

    def __init__(self,
                 host: Any,
                 streaming: bool = False,
                 model_name: Any = None) -> None:
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

    def request_http(self, endpoint: str, query: dict, extra_param: Optional[dict] = None) -> Any:
        """
        Sends a HTTP POST request to the specified endpoint and processes the response.

        Args:
            endpoint (str): The API endpoint to send the request to.
            query (dict): The query parameters to include in the request.
            extra_param (dict, optional): Additional parameters to include in the request.

        Returns:
            Any: The parsed response from the API.

        Raises:
            pytest.Fail: If the request fails due to an exception.
        """
        headers = {"Content-Type": "application/json"}
        data = self._construct_request_data(endpoint, query, extra_param)
        logger.info(data)

        try:
            url = f"{self.host}{endpoint}"
            time.sleep(10)
            logger.info(url)
            response = requests.post(url, headers=headers, json=data, verify=False)
            logger.info(response)
            logger.info(response.status_code)
            response.raise_for_status()
            message = response.json()
            return self._parse_response(endpoint, message)
        except (requests.exceptions.RequestException, json.JSONDecodeError) as err:
            logger.exception("Request error")
            pytest.fail(f"Test failed due to an unexpected exception: {err}")
            return str(err)

    def streaming_request_http(self, endpoint: str, query: dict, extra_param: Optional[dict] = None) -> str:
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
        headers = {"Content-Type": "application/json"}
        data = self._construct_request_data(endpoint, query, extra_param, streaming=True)
        time.sleep(10)
        tokens = []
        try:
            url = f"{self.host}{endpoint}"
            response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
            logger.info(response)
            response.raise_for_status()
            for line in response.iter_lines():
                _, found, data = line.partition(b"data: ")
                if found and data != b"[DONE]":
                    message = json.loads(data)
                    token = self._parse_streaming_response(endpoint, message)
                    tokens.append(token)
        except (requests.exceptions.RequestException, json.JSONDecodeError) as err:
            logger.exception("Streaming request error")
            return str(err)

        return "".join(tokens)

    @staticmethod
    def get_request_http(host: str, endpoint: str) -> dict:
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
        headers = {"Content-Type": "application/json"}
        url = f"{host}{endpoint}"
        try:
            response = requests.get(url, headers=headers, verify=False)
            logger.info(response)
            response.raise_for_status()
            message = response.json()
            return message.get("data", {})
        except (requests.exceptions.RequestException, json.JSONDecodeError) as err:
            logger.exception("Request error")
            return str(err)

    def _construct_request_data(self, endpoint: str, query: dict, extra_param: Optional[dict] = None,
                                streaming: bool = False) -> dict:
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
        if "/v1/chat/completions" in endpoint:
            data = {
                "messages": query,
                "temperature": 0.1,
                "seed": 1037,
                "stream": streaming
            }
        elif "/v1/embeddings" in endpoint:
            data = {
                "input": query["text"],
                "encoding_format": 0.1,
            }
        else:
            data = {
                "prompt": query["text"],
                "temperature": 1.0,
                "top_p": 0.9,
                "seed": 1037,
                "stream": streaming
            }

        if self.model_name:
            data["model"] = self.model_name

        if extra_param:
            data.update(extra_param)  # Add the extra parameters if provided

        return data

    def _parse_response(self, endpoint: str, message: dict) -> Any:
        """
        Parses the response message based on the endpoint.

        Args:
            endpoint (str): The API endpoint that was queried.
            message (dict): The JSON response message.

        Returns:
            Any: The parsed response data.
        """
        if "/v1/chat/completions" in endpoint:
            logger.info(message["choices"][0])
            return message["choices"][0]
        elif "/v1/embeddings" in endpoint:
            logger.info(message["choices"][0])
            return message["choices"][0]
        else:
            logger.info(message["choices"][0])
            return message["choices"][0]

    def _parse_streaming_response(self, endpoint: str, message: dict) -> str:
        """
        Parses a streaming response message based on the endpoint.

        Args:
            endpoint (str): The API endpoint that was queried.
            message (dict): The JSON response message.

        Returns:
            str: The parsed streaming response data.
        """
        if "/v1/chat/completions" in endpoint and not message["choices"][0]['delta'].get('content'):
            message["choices"][0]['delta']['content'] = ""
        if message.get("error"):
            return message.get("error")
        return message["choices"][0].get('delta', {}).get('content', '') if "/v1/chat/completions" in endpoint else \
            message["choices"][0].get("text", "")
