import requests
import http
from typing import Dict

from timeout_sampler import retry


def get_auth_headers(token: str) -> Dict[str, str]:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


@retry(exceptions_dict={TimeoutError: []}, wait_timeout=10, sleep=2)
def check_guardrails_health_endpoint(
    host: str,
    token: str,
    ca_bundle_file: str,
) -> bool:
    response = requests.get(url=f"https://{host}/health", headers=get_auth_headers(token=token), verify=ca_bundle_file)
    return response.status_code == http.HTTPStatus.OK
