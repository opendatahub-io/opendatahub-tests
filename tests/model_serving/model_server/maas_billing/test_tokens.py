import json

from tests.model_serving.model_server.maas_billing.utils import b64url_decode


class TestMintedToken:
    def test_minted_token_generated(self, minted_token: str) -> None:
        """Smoke: a MaaS token can be minted."""
        assert isinstance(minted_token, str) and len(minted_token) > 10, "no usable token minted"

    def test_minted_token_is_jwt(self, minted_token: str) -> None:
        """Minted token looks like a JWT and has a JSON header."""
        parts = minted_token.split(".")
        assert len(parts) == 3, "not a JWT (expected header.payload.signature)"

        header_json = b64url_decode(parts[0]).decode("utf-8")
        header = json.loads(header_json)
        assert isinstance(header, dict), "JWT header not a JSON object"
