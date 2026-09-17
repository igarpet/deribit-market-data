from __future__ import annotations

import requests
import pandas as pd


BASE_URL = "https://www.deribit.com/api/v2"


class DeribitClient:
    def __init__(self):
        self.base_url = BASE_URL
        self.session = requests.Session()

    def _get(self, method: str, params: dict | None = None):
        url = f"{self.base_url}/{method}"

        response = self.session.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        payload = response.json()

        if "error" in payload:
            raise RuntimeError(payload["error"])

        return payload["result"]

    def get_time(self):
        return self._get("public/get_time")

    def get_currencies(self):
        return self._get("public/get_currencies")

    def get_option_instruments(
        self,
        currency: str,
        expired: bool = False
    ) -> pd.DataFrame:

        result = self._get(
            "public/get_instruments",
            params={
                "currency": currency,
                "kind": "option",
                "expired": str(expired).lower()
            }
        )

        return pd.DataFrame(result)


if __name__ == "__main__":

    client = DeribitClient()

    btc = client.get_option_instruments("BTC")

    print(f"Active BTC options: {len(btc)}")

    print(
        btc[
            [
                "instrument_name",
                "strike",
                "option_type",
                "expiration_timestamp"
            ]
        ].head()
    )