import pytest
import requests

from src import fmp_client


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or []

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._payload


def test_get_profile_retries_with_alternate_key_on_limit_error(monkeypatch, mocker):
    monkeypatch.setenv("FMP_API_KEY", "primary")
    monkeypatch.setenv("FMP_API_KEY_ALTERNATE", "alternate")
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params["apikey"])
        if params["apikey"] == "primary":
            return _Response(429)
        return _Response(200, [{"symbol": "ABC", "mktCap": 123}])

    mocker.patch("src.fmp_client.requests.get", side_effect=fake_get)

    assert fmp_client.get_profile("ABC") == {"symbol": "ABC", "mktCap": 123}
    assert calls == ["primary", "alternate"]


def test_get_profile_does_not_retry_non_limit_errors(monkeypatch, mocker):
    monkeypatch.setenv("FMP_API_KEY", "primary")
    monkeypatch.setenv("FMP_API_KEY_ALTERNATE", "alternate")
    mocker.patch("src.fmp_client.requests.get", return_value=_Response(500))

    with pytest.raises(requests.HTTPError):
        fmp_client.get_profile("ABC")
