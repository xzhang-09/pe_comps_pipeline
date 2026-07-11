import importlib

import src.sic_universe_builder as sic_universe_builder


class _FakeResponse:
    def __init__(self, text="", json_data=None, status_code=200):
        self.text = text
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400 and self.status_code != 404:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def _atom_with_ciks(ciks):
    entries = "".join(f"<cik>{c}</cik>" for c in ciks)
    return f"<feed>{entries}</feed>"


def test_fetch_ciks_for_sic_paginates(mocker):
    page1 = _atom_with_ciks(range(1, 101))  # full page -> fetch next page
    page2 = _atom_with_ciks(range(101, 121))  # partial page -> stop

    mock_get = mocker.patch(
        "src.sic_universe_builder.requests.get",
        side_effect=[_FakeResponse(text=page1), _FakeResponse(text=page2)],
    )

    ciks = sic_universe_builder._fetch_ciks_for_sic("9999")

    assert ciks == list(range(1, 121))
    assert mock_get.call_count == 2


def test_fetch_ciks_for_sic_single_partial_page(mocker):
    mocker.patch(
        "src.sic_universe_builder.requests.get",
        return_value=_FakeResponse(text=_atom_with_ciks([1, 2, 3])),
    )

    ciks = sic_universe_builder._fetch_ciks_for_sic("9999")

    assert ciks == [1, 2, 3]


def test_fetch_company_tickers_returns_list(mocker):
    mocker.patch(
        "src.sic_universe_builder.requests.get",
        return_value=_FakeResponse(json_data={"tickers": ["ABC", "ABC.W"]}),
    )

    tickers = sic_universe_builder._fetch_company_tickers(123)

    assert tickers == ["ABC", "ABC.W"]


def test_fetch_company_tickers_404_returns_empty(mocker):
    mocker.patch(
        "src.sic_universe_builder.requests.get",
        return_value=_FakeResponse(status_code=404),
    )

    tickers = sic_universe_builder._fetch_company_tickers(999)

    assert tickers == []


def test_discover_excludes_companies_without_ticker(mocker, tmp_path):
    mocker.patch.object(sic_universe_builder, "CACHE_DIR", tmp_path)
    mocker.patch("src.sic_universe_builder._fetch_ciks_for_sic", return_value=[1, 2])
    mocker.patch(
        "src.sic_universe_builder._fetch_company_tickers",
        side_effect=lambda cik: ["AAA"] if cik == 1 else [],
    )

    tickers = sic_universe_builder.discover_tickers_by_sic(["9999"])

    assert tickers == ["AAA"]


def test_discover_dedupes_across_sic_codes(mocker, tmp_path):
    mocker.patch.object(sic_universe_builder, "CACHE_DIR", tmp_path)
    mocker.patch("src.sic_universe_builder._fetch_ciks_for_sic", return_value=[1])
    mocker.patch("src.sic_universe_builder._fetch_company_tickers", return_value=["AAA"])

    tickers = sic_universe_builder.discover_tickers_by_sic(["1111", "2222"])

    assert tickers == ["AAA"]


def test_discover_second_call_uses_cache(mocker, tmp_path):
    mocker.patch.object(sic_universe_builder, "CACHE_DIR", tmp_path)
    mock_fetch_ciks = mocker.patch("src.sic_universe_builder._fetch_ciks_for_sic", return_value=[1])
    mocker.patch("src.sic_universe_builder._fetch_company_tickers", return_value=["AAA"])

    sic_universe_builder.discover_tickers_by_sic(["3333"])
    sic_universe_builder.discover_tickers_by_sic(["3333"])

    mock_fetch_ciks.assert_called_once()


def test_discover_cik_lookup_failure_is_skipped(mocker, tmp_path):
    mocker.patch.object(sic_universe_builder, "CACHE_DIR", tmp_path)
    mocker.patch("src.sic_universe_builder._fetch_ciks_for_sic", return_value=[1, 2])
    mocker.patch(
        "src.sic_universe_builder._fetch_company_tickers",
        side_effect=[Exception("network error"), ["BBB"]],
    )

    tickers = sic_universe_builder.discover_tickers_by_sic(["4444"])

    assert tickers == ["BBB"]


def test_fetch_company_profile_returns_full_payload(mocker):
    mocker.patch(
        "src.sic_universe_builder.requests.get",
        return_value=_FakeResponse(json_data={"tickers": ["XYZ"], "sic": "3559", "name": "Test Co"}),
    )

    profile = sic_universe_builder.fetch_company_profile(123)

    assert profile == {"tickers": ["XYZ"], "sic": "3559", "name": "Test Co"}


def test_fetch_company_profile_404_returns_none(mocker):
    mocker.patch(
        "src.sic_universe_builder.requests.get",
        return_value=_FakeResponse(status_code=404),
    )

    assert sic_universe_builder.fetch_company_profile(999) is None


def test_fetch_sic_for_ticker_uses_company_tickers_index_then_profile(mocker):
    mocker.patch(
        "src.sic_universe_builder.requests.get",
        side_effect=[
            _FakeResponse(json_data={"0": {"ticker": "AAA", "cik_str": 123}}),
            _FakeResponse(json_data={"tickers": ["AAA"], "sic": "3569", "name": "AAA Inc."}),
        ],
    )

    assert sic_universe_builder.fetch_sic_for_ticker("aaa") == "3569"


def test_fetch_sic_for_ticker_returns_none_when_ticker_missing(mocker):
    mocker.patch(
        "src.sic_universe_builder.requests.get",
        return_value=_FakeResponse(json_data={"0": {"ticker": "BBB", "cik_str": 456}}),
    )

    assert sic_universe_builder.fetch_sic_for_ticker("AAA") is None


def test_sec_headers_use_environment_identity(monkeypatch):
    monkeypatch.setenv("SEC_IDENTITY", "PE Comps test@example.com")

    module = importlib.reload(sic_universe_builder)

    assert module.HEADERS["User-Agent"] == "PE Comps test@example.com"


def test_sec_headers_default_identity_logs_warning(monkeypatch, caplog):
    monkeypatch.delenv("SEC_IDENTITY", raising=False)

    module = importlib.reload(sic_universe_builder)

    assert module.HEADERS["User-Agent"] == module.DEFAULT_SEC_IDENTITY
    assert "SEC_IDENTITY is not set" in caplog.text
