import json

import pandas as pd
import pytest

import src.fetcher as fetcher


def _financials_df():
    return pd.DataFrame(
        {
            pd.Timestamp("2025-12-31"): [50.0, 200.0],
            pd.Timestamp("2024-12-31"): [45.0, 190.0],
            pd.Timestamp("2023-12-31"): [40.0, 180.0],
            pd.Timestamp("2022-12-31"): [35.0, 170.0],
        },
        index=["EBITDA", "Total Revenue"],
    )


def _ticker_instance(mocker, info):
    instance = mocker.MagicMock()
    instance.info = info
    instance.financials = _financials_df()
    instance.cashflow = pd.DataFrame()
    return instance


@pytest.fixture(autouse=True)
def no_edgar_network(mocker):
    # EDGAR is wrapped in try/except and falls back to yfinance on any
    # failure — raising here keeps tests network-free.
    mocker.patch("src.fetcher.requests.get", side_effect=Exception("no network in tests"))


def test_cache_hit_skips_api_call(mocker, sample_company):
    fetcher.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_payload = dict(sample_company, ticker="AAA")
    (fetcher.CACHE_DIR / "AAA.json").write_text(json.dumps(cache_payload), encoding="utf-8")

    mock_ticker_cls = mocker.patch("src.fetcher.yf.Ticker")

    results = fetcher.fetch_batch(["AAA"], {"universe": {"max_candidates": 10}})

    mock_ticker_cls.assert_not_called()
    assert results[0]["ticker"] == "AAA"


def test_cache_miss_calls_api(mocker, sample_config):
    info = {
        "longName": "Bravo Industrial",
        "marketCap": 5_000_000_000,
        "grossMargins": 0.4,
        "totalDebt": 100_000_000,
        "totalCash": 20_000_000,
        "enterpriseValue": 600_000_000,
        "sector": "Industrials",
    }
    instance = _ticker_instance(mocker, info)
    mock_ticker_cls = mocker.patch("src.fetcher.yf.Ticker", return_value=instance)

    fetcher.fetch_batch(["BBB"], sample_config)

    mock_ticker_cls.assert_called_once_with("BBB")


def test_retry_called_three_times_on_failure(mocker):
    mocker.patch("time.sleep")
    mock_ticker_cls = mocker.patch("src.fetcher.yf.Ticker", side_effect=Exception("boom"))

    with pytest.raises(Exception):
        fetcher._fetch_single("CCC")

    assert mock_ticker_cls.call_count == 3


def test_failed_ticker_written_to_csv(mocker, sample_config):
    mocker.patch("time.sleep")
    mocker.patch("src.fetcher.yf.Ticker", side_effect=Exception("boom"))

    fetcher.fetch_batch(["DDD"], sample_config)

    assert fetcher.FAILED_TICKERS_CSV.exists()
    content = fetcher.FAILED_TICKERS_CSV.read_text(encoding="utf-8")
    assert "DDD" in content


def test_ev_ebitda_outlier_set_to_none():
    record = {"ev_ebitda": 150, "ebitda_margin": 0.2, "revenue_ttm_usd_mm": 100, "gross_margin": 0.4}
    result = fetcher._validate(record, "EEE")
    assert result["ev_ebitda"] is None


def test_negative_ev_ebitda_set_to_none():
    record = {"ev_ebitda": -3, "ebitda_margin": 0.2, "revenue_ttm_usd_mm": 100, "gross_margin": 0.4}
    result = fetcher._validate(record, "FFF")
    assert result["ev_ebitda"] is None
