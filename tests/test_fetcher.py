import json

import pandas as pd
import pytest

import src.fetcher as fetcher


def _statement_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _income_df(revenue_by_period, operating_income=None, gross_profit=None, cogs=None):
    periods = list(revenue_by_period.keys())
    rows = [{"standard_concept": "Revenue", **revenue_by_period}]
    if operating_income is not None:
        rows.append({"standard_concept": "OperatingIncomeLoss", periods[0]: operating_income})
    if gross_profit is not None:
        rows.append({"standard_concept": "GrossProfit", periods[0]: gross_profit})
    if cogs is not None:
        rows.append({"standard_concept": "CostOfGoodsAndServicesSold", periods[0]: cogs})
    return _statement_df(rows)


def _cashflow_df(depreciation_amortization=None):
    if depreciation_amortization is None:
        return _statement_df([])
    return _statement_df([{"standard_concept": "DepreciationExpense", "p0": depreciation_amortization}])


def _balance_sheet_df(cash=None, short_term_debt=None, current_portion_ltd=None, long_term_debt=None):
    rows = []
    if cash is not None:
        rows.append({"standard_concept": "CashAndMarketableSecurities", "p0": cash})
    if short_term_debt is not None:
        rows.append({"standard_concept": "ShortTermDebt", "p0": short_term_debt})
    if current_portion_ltd is not None:
        rows.append({"standard_concept": "CurrentPortionOfLongTermDebt", "p0": current_portion_ltd})
    if long_term_debt is not None:
        rows.append({"standard_concept": "LongTermDebt", "p0": long_term_debt})
    return _statement_df(rows)


class _StatementWrapper:
    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df


class FakeFinancials:
    def __init__(self, income_df, cashflow_df, balance_sheet_df=None, capex=None):
        self._income_df = income_df
        self._cashflow_df = cashflow_df
        self._balance_sheet_df = balance_sheet_df if balance_sheet_df is not None else _statement_df([])
        self._capex = capex

    def income_statement(self):
        return _StatementWrapper(self._income_df)

    def cashflow_statement(self):
        return _StatementWrapper(self._cashflow_df)

    def balance_sheet(self):
        return _StatementWrapper(self._balance_sheet_df)

    def get_capital_expenditures(self):
        return self._capex


class _FakeTenK:
    def __init__(self, business_text):
        self.business = business_text


class FakeFiling:
    def __init__(self, business_text):
        self._business_text = business_text

    def obj(self):
        return _FakeTenK(self._business_text)


class FakeFilings:
    def __init__(self, filing):
        self._filing = filing

    def latest(self):
        return self._filing


class FakeCompany:
    def __init__(self, name="Test Co.", financials=None, business_text=None):
        self.name = name
        self._financials = financials
        self._filing = FakeFiling(business_text) if business_text is not None else None

    def get_financials(self):
        return self._financials

    def get_filings(self, form=None):
        return FakeFilings(self._filing)


def _healthy_company(mocker, revenue=200_000_000.0, operating_income=30_000_000.0,
                      depreciation_amortization=10_000_000.0, gross_profit=70_000_000.0, capex=8_000_000.0,
                      cash=None, short_term_debt=None, current_portion_ltd=None, long_term_debt=None,
                      business_text=None):
    income_df = _income_df({"p0": revenue}, operating_income=operating_income, gross_profit=gross_profit)
    cashflow_df = _cashflow_df(depreciation_amortization)
    balance_sheet_df = _balance_sheet_df(
        cash=cash, short_term_debt=short_term_debt,
        current_portion_ltd=current_portion_ltd, long_term_debt=long_term_debt,
    )
    financials = FakeFinancials(income_df, cashflow_df, balance_sheet_df=balance_sheet_df, capex=capex)
    return FakeCompany(name="Test Co.", financials=financials, business_text=business_text)


def test_cache_hit_skips_api_call(mocker, sample_company):
    fetcher.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_payload = dict(sample_company, ticker="AAA")
    (fetcher.CACHE_DIR / "AAA.json").write_text(json.dumps(cache_payload), encoding="utf-8")

    mock_company_cls = mocker.patch("src.fetcher.edgar.Company")
    mock_fmp = mocker.patch("src.fetcher.fmp_client.get_profile")

    results = fetcher.fetch_batch(["AAA"], {"universe": {"max_candidates": 10}})

    mock_company_cls.assert_not_called()
    mock_fmp.assert_not_called()
    assert results[0]["ticker"] == "AAA"


def test_cache_miss_calls_api(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(mocker))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 5_000_000_000, "sector": "Industrials",
    })

    fetcher.fetch_batch(["BBB"], sample_config)

    # Company() is called twice per ticker: once for financials, once for
    # the 10-K filing used to pull the business description.
    fetcher.edgar.Company.assert_any_call("BBB")


def test_retry_called_three_times_on_failure(mocker):
    mocker.patch("time.sleep")
    mock_company_cls = mocker.patch("src.fetcher.edgar.Company", side_effect=Exception("EDGAR down"))

    with pytest.raises(Exception):
        fetcher._fetch_single("CCC")

    assert mock_company_cls.call_count == 3


def test_failed_ticker_written_to_csv(mocker, sample_config):
    mocker.patch("time.sleep")
    mocker.patch("src.fetcher.edgar.Company", side_effect=Exception("EDGAR down"))

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


def test_fmp_failure_keeps_edgar_data(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(
        mocker, revenue=200_000_000.0, operating_income=30_000_000.0, depreciation_amortization=10_000_000.0,
    ))
    mocker.patch("src.fetcher.fmp_client.get_profile", side_effect=Exception("FMP down"))

    results = fetcher.fetch_batch(["GGG"], sample_config)
    record = results[0]

    # EBITDA = operating_income + D&A = 40,000,000; margin = 40MM / 200MM = 0.2
    assert record["ebitda_margin"] == pytest.approx(0.2)
    assert record["revenue_ttm_usd_mm"] == pytest.approx(200.0)
    assert record["market_cap_usd_mm"] is None
    assert record["ev_ebitda"] is None


def test_net_debt_ebitda_computed_purely_from_edgar(mocker, sample_config):
    # Net debt = (short-term 5MM + current-portion-LTD 10MM + LTD 35MM) - cash 10MM = 40MM.
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(
        mocker, revenue=200_000_000.0, operating_income=30_000_000.0, depreciation_amortization=10_000_000.0,
        cash=10_000_000.0, short_term_debt=5_000_000.0, current_portion_ltd=10_000_000.0, long_term_debt=35_000_000.0,
    ))
    mocker.patch("src.fetcher.fmp_client.get_profile", side_effect=Exception("FMP down"))

    results = fetcher.fetch_batch(["HHH2"], sample_config)
    record = results[0]

    # net_debt_ebitda must be populated even though FMP failed entirely —
    # it never depends on FMP at all.
    assert record["net_debt_ebitda"] == pytest.approx(40.0 / 40.0)
    assert record["market_cap_usd_mm"] is None


def test_ev_ebitda_from_market_cap_and_net_debt(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(
        mocker, revenue=200_000_000.0, operating_income=30_000_000.0, depreciation_amortization=10_000_000.0,
        cash=10_000_000.0, long_term_debt=50_000_000.0,
    ))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 400_000_000, "sector": "Industrials",
    })

    results = fetcher.fetch_batch(["HHH"], sample_config)
    record = results[0]

    # EBITDA = 40MM. Net debt = 50MM - 10MM = 40MM. EV = market cap (400MM) + net debt (40MM) = 440MM.
    ebitda_mm = 40.0
    expected_ev_mm = 400.0 + 40.0
    assert record["ev_ebitda"] == pytest.approx(expected_ev_mm / ebitda_mm)
    assert record["net_debt_ebitda"] == pytest.approx(40.0 / ebitda_mm)


def test_fmp_lookup_sleeps_between_tickers_not_before_first(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(mocker))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 1_000_000_000, "sector": "Industrials",
    })
    mock_sleep = mocker.patch("src.fetcher.time.sleep")

    fetcher.fetch_batch(["III", "JJJ"], sample_config)

    mock_sleep.assert_called_once_with(fetcher.FMP_REQUEST_DELAY_SECONDS)


def test_business_description_from_edgar(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(mocker, business_text="A" * 2000))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 1_000_000_000, "sector": "Industrials",
    })

    results = fetcher.fetch_batch(["KKK"], sample_config)
    record = results[0]

    assert record["description_source"] == "edgar"
    assert len(record["business_description"]) == fetcher.BUSINESS_DESCRIPTION_MAX_CHARS


def test_business_description_falls_back_to_fmp(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(mocker, business_text=None))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 1_000_000_000, "sector": "Industrials",
        "description": "Fallback description from FMP.",
    })

    results = fetcher.fetch_batch(["LLL"], sample_config)
    record = results[0]

    assert record["business_description"] == "Fallback description from FMP."
    assert record["description_source"] == "fmp"


def test_business_description_none_when_both_sources_fail(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(mocker, business_text=None))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 1_000_000_000, "sector": "Industrials",
    })

    results = fetcher.fetch_batch(["MMM2"], sample_config)
    record = results[0]

    assert record["business_description"] is None
    assert record["description_source"] is None
