import importlib
import json

import pandas as pd
import pytest

import src.fetcher as fetcher


def _statement_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _income_df(revenue_by_period, operating_income=None, gross_profit=None, cogs=None,
               net_income=None, interest_expense=None):
    periods = list(revenue_by_period.keys())
    rows = [{"standard_concept": "Revenue", **revenue_by_period}]
    if operating_income is not None:
        rows.append({"standard_concept": "OperatingIncomeLoss", periods[0]: operating_income})
    if gross_profit is not None:
        rows.append({"standard_concept": "GrossProfit", periods[0]: gross_profit})
    if cogs is not None:
        rows.append({"standard_concept": "CostOfGoodsAndServicesSold", periods[0]: cogs})
    if net_income is not None:
        rows.append({"standard_concept": "NetIncome", periods[0]: net_income})
    if interest_expense is not None:
        rows.append({"standard_concept": "InterestExpense", periods[0]: interest_expense})
    return _statement_df(rows)


def _cashflow_df(depreciation_amortization=None, operating_cash_flow=None):
    rows = []
    if depreciation_amortization is not None:
        rows.append({"standard_concept": "DepreciationExpense", "p0": depreciation_amortization})
    if operating_cash_flow is not None:
        rows.append({"standard_concept": "NetCashFromOperatingActivities", "p0": operating_cash_flow})
    return _statement_df(rows)


def _balance_sheet_df(cash=None, short_term_debt=None, current_portion_ltd=None, long_term_debt=None,
                      minority_interest=None, preferred_equity=None,
                      operating_lease_current=None, operating_lease_noncurrent=None, operating_lease_total=None,
                      total_equity=None):
    rows = []
    if cash is not None:
        rows.append({"standard_concept": "CashAndMarketableSecurities", "p0": cash})
    if total_equity is not None:
        rows.append({"standard_concept": "AllEquityBalance", "p0": total_equity})
    if short_term_debt is not None:
        rows.append({"standard_concept": "ShortTermDebt", "p0": short_term_debt})
    if current_portion_ltd is not None:
        rows.append({"standard_concept": "CurrentPortionOfLongTermDebt", "p0": current_portion_ltd})
    if long_term_debt is not None:
        rows.append({"standard_concept": "LongTermDebt", "p0": long_term_debt})
    if minority_interest is not None:
        rows.append({"standard_concept": "MinorityInterest", "p0": minority_interest})
    if preferred_equity is not None:
        rows.append({"standard_concept": "PreferredStock", "p0": preferred_equity})
    if operating_lease_total is not None:
        rows.append({"standard_concept": "OperatingLeaseDebtEquivalent", "p0": operating_lease_total})
    if operating_lease_current is not None:
        rows.append({"standard_concept": "OperatingLeaseCurrentDebtEquivalent", "p0": operating_lease_current})
    if operating_lease_noncurrent is not None:
        rows.append({"standard_concept": "OperatingLeaseNonCurrentDebtEquivalent", "p0": operating_lease_noncurrent})
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
                      minority_interest=None, preferred_equity=None,
                      operating_lease_current=None, operating_lease_noncurrent=None, operating_lease_total=None,
                      net_income=None, interest_expense=None, operating_cash_flow=None, total_equity=None,
                      business_text=None):
    income_df = _income_df({"p0": revenue}, operating_income=operating_income, gross_profit=gross_profit,
                           net_income=net_income, interest_expense=interest_expense)
    cashflow_df = _cashflow_df(depreciation_amortization, operating_cash_flow=operating_cash_flow)
    balance_sheet_df = _balance_sheet_df(
        cash=cash, short_term_debt=short_term_debt,
        current_portion_ltd=current_portion_ltd, long_term_debt=long_term_debt,
        minority_interest=minority_interest, preferred_equity=preferred_equity,
        operating_lease_current=operating_lease_current, operating_lease_noncurrent=operating_lease_noncurrent,
        operating_lease_total=operating_lease_total, total_equity=total_equity,
    )
    financials = FakeFinancials(income_df, cashflow_df, balance_sheet_df=balance_sheet_df, capex=capex)
    return FakeCompany(name="Test Co.", financials=financials, business_text=business_text)


def _config_with_leases(sample_config, include_leases):
    return {**sample_config, "valuation": {"include_operating_leases_in_ev": include_leases}}


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


def test_ev_bridge_includes_minority_interest_and_preferred(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(
        mocker, revenue=200_000_000.0, operating_income=30_000_000.0, depreciation_amortization=10_000_000.0,
        cash=10_000_000.0, long_term_debt=50_000_000.0,
        minority_interest=20_000_000.0, preferred_equity=15_000_000.0,
    ))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 400_000_000, "sector": "Industrials",
    })

    record = fetcher.fetch_batch(["MI1"], sample_config)[0]

    # EV = market cap 400 + net debt 40 + minority 20 + preferred 15 = 475MM.
    ebitda_mm = 40.0
    expected_ev_mm = 400.0 + 40.0 + 20.0 + 15.0
    assert record["enterprise_value_usd_mm"] == pytest.approx(expected_ev_mm)
    assert record["ev_ebitda"] == pytest.approx(expected_ev_mm / ebitda_mm)
    # The leverage feature must NOT absorb minority/preferred — it stays debt-cash.
    assert record["net_debt_ebitda"] == pytest.approx(40.0 / ebitda_mm)
    assert record["minority_interest_usd_mm"] == pytest.approx(20.0)
    assert record["preferred_equity_usd_mm"] == pytest.approx(15.0)


def test_operating_leases_excluded_from_ev_by_default(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(
        mocker, revenue=200_000_000.0, operating_income=30_000_000.0, depreciation_amortization=10_000_000.0,
        cash=10_000_000.0, long_term_debt=50_000_000.0,
        operating_lease_current=5_000_000.0, operating_lease_noncurrent=25_000_000.0,
    ))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 400_000_000, "sector": "Industrials",
    })

    record = fetcher.fetch_batch(["OL1"], sample_config)[0]

    # Default config has no valuation block -> leases excluded; EV = 400 + 40 = 440.
    assert record["enterprise_value_usd_mm"] == pytest.approx(440.0)
    assert record["valuation_source"] == "sec_xbrl_derived"
    # The liability is still captured on the record for transparency.
    assert record["operating_lease_liability_usd_mm"] == pytest.approx(30.0)


def test_operating_leases_included_in_ev_when_config_opts_in(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(
        mocker, revenue=200_000_000.0, operating_income=30_000_000.0, depreciation_amortization=10_000_000.0,
        cash=10_000_000.0, long_term_debt=50_000_000.0,
        operating_lease_current=5_000_000.0, operating_lease_noncurrent=25_000_000.0,
    ))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 400_000_000, "sector": "Industrials",
    })

    config = _config_with_leases(sample_config, include_leases=True)
    record = fetcher.fetch_batch(["OL2"], config)[0]

    # EV = 400 + net debt 40 + operating leases (5 + 25) = 470MM.
    assert record["enterprise_value_usd_mm"] == pytest.approx(470.0)
    assert record["valuation_source"] == "sec_xbrl_derived_lease_adj"


def test_ev_bridge_with_no_minority_or_preferred_matches_old_bridge(mocker, sample_config):
    # Regression: a plain company (no minority/preferred/leases) must produce
    # exactly the old market_cap + net_debt EV, so existing comps don't move.
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(
        mocker, revenue=200_000_000.0, operating_income=30_000_000.0, depreciation_amortization=10_000_000.0,
        cash=10_000_000.0, long_term_debt=50_000_000.0,
    ))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 400_000_000, "sector": "Industrials",
    })

    record = fetcher.fetch_batch(["PLAIN1"], sample_config)[0]

    assert record["enterprise_value_usd_mm"] == pytest.approx(440.0)
    assert record["minority_interest_usd_mm"] is None
    assert record["preferred_equity_usd_mm"] is None


def test_additional_pe_metrics_computed(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(
        mocker, revenue=200_000_000.0, operating_income=30_000_000.0, depreciation_amortization=10_000_000.0,
        gross_profit=70_000_000.0, capex=8_000_000.0,
        cash=10_000_000.0, long_term_debt=50_000_000.0, total_equity=100_000_000.0,
        net_income=18_000_000.0, interest_expense=6_000_000.0, operating_cash_flow=35_000_000.0,
    ))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 400_000_000, "sector": "Industrials",
    })

    record = fetcher.fetch_batch(["PE1"], sample_config)[0]

    # EBITDA = 40MM, EBIT = 30MM, EV = market cap 400 + net debt 40 = 440MM.
    # FCF = CFO 35 - capex 8 = 27MM.
    assert record["free_cash_flow_usd_mm"] == pytest.approx(27.0)
    assert record["fcf_conversion"] == pytest.approx(27.0 / 40.0)           # 0.675
    assert record["interest_coverage"] == pytest.approx(30.0 / 6.0)         # EBIT / interest = 5.0
    assert record["debt_to_equity"] == pytest.approx(50.0 / 100.0)          # 0.5
    assert record["ev_ebit"] == pytest.approx(440.0 / 30.0)                 # ~14.67x
    assert record["ev_gross_profit"] == pytest.approx(440.0 / 70.0)         # ~6.29x
    assert record["pe_ratio"] == pytest.approx(400.0 / 18.0)               # ~22.2x
    assert record["fcf_yield"] == pytest.approx(27.0 / 440.0)               # ~6.1%


def test_pe_ratio_none_when_net_income_negative(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(
        mocker, revenue=200_000_000.0, operating_income=30_000_000.0, depreciation_amortization=10_000_000.0,
        cash=10_000_000.0, long_term_debt=50_000_000.0,
        net_income=-5_000_000.0,
    ))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 400_000_000, "sector": "Industrials",
    })

    record = fetcher.fetch_batch(["PE2"], sample_config)[0]

    # A negative-earnings comp yields a meaningless P/E -> left None.
    assert record["pe_ratio"] is None
    # but EV/EBIT still computes off the positive EBIT.
    assert record["ev_ebit"] == pytest.approx(440.0 / 30.0)


def test_new_metrics_none_when_source_lines_absent(mocker, sample_config):
    # No interest expense, no operating cash flow, no equity tagged -> the
    # dependent ratios stay None rather than dividing by a missing value.
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(
        mocker, revenue=200_000_000.0, operating_income=30_000_000.0, depreciation_amortization=10_000_000.0,
        cash=10_000_000.0, long_term_debt=50_000_000.0,
    ))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 400_000_000, "sector": "Industrials",
    })

    record = fetcher.fetch_batch(["NA1"], sample_config)[0]

    assert record["fcf_conversion"] is None
    assert record["interest_coverage"] is None
    assert record["debt_to_equity"] is None
    assert record["fcf_yield"] is None


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


def test_edgar_identity_uses_environment_variable(monkeypatch, mocker):
    mock_set_identity = mocker.patch("src.fetcher.edgar.set_identity")
    monkeypatch.setenv("SEC_IDENTITY", "PE Comps test@example.com")

    importlib.reload(fetcher)

    mock_set_identity.assert_called_with("PE Comps test@example.com")


def test_fetch_batch_preserves_structured_universe_metadata(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(mocker))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={})

    candidate = {
        "ticker": "META1",
        "source_bucket": "primary",
        "matched_sic_codes": ["3714"],
        "sic_2_digit": "37",
        "sic_3_digit": "371",
        "industry_cluster": "auto_parts",
    }

    record = fetcher.fetch_batch([candidate], sample_config)[0]

    assert record["ticker"] == "META1"
    assert record["universe_metadata"]["source_bucket"] == "primary"
    assert record["source_bucket"] == "primary"
    assert record["industry_cluster"] == "auto_parts"


def test_valuation_source_is_sec_xbrl_derived_when_profile_has_market_cap(mocker, sample_config):
    mocker.patch("src.fetcher.edgar.Company", return_value=_healthy_company(
        mocker, revenue=200_000_000.0, operating_income=30_000_000.0, depreciation_amortization=10_000_000.0,
        cash=10_000_000.0, long_term_debt=50_000_000.0,
    ))
    mocker.patch("src.fetcher.fmp_client.get_profile", return_value={
        "marketCap": 400_000_000, "sector": "Industrials",
    })

    record = fetcher.fetch_batch(["FMP1"], sample_config)[0]

    assert record["valuation_source"] == "sec_xbrl_derived"
    assert record["ev_ebitda"] is not None
