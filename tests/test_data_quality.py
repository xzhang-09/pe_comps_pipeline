import src.data_quality as data_quality


def _company(ticker, **overrides):
    base = {
        "ticker": ticker,
        "company_name": f"{ticker} Inc.",
        "market_cap_usd_mm": 500.0,
        "revenue_ttm_usd_mm": 200.0,
        "ebitda_margin": 0.20,
        "gross_margin": 0.35,
        "revenue_cagr_3yr": 0.05,
        "net_debt_ebitda": 1.5,
        "capex_revenue": 0.04,
        "ev_ebitda": 12.0,
        "ev_revenue": 2.4,
        "gics_sector": "20",
        "business_description": "A" * 300,
        "description_source": "edgar",
        "fetch_timestamp": "2026-06-16T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def test_report_contains_all_fields():
    companies = [_company(f"T{i}") for i in range(5)]

    report = data_quality.generate_report(companies)

    assert "ebitda_margin" in report
    assert "ev_ebitda" in report


def test_missing_rate_calculation():
    companies = [
        _company("A", ev_ebitda=None),
        _company("B", ev_ebitda=None),
        _company("C", ev_ebitda=10.0),
        _company("D", ev_ebitda=14.0),
    ]

    report = data_quality.generate_report(companies)

    assert "50.0%" in report


def test_report_file_written():
    companies = [_company(f"T{i}") for i in range(3)]

    data_quality.generate_report(companies)

    assert data_quality.OUTPUT_PATH.exists()


def test_short_description_count():
    companies = [
        _company("A", business_description="short"),
        _company("B", business_description="also short"),
        _company("C", business_description="C" * 300),
        _company("D", business_description="D" * 300),
    ]

    report = data_quality.generate_report(companies)

    assert "Short (<200 chars): 2 companies" in report
