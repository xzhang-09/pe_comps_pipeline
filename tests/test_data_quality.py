import scripts.data_quality as data_quality


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


def test_report_includes_candidate_eligibility_counts():
    companies = [
        _company("A", ev_ebitda=12.0, business_description="A" * 300),
        _company("B", ev_ebitda=None, business_description="B" * 300),
        _company("C", ev_ebitda=None, business_description=None),
    ]

    report = data_quality.generate_report(companies)

    assert "Similarity candidates: 2" in report
    assert "Model training rows: 1" in report


def test_report_breaks_down_valuation_source_and_source_bucket():
    companies = [
        _company("A", valuation_source="sec_xbrl_derived", source_bucket="primary"),
        _company("B", valuation_source="sec_xbrl_derived", source_bucket="adjacent"),
        _company("C", valuation_source=None, source_bucket="primary"),
    ]

    report = data_quality.generate_report(companies)

    assert "sec_xbrl_derived" in report
    assert "primary" in report
    assert "adjacent" in report


def test_report_separates_comps_and_training_only_candidates():
    companies = [
        _company("A", source_bucket="primary"),
        _company("B", source_bucket="adjacent"),
        _company("C", source_bucket="training"),
    ]

    report = data_quality.generate_report(companies)

    assert "Comps universe candidates: 2" in report
    assert "Training-only candidates: 1" in report
