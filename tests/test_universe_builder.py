import pytest

import src.universe_builder as universe_builder


@pytest.fixture(autouse=True)
def sane_sic_preflight(mocker):
    """
    universe_builder.build() now preflights every SIC code against SEC
    (see _preflight_sic_codes), which would hit the real network in these
    tests. Default every code to a healthy filer count; the preflight
    tests below override this with their own patch of the same target.
    """
    mocker.patch(
        "src.universe_builder.sic_universe_builder.preflight_sic_codes",
        side_effect=lambda sics: {sic: 10 for sic in sics},
    )


def test_filter_by_market_cap_below_threshold_filtered_out():
    companies = [{"ticker": "CCC", "source_bucket": "primary", "market_cap_usd_mm": 10.0}]

    result = universe_builder.filter_by_market_cap(companies)

    assert result == []


def test_filter_by_market_cap_above_threshold_kept():
    companies = [{"ticker": "DDD", "source_bucket": "adjacent", "market_cap_usd_mm": 5_000.0}]

    result = universe_builder.filter_by_market_cap(companies)

    assert result == companies


def test_filter_by_market_cap_missing_value_kept():
    companies = [{"ticker": "BBB", "source_bucket": "primary", "market_cap_usd_mm": None}]

    result = universe_builder.filter_by_market_cap(companies)

    assert result == companies


def test_filter_by_market_cap_makes_no_fmp_calls(mocker):
    mock_get_profile = mocker.patch("src.fmp_client.get_profile")
    companies = [{"ticker": "EEE", "source_bucket": "primary", "market_cap_usd_mm": 5_000.0}]

    universe_builder.filter_by_market_cap(companies)

    mock_get_profile.assert_not_called()


def test_filter_by_financials_applies_configured_revenue_and_margin_band(make_config):
    companies = [
        {"ticker": "TOO_SMALL", "revenue_ttm_usd_mm": 20.0, "ebitda_margin": 0.20},
        {"ticker": "TOO_LARGE", "revenue_ttm_usd_mm": 900.0, "ebitda_margin": 0.20},
        {"ticker": "LOW_MARGIN", "revenue_ttm_usd_mm": 200.0, "ebitda_margin": 0.01},
        {"ticker": "KEEP", "revenue_ttm_usd_mm": 200.0, "ebitda_margin": 0.20},
    ]
    config = make_config(universe={"min_revenue_usd_mm": 30, "max_revenue_usd_mm": 800, "min_ebitda_margin": 0.05})

    result = universe_builder.filter_by_financials(companies, config)

    assert [company["ticker"] for company in result] == ["KEEP"]


def test_filter_by_financials_keeps_missing_values(make_config):
    companies = [{"ticker": "UNKNOWN", "revenue_ttm_usd_mm": None, "ebitda_margin": None}]
    config = make_config(universe={"min_revenue_usd_mm": 30, "max_revenue_usd_mm": 800, "min_ebitda_margin": 0.05})

    result = universe_builder.filter_by_financials(companies, config)

    assert result == companies


def test_build_uses_sic_discovery_for_primary_bucket(mocker, make_config):
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: ["SIC1"] if sics == ["1111"] else [],
    )

    result = universe_builder.build(make_config())

    assert result[0]["ticker"] == "SIC1"
    assert result[0]["source_bucket"] == "primary"
    assert result[0]["matched_sic_codes"] == ["1111"]


def test_build_adjacent_excludes_overlap_with_primary(mocker, make_config):
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: ["SHARED"],
    )

    result = universe_builder.build(make_config())

    assert [r["ticker"] for r in result].count("SHARED") == 1
    assert result[0]["source_bucket"] == "primary"


def test_build_respects_per_bucket_quota(mocker, make_config):
    primary_pool = [f"P{i}" for i in range(10)]
    adjacent_pool = [f"A{i}" for i in range(10)]
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: primary_pool if sics == ["1111"] else adjacent_pool,
    )

    result = universe_builder.build(make_config(universe={"max_candidates": 10, "primary_allocation_pct": 0.8}))

    assert len([r for r in result if r["ticker"].startswith("P")]) == 8
    assert len([r for r in result if r["ticker"].startswith("A")]) == 2
    assert len(result) == 10


def test_build_injects_must_include_tickers_after_quota(mocker, make_config):
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: ["BASE"] if sics == ["1111"] else [],
    )
    config = make_config(universe={"max_candidates": 1, "must_include_tickers": ["manual"]})

    result = universe_builder.build(config)

    assert [r["ticker"] for r in result] == ["BASE", "MANUAL"]
    assert result[1]["candidate_source"] == "analyst_specified"


def test_build_excludes_tickers_case_insensitively_and_wins_over_must_include(mocker, make_config):
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: ["DROP", "KEEP"] if sics == ["1111"] else [],
    )
    config = make_config(universe={"must_include_tickers": ["drop", "manual"], "exclude_tickers": ["DROP", "MANUAL"]})

    result = universe_builder.build(config)

    assert [r["ticker"] for r in result] == ["KEEP"]


def test_build_expands_adjacent_sics_from_seed_tickers_and_must_includes_seed(mocker, make_config):
    mocker.patch("src.universe_builder.sic_universe_builder.fetch_sic_for_ticker", return_value="3569")
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: ["BASE"] if sics == ["1111"] else (["SEEDPEER"] if sics == ["3569"] else []),
    )
    config = make_config(
        target_company={"adjacent_sic_codes": []},
        universe={"seed_tickers": ["seed"]},
    )

    result = universe_builder.build(config)

    by_ticker = {r["ticker"]: r for r in result}
    assert "SEEDPEER" in by_ticker
    assert by_ticker["SEEDPEER"]["adjacent_matched_sic_codes"] == ["3569"]
    assert by_ticker["SEED"]["candidate_source"] == "analyst_specified"


def test_build_exclude_tickers_remove_seed_ticker_itself(mocker, make_config):
    mocker.patch("src.universe_builder.sic_universe_builder.fetch_sic_for_ticker", return_value="3569")
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: [] if sics == ["1111"] else ["SEEDPEER"],
    )
    config = make_config(
        target_company={"adjacent_sic_codes": []},
        universe={"seed_tickers": ["SEED"], "exclude_tickers": ["SEED"]},
    )

    result = universe_builder.build(config)

    assert [r["ticker"] for r in result] == ["SEEDPEER"]


def test_build_maps_sic_to_industry_cluster(mocker, make_config):
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: ["AUTO"] if sics == ["3714"] else [],
    )
    config = make_config(
        target_company={"primary_sic_codes": ["3714"], "adjacent_sic_codes": []},
        universe={"sic_clusters": {"3714": "auto_parts"}},
    )

    result = universe_builder.build(config)

    assert result == [{
        "ticker": "AUTO",
        "matched_sic_codes": ["3714"],
        "primary_matched_sic_codes": ["3714"],
        "adjacent_matched_sic_codes": [],
        "source_bucket": "primary",
        "sic_2_digit": "37",
        "sic_3_digit": "371",
        "industry_cluster": "auto_parts",
        "candidate_source": "sec_sic",
    }]


def test_build_appends_embedding_discovery_candidates_after_sic_quota(mocker, make_config):
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: ["SIC1", "SIC2"] if sics == ["1111"] else [],
    )
    discover = mocker.patch(
        "src.universe_builder.embedding_universe_builder.discover",
        return_value=[
            {
                "ticker": "SEM1",
                "matched_sic_codes": [],
                "primary_matched_sic_codes": [],
                "adjacent_matched_sic_codes": [],
                "source_bucket": "embedding",
                "sic_2_digit": None,
                "sic_3_digit": None,
                "industry_cluster": None,
                "candidate_source": "embedding",
                "embedding_similarity": 0.91,
            },
        ],
    )

    result = universe_builder.build(make_config(universe={"max_candidates": 1, "primary_allocation_pct": 1.0, "discovery_mode": "sic+embedding"}))

    assert [r["ticker"] for r in result] == ["SIC1", "SEM1"]
    assert result[1]["source_bucket"] == "embedding"
    assert result[1]["embedding_similarity"] == 0.91
    discover.assert_called_once()


def test_build_keeps_sic_source_when_embedding_candidate_duplicates_sic(mocker, make_config):
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: ["SHARED"] if sics == ["1111"] else [],
    )
    mocker.patch(
        "src.universe_builder.embedding_universe_builder.discover",
        return_value=[
            {
                "ticker": "SHARED",
                "matched_sic_codes": [],
                "primary_matched_sic_codes": [],
                "adjacent_matched_sic_codes": [],
                "source_bucket": "embedding",
                "sic_2_digit": None,
                "sic_3_digit": None,
                "industry_cluster": None,
                "candidate_source": "embedding",
                "embedding_similarity": 0.88,
            },
        ],
    )

    result = universe_builder.build(make_config(universe={"discovery_mode": "sic+embedding"}))

    assert len(result) == 1
    assert result[0]["ticker"] == "SHARED"
    assert result[0]["source_bucket"] == "primary"
    assert result[0]["candidate_source"] == "sec_sic"
    assert result[0]["embedding_similarity"] == 0.88


def test_build_suggest_sic_embedding_uses_validated_suggestions_without_ground_truth(mocker, make_config):
    config = make_config(
        target_company={
            "description": "Makes pumps for chemical processing.",
            "primary_sic_codes": ["1111"],
            "adjacent_sic_codes": ["2222"],
        },
        universe={"discovery_mode": "suggest-sic+embedding"},
    )

    def suggest(received):
        assert received.target_company.description == "Makes pumps for chemical processing."
        assert received.llm.max_tokens == 1200
        assert "banker" not in received.model_dump_json().lower()
        return [
            {"sic_code": "3561", "bucket": "primary"},
            {"sic_code": "3569", "bucket": "adjacent"},
        ]

    mocker.patch("src.universe_builder.llm_analyzer.suggest_sic_codes", side_effect=suggest)
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: [f"SIC-{sics[0]}"] if len(sics) == 1 else [],
    )
    embedding = mocker.patch("src.universe_builder.embedding_universe_builder.discover", return_value=[])

    result = universe_builder.build(config)

    passed_config = embedding.call_args.args[0]
    assert passed_config.target_company.primary_sic_codes == ["1111", "3561"]
    assert passed_config.target_company.adjacent_sic_codes == ["2222", "3569"]
    assert {row["ticker"] for row in result} >= {"SIC-1111", "SIC-3561", "SIC-2222", "SIC-3569"}

    # The suggested codes only exist on build()'s local config copy, so the
    # snapshot is the audit trail for what discovery actually searched.
    snapshot = universe_builder.last_discovery_snapshot()
    assert snapshot["primary_sic_codes"] == ["1111", "3561"]
    assert snapshot["adjacent_sic_codes"] == ["2222", "3569"]


def test_build_aborts_on_zero_filer_sic_code(mocker, make_config):
    mocker.patch(
        "src.universe_builder.sic_universe_builder.preflight_sic_codes",
        side_effect=lambda sics: {sic: (0 if sic == "3535" else 10) for sic in sics},
    )
    discover = mocker.patch("src.universe_builder.sic_universe_builder.discover_tickers_by_sic")
    config = make_config(target_company={"primary_sic_codes": ["3535", "3714"], "adjacent_sic_codes": []})

    with pytest.raises(ValueError, match="no SEC filers with a usable US ticker for SIC code\\(s\\) 3535"):
        universe_builder.build(config)
    discover.assert_not_called()


def test_build_aborts_on_broad_sic_code_unless_allowed(mocker, make_config):
    mocker.patch(
        "src.universe_builder.sic_universe_builder.preflight_sic_codes",
        side_effect=lambda sics: {sic: (2000 if sic == "7373" else 10) for sic in sics},
    )
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: ["TICK"],
    )
    config = make_config(target_company={"primary_sic_codes": ["7373"], "adjacent_sic_codes": []})

    with pytest.raises(ValueError, match="7373 \\(2000 filers\\)"):
        universe_builder.build(config)

    allowed = make_config(
        target_company={"primary_sic_codes": ["7373"], "adjacent_sic_codes": []},
        universe={"allow_broad_sic_codes": True},
    )
    assert universe_builder.build(allowed)  # override lets the run proceed
