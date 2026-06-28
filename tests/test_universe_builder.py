import src.universe_builder as universe_builder


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
