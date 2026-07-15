import src.universe_builder as universe_builder


def test_market_cap_fetch_failure_keeps_ticker(mocker):
    mocker.patch("src.universe_builder.fmp_client.get_profile", side_effect=Exception("FMP down"))

    result = universe_builder._filter_by_market_cap(["AAA"])

    assert result == ["AAA"]


def test_market_cap_missing_keeps_ticker(mocker):
    mocker.patch("src.universe_builder.fmp_client.get_profile", return_value={})

    result = universe_builder._filter_by_market_cap(["BBB"])

    assert result == ["BBB"]


def test_market_cap_below_threshold_filtered_out(mocker):
    mocker.patch("src.universe_builder.fmp_client.get_profile", return_value={"marketCap": 10_000_000})

    result = universe_builder._filter_by_market_cap(["CCC"])

    assert result == []


def test_market_cap_above_threshold_kept(mocker):
    mocker.patch("src.universe_builder.fmp_client.get_profile", return_value={"marketCap": 5_000_000_000})

    result = universe_builder._filter_by_market_cap(["DDD"])

    assert result == ["DDD"]


def test_market_cap_second_call_uses_cache_not_fmp(mocker):
    mock_get_profile = mocker.patch(
        "src.universe_builder.fmp_client.get_profile", return_value={"marketCap": 5_000_000_000},
    )

    universe_builder._filter_by_market_cap(["EEE"])
    universe_builder._filter_by_market_cap(["EEE"])

    mock_get_profile.assert_called_once_with("EEE")


def _config(max_candidates=10, primary_allocation_pct=0.8, primary_sics=None, adjacent_sics=None):
    return {
        "target_company": {
            "primary_sic_codes": primary_sics if primary_sics is not None else ["1111"],
            "adjacent_sic_codes": adjacent_sics if adjacent_sics is not None else ["2222"],
        },
        "universe": {
            "max_candidates": max_candidates,
            "primary_allocation_pct": primary_allocation_pct,
        },
    }


def test_build_uses_sic_discovery_for_primary_bucket(mocker):
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: ["SIC1"] if sics == ["1111"] else [],
    )
    mocker.patch("src.universe_builder.fmp_client.get_profile", return_value={"marketCap": 5_000_000_000})

    result = universe_builder.build(_config())

    assert "SIC1" in result


def test_build_adjacent_excludes_overlap_with_primary(mocker):
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: ["SHARED"],
    )
    mocker.patch("src.universe_builder.fmp_client.get_profile", return_value={"marketCap": 5_000_000_000})

    result = universe_builder.build(_config())

    assert result.count("SHARED") == 1


def test_build_respects_per_bucket_quota(mocker):
    primary_pool = [f"P{i}" for i in range(10)]
    adjacent_pool = [f"A{i}" for i in range(10)]
    mocker.patch(
        "src.universe_builder.sic_universe_builder.discover_tickers_by_sic",
        side_effect=lambda sics: primary_pool if sics == ["1111"] else adjacent_pool,
    )
    mocker.patch("src.universe_builder.fmp_client.get_profile", return_value={"marketCap": 5_000_000_000})

    result = universe_builder.build(_config(max_candidates=10, primary_allocation_pct=0.8))

    assert len([t for t in result if t.startswith("P")]) == 8
    assert len([t for t in result if t.startswith("A")]) == 2
    assert len(result) == 10
