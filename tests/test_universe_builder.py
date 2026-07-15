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
    mocker.patch("src.universe_builder.fmp_client.get_profile", return_value={"marketCap": 100_000_000})

    result = universe_builder._filter_by_market_cap(["CCC"])

    assert result == []


def test_market_cap_above_threshold_kept(mocker):
    mocker.patch("src.universe_builder.fmp_client.get_profile", return_value={"marketCap": 5_000_000_000})

    result = universe_builder._filter_by_market_cap(["DDD"])

    assert result == ["DDD"]
