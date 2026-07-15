import numpy as np
import pandas as pd

import src.scorer as scorer

N_ROWS = 60
FINANCIAL_COLUMNS = (
    "revenue_ttm_log", "ebitda_margin", "gross_margin",
    "revenue_cagr_3yr", "net_debt_ebitda", "capex_revenue",
)


def _synthetic_feature_matrix(n=N_ROWS, seed=42):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n)]

    ebitda_margin = rng.uniform(0.05, 0.35, n)
    revenue_cagr_3yr = rng.normal(0.05, 0.05, n)
    df = pd.DataFrame({
        "revenue_ttm_log": rng.normal(5, 1, n),
        "ebitda_margin": ebitda_margin,
        "gross_margin": rng.uniform(0.2, 0.6, n),
        "revenue_cagr_3yr": revenue_cagr_3yr,
        "net_debt_ebitda": rng.normal(2.0, 1.0, n),
        "capex_revenue": rng.uniform(0.01, 0.08, n),
        "business_model_services": rng.integers(0, 2, n),
    }, index=tickers)

    noise = rng.normal(0, 0.3, n)
    df["ev_ebitda_log"] = 2.3 + 0.5 * ebitda_margin + 0.3 * revenue_cagr_3yr + noise

    return df


def _medians(feature_matrix):
    global_medians = {col: float(feature_matrix[col].median()) for col in FINANCIAL_COLUMNS}
    return {"by_group": {"manufacturing": global_medians}, "global": global_medians}


def _target_config():
    return {"revenue_usd_mm": 150, "ebitda_margin_estimate": 0.18}


def _target_llm_features():
    return {
        "business_model": "manufacturing",
        "revenue_recurrence": "high",
        "customer_type": "B2B",
        "capital_intensity": "asset_heavy",
        "primary_value_driver": "scale",
    }


def _run(fm=None):
    fm = fm if fm is not None else _synthetic_feature_matrix()
    return scorer.run(fm, _target_config(), _target_llm_features(), _medians(fm))


def test_company_scores_has_all_tickers():
    fm = _synthetic_feature_matrix()
    result = _run(fm)

    assert len(result["company_scores"]) == len(fm)


def test_residual_abs_is_nonnegative_distance():
    result = _run()

    assert (result["company_scores"]["residual_abs"] >= 0).all()


def test_company_closer_to_target_profile_has_smaller_distance():
    fm = _synthetic_feature_matrix()
    target_config = _target_config()  # revenue_usd_mm=150 -> revenue_ttm_log ~ log1p(150) = 5.02
    medians = _medians(fm)

    near = fm.iloc[0].copy()
    near["revenue_ttm_log"] = np.log1p(150)
    near["ebitda_margin"] = 0.18
    far = fm.iloc[1].copy()
    far["revenue_ttm_log"] = np.log1p(150_000)
    far["ebitda_margin"] = 0.01

    fm = fm.copy()
    fm.iloc[0] = near
    fm.iloc[1] = far

    result = scorer.run(fm, target_config, _target_llm_features(), medians)
    scores = result["company_scores"]

    assert scores.loc[fm.index[0], "residual_abs"] < scores.loc[fm.index[1], "residual_abs"]


def test_feature_distance_breakdown_excludes_label_and_sums_to_features():
    fm = _synthetic_feature_matrix()
    result = _run(fm)
    top_tickers = list(fm.index[:5])

    breakdown = scorer.feature_distance_breakdown(result["feature_distance_sq_diff"], top_tickers, top_n=10)

    assert set(breakdown["feature"]) == set(FINANCIAL_COLUMNS)
    assert "ev_ebitda_log" not in set(breakdown["feature"])


def test_no_feature_weights_config_reproduces_unweighted_distance():
    fm = _synthetic_feature_matrix()
    medians = _medians(fm)

    unweighted = scorer.run(fm, _target_config(), _target_llm_features(), medians)
    explicit_empty = scorer.run(fm, _target_config(), _target_llm_features(), medians, {})

    pd.testing.assert_series_equal(
        unweighted["company_scores"]["residual_abs"], explicit_empty["company_scores"]["residual_abs"],
    )


def test_feature_weight_changes_ranking():
    # Build two companies equidistant from the target under unweighted
    # distance (one matches on ebitda_margin, the other on revenue_cagr_3yr)
    # — heavily upweighting ebitda_margin should make the company that
    # matches the target's ebitda_margin rank closer.
    fm = _synthetic_feature_matrix()
    medians = _medians(fm)
    target_ebitda_margin = 0.18

    matches_margin = fm.iloc[0].copy()
    matches_margin["ebitda_margin"] = target_ebitda_margin
    matches_growth = fm.iloc[1].copy()
    matches_growth["ebitda_margin"] = fm["ebitda_margin"].mean() + 0.5  # far from target on margin

    fm = fm.copy()
    fm.iloc[0] = matches_margin
    fm.iloc[1] = matches_growth

    weighted = scorer.run(
        fm, _target_config(), _target_llm_features(), medians,
        {"manufacturing": {"ebitda_margin": 50.0}},
    )
    scores = weighted["company_scores"]

    assert scores.loc[fm.index[0], "residual_abs"] < scores.loc[fm.index[1], "residual_abs"]


def test_feature_weights_uses_matching_business_model_template():
    config = {"manufacturing": {"ebitda_margin": 2.0}, "default": {"ebitda_margin": 9.0}}

    weights = scorer._feature_weights(config, "manufacturing")

    assert weights["ebitda_margin"] == 2.0
    assert weights["gross_margin"] == scorer.DEFAULT_WEIGHT  # unspecified -> 1.0


def test_feature_weights_falls_back_to_default_template_for_unmatched_business_model():
    config = {"manufacturing": {"ebitda_margin": 2.0}, "default": {"ebitda_margin": 9.0}}

    weights = scorer._feature_weights(config, "marketplace")

    assert weights["ebitda_margin"] == 9.0


def test_feature_weights_falls_back_to_default_template_for_null_business_model():
    config = {"manufacturing": {"ebitda_margin": 2.0}, "default": {"ebitda_margin": 9.0}}

    weights = scorer._feature_weights(config, None)

    assert weights["ebitda_margin"] == 9.0


def test_feature_weights_all_default_when_no_template_or_default_matches():
    weights = scorer._feature_weights({"manufacturing": {"ebitda_margin": 2.0}}, "marketplace")

    assert (weights == scorer.DEFAULT_WEIGHT).all()


def _full_target_config():
    return {
        "revenue_usd_mm": 150,
        "ebitda_margin_estimate": 0.18,
        "gross_margin_estimate": 0.35,
        "revenue_cagr_3yr_estimate": 0.04,
        "net_debt_ebitda_estimate": 2.0,
        "capex_revenue_estimate": 0.05,
    }


def test_observed_features_are_only_the_provided_ones():
    # The default target config supplies revenue + ebitda_margin only.
    fm = _synthetic_feature_matrix()
    result = _run(fm)

    assert set(result["observed_target_features"]) == {"revenue_ttm_log", "ebitda_margin"}


def test_observed_features_includes_all_six_when_all_provided():
    fm = _synthetic_feature_matrix()
    result = scorer.run(fm, _full_target_config(), _target_llm_features(), _medians(fm))

    assert set(result["observed_target_features"]) == set(FINANCIAL_COLUMNS)


def test_imputed_feature_does_not_affect_ranking():
    # Two companies identical on the provided features (revenue, ebitda_margin)
    # but differing on gross_margin — which the default target does NOT provide,
    # so it's imputed and must not influence the distance. They should tie.
    fm = _synthetic_feature_matrix()
    medians = _medians(fm)

    a = fm.iloc[0].copy()
    b = fm.iloc[1].copy()
    for col in FINANCIAL_COLUMNS:
        b[col] = a[col]
    b["gross_margin"] = a["gross_margin"] + 0.25  # only difference, on an imputed feature

    fm = fm.copy()
    fm.iloc[0] = a
    fm.iloc[1] = b

    scores = scorer.run(fm, _target_config(), _target_llm_features(), medians)["company_scores"]

    assert scores.loc[fm.index[0], "residual_abs"] == scores.loc[fm.index[1], "residual_abs"]


def test_providing_a_feature_makes_it_drive_ranking():
    # Same two companies as above, but now the target provides gross_margin.
    # The company matching the target's gross_margin should now rank closer.
    fm = _synthetic_feature_matrix()
    medians = _medians(fm)
    target = _full_target_config()

    a = fm.iloc[0].copy()
    b = fm.iloc[1].copy()
    for col in FINANCIAL_COLUMNS:
        b[col] = a[col]
    a["gross_margin"] = target["gross_margin_estimate"]  # matches target
    b["gross_margin"] = target["gross_margin_estimate"] + 0.25  # far from target

    fm = fm.copy()
    fm.iloc[0] = a
    fm.iloc[1] = b

    scores = scorer.run(fm, target, _target_llm_features(), medians)["company_scores"]

    assert scores.loc[fm.index[0], "residual_abs"] < scores.loc[fm.index[1], "residual_abs"]


def test_no_provided_features_falls_back_to_all_features():
    fm = _synthetic_feature_matrix()
    result = scorer.run(fm, {}, _target_llm_features(), _medians(fm))

    assert set(result["observed_target_features"]) == set(FINANCIAL_COLUMNS)
    # Degenerate all-zero distances would mean nothing got scored apart.
    assert result["company_scores"]["residual_abs"].nunique() > 1
