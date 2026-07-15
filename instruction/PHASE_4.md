# Phase 4: Feature Engineering + XGBoost Model

## Project Context

You are building a PE Comparable Company Analysis Pipeline. This tool helps
PE analysts build a comparable company set (comps) for valuing private companies.
The pipeline fetches public company data, uses LLM to extract business model
features, then uses XGBoost to predict EV/EBITDA multiples — enabling
identification of the most "clean" and relevant comparable companies.

This is Phase 4 of 6.

---

## What Already Exists (Do Not Modify)

- `src/universe_builder.py` — builds ticker list
- `src/fetcher.py` — financial data with caching, retry, logging
- `src/data_quality.py` — data quality report
- `src/llm_analyzer.py` — LLM extraction with checkpoint, judge, retry
- `data/cache/` — ~300 company JSON files
- `data/checkpoints/llm_checkpoint.json` — LLM extraction results for all companies
- `outputs/data_quality_report.txt` — field completeness statistics
- All 17 previous tests passing

---

## What To Build in Phase 4

Two new modules:
- `src/feature_builder.py`
- `src/scorer.py`

Two new test files:
- `tests/test_feature_builder.py`
- `tests/test_scorer.py`

---

## src/feature_builder.py

### Purpose

Merge financial data from the fetcher with LLM extraction results from
llm_analyzer into a clean numeric feature matrix that XGBoost can train on.

### Input

- `companies`: list of company dicts from fetcher
- `llm_features`: dict mapping ticker → extraction result from llm_analyzer

### Output

A tuple of two objects:
1. A pandas DataFrame where each row is one company, columns are features
   plus the label column `ev_ebitda_log`
2. A pandas Series of raw (untransformed) `ev_ebitda` values indexed by ticker,
   for use in the final report

### Rows to drop (in this order)

1. Drop companies where `ev_ebitda` is None — no label, cannot train
2. Drop companies where more than 3 of the 6 financial features are None
3. Drop companies where `llm_features` has `extraction_failed=True`

Log how many rows were dropped at each step.

### Numeric feature transformations

Apply these transformations to financial fields:

- `revenue_ttm_usd_mm` → apply `numpy.log1p()`, store as `revenue_ttm_log`
  Reason: revenue is right-skewed (a few giants distort the distribution)
- `ev_ebitda` → apply `numpy.log1p()`, store as `ev_ebitda_log`
  This is the ML label (regression target), not a feature
- All other numeric fields: use as-is (no transformation)

Financial feature columns in the output matrix (6 total):
```
revenue_ttm_log
ebitda_margin
gross_margin
revenue_cagr_3yr
net_debt_ebitda
capex_revenue
```

### Missing value imputation

After transformation, impute remaining None / NaN values:
- For each numeric feature column: replace NaN with the column median
- Calculate medians on the training data only
- Store the medians dict so it can be reused when predicting on new data
- Log which columns had imputation applied and how many values were filled

### LLM feature encoding

Convert LLM categorical fields to numeric using one-hot encoding
via `pandas.get_dummies()`:

Fields to encode:
- `business_model` (up to 6 categories)
- `revenue_recurrence` (3 categories)
- `customer_type` (4 categories)
- `capital_intensity` (3 categories)
- `primary_value_driver` (5 categories)

Use `drop_first=True` in get_dummies to avoid multicollinearity.

Prefix each column with the field name:
`business_model_services`, `revenue_recurrence_medium`, etc.

If a company's LLM extraction is missing a field (value is None),
treat it as its own category: fill None before encoding with the
string "unknown" for that field.

Do NOT include these LLM fields as features:
- `sub_sector_description` (free text, not encodable this way)
- `confidence`, `judge_score`, `judge_reason`, `low_confidence_flag`

### Final feature matrix column order

1. The 6 financial features (numeric)
2. All one-hot encoded LLM features (order determined by get_dummies output)
3. The label column `ev_ebitda_log` (last column)

Also attach the ticker as the DataFrame index (not a column).

### Medians storage

Return the imputation medians as part of the function output so they can
be used in Phase 6 when predicting for the target company:

```python
def build(
    companies: list[dict],
    llm_features: dict[str, dict],
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Returns:
        feature_matrix: DataFrame with features + ev_ebitda_log, indexed by ticker
        ev_ebitda_raw: Series of untransformed ev_ebitda values, indexed by ticker
        imputation_medians: dict mapping column_name -> median_value
    """
```

---

## src/scorer.py

### Purpose

Train an XGBoost regression model to predict EV/EBITDA multiples from
the feature matrix. Use this model to:
1. Get predicted multiples for all companies in the dataset
2. Predict a multiple range for the private target company

### Critical design note

The XGBoost model predicts EV/EBITDA multiples. It does NOT score similarity.

The business logic is: companies whose actual EV/EBITDA is well-explained
by their financial characteristics (i.e., small residual between actual and
predicted) are the most reliable comparable companies — their multiples
reflect underlying business fundamentals, not anomalies.

### XGBoost model configuration

```python
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    early_stopping_rounds=30,
    random_state=42,
)
```

### Training and cross-validation

Use 5-fold cross-validation to estimate model quality:
- Use `sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)`
- In each fold: train on 4/5 of the data, predict on the remaining 1/5
- Collect out-of-fold predictions for all companies
- Calculate RMSE of out-of-fold predictions in log space
- Convert the log-space RMSE to multiple space for human readability:
  `cv_rmse_multiple_space = numpy.expm1(log_rmse)`
  (this is the ± error in actual EV/EBITDA multiples)
- After CV: train a final model on ALL data (no held-out set)
- This final model is used for predictions

### SHAP feature importance

After training the final model:
- Use `shap.TreeExplainer(model)` on the full training set
- Calculate mean absolute SHAP value per feature
- Return as a DataFrame with columns: `feature`, `mean_abs_shap`
  sorted by `mean_abs_shap` descending

Important sanity check: `ebitda_margin` and `revenue_cagr_3yr`
should appear in the top 5 features. If they do not, log a warning:
`WARNING: Expected financial features not in top 5 — check data quality`

### Company scores output

Calculate for each company:
- `ev_ebitda_actual`: raw EV/EBITDA from yfinance
- `ev_ebitda_predicted`: model prediction converted back from log space
  using `numpy.expm1()`
- `residual_abs`: absolute difference between actual and predicted
- `residual_pct`: residual_abs / ev_ebitda_actual

Return as a DataFrame with ticker as index.

### Model persistence

- Save the trained model to `data/models/xgb_model.json` after training
  using `model.save_model()`
- Create `data/models/` directory if it does not exist
- On the next run: if `data/models/xgb_model.json` exists AND
  `force_retrain=False`, load the model instead of retraining
  Log: `INFO: Loaded existing model from data/models/xgb_model.json`
- Also save the imputation medians to `data/models/imputation_medians.json`

### Target company prediction

```python
def predict_target(
    model: xgb.XGBRegressor,
    target_config: dict,
    target_llm_features: dict,
    feature_columns: list[str],
    imputation_medians: dict,
    cv_rmse_log: float,
) -> dict:
    """
    Predict EV/EBITDA for the private target company.

    Build a single-row feature vector from:
    - target_config financial estimates (revenue_usd_mm, ebitda_margin_estimate)
    - target_llm_features (from llm_analyzer.analyze_target)
    - Use imputation_medians for any missing fields

    Apply the same transformations as feature_builder:
    - log1p on revenue
    - one-hot encode LLM fields using the same column structure as training

    Returns:
    {
        "predicted_ev_ebitda": float,   # numpy.expm1(model.predict(...))
        "range_low": float,             # predicted - 1.5 * cv_rmse_multiple_space
        "range_high": float,            # predicted + 1.5 * cv_rmse_multiple_space
        "cv_rmse_multiple_space": float
    }
    """
```

### Public interface

```python
def run(
    feature_matrix: pd.DataFrame,
    target_config: dict,
    target_llm_features: dict,
    force_retrain: bool = False,
) -> dict:
    """
    Full scoring run.

    Returns:
    {
        "model": trained XGBRegressor,
        "feature_columns": list[str],
        "cv_rmse_log": float,
        "cv_rmse_multiple_space": float,
        "feature_importance": pd.DataFrame,   # columns: feature, mean_abs_shap
        "company_scores": pd.DataFrame,       # index: ticker, columns: actual/predicted/residuals
        "target_prediction": dict,            # from predict_target()
    }
    """
```

---

## tests/test_feature_builder.py

Write these 5 tests:

1. `test_rows_missing_label_dropped`
   Include 2 companies with ev_ebitda=None in input.
   Assert neither appears in the output DataFrame.

2. `test_revenue_log_transform_applied`
   Include a company with revenue_ttm_usd_mm=100.0.
   Assert the `revenue_ttm_log` column value equals numpy.log1p(100.0).

3. `test_ev_ebitda_log_transform_is_label`
   Include a company with ev_ebitda=12.0.
   Assert `ev_ebitda_log` column value equals numpy.log1p(12.0).

4. `test_nan_imputed_with_median`
   Include 3 companies: two with ebitda_margin=[0.20, 0.30], one with None.
   Assert the None is filled with 0.25 (median of 0.20 and 0.30).

5. `test_llm_fields_one_hot_encoded`
   Include companies with different business_model values.
   Assert the output DataFrame has columns starting with "business_model_".

---

## tests/test_scorer.py

Write these 5 tests:

1. `test_company_scores_has_all_tickers`
   After running scorer.run() on sample data, assert company_scores
   has the same number of rows as the input feature_matrix.

2. `test_cv_rmse_is_positive`
   Assert `cv_rmse_multiple_space` is a positive float.

3. `test_feature_importance_has_correct_shape`
   Assert feature_importance DataFrame has one row per feature column
   (excluding ev_ebitda_log).

4. `test_model_saved_to_disk`
   After running scorer.run(), assert `data/models/xgb_model.json` exists.

5. `test_model_loaded_on_second_run`
   Run scorer.run() twice.
   Mock xgb.XGBRegressor.fit to count calls.
   Assert fit() was only called once (second run loads from disk).

For scorer tests, use synthetic feature data (at least 50 rows with
random values) rather than real data, to keep tests fast and deterministic.

---

## What NOT to do

- Do NOT use similarity scoring as the XGBoost objective
- Do NOT generate ML labels from rules — ev_ebitda from yfinance is the label
- Do NOT include `ev_ebitda_log` as a feature (it is the label only)
- Do NOT implement the report or pipeline orchestrator yet
- Do NOT use real API calls in tests

---

## Definition of Done for Phase 4

**Tests:**
- `pytest tests/ -v` passes with 0 failures (17 existing + 10 new = 27 total)

**Model quality check (run the actual model):**
Load all cache files and the LLM checkpoint, build the feature matrix,
run scorer.run(), then check:

1. Feature matrix has at least 150 rows (if fewer, check data quality report
   from Phase 2 to understand where the dropoff happened)

2. `cv_rmse_multiple_space` is between 2.0 and 6.0
   - Below 2.0: likely overfitting, check for data leakage
   - Above 6.0: model is not learning, check features and data quality

3. Check `feature_importance` top 5 features — `ebitda_margin` or
   `revenue_cagr_3yr` should appear. If neither does, something is wrong
   with the data pipeline.

4. Check 10 random rows in `company_scores`:
   - `ev_ebitda_predicted` values should be between 3x and 40x for most companies
   - No negative predicted values
   - `residual_pct` for most companies should be under 50%
   (a few outliers above 50% are normal)

5. `data/models/xgb_model.json` and `data/models/imputation_medians.json` exist

6. Run scorer.run() a second time — it should load from disk and complete
   significantly faster than the first run
