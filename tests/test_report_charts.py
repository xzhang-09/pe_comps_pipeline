import re

import pandas as pd

from src.report_charts import (
    LABEL_CHAR_WIDTH,
    LABEL_HEIGHT,
    SCATTER_MARGIN,
    _place_label,
    _revenue_multiple_scatter_svg,
)


def _boxes_overlap(a, b):
    (ax0, ax1, ay), (bx0, bx1, by) = a, b
    return not (ax1 < bx0 or ax0 > bx1) and abs(ay - by) < LABEL_HEIGHT - 1


def test_place_label_dodges_earlier_labels():
    placed = []
    bounds = (0.0, 720.0)
    first = _place_label(100.0, 100.0, "AAAA", placed, bounds)
    second = _place_label(100.0, 100.0, "BBBB", placed, bounds)
    third = _place_label(100.0, 100.0, "CCCC", placed, bounds)

    assert first != second != third
    assert len(placed) == 3
    assert not _boxes_overlap(placed[0], placed[1])
    assert not _boxes_overlap(placed[0], placed[2])
    assert not _boxes_overlap(placed[1], placed[2])


def test_place_label_prefers_left_side_near_right_bound():
    placed = []
    bounds = (0.0, 720.0)
    x, _, anchor = _place_label(710.0, 100.0, "WIDE_LABEL", placed, bounds)
    assert anchor == "end"
    assert x < 710.0


def _scatter_fixture():
    tickers = [f"P{i}" for i in range(8)] + ["T1", "T2", "T3"]
    company_scores = pd.DataFrame(
        {"ev_ebitda_actual": [10.0 + i for i in range(len(tickers))],
         "residual_abs": [0.5] * len(tickers)},
        index=tickers,
    )
    companies_by_ticker = {t: {"revenue_ttm_usd_mm": 100.0 + 50 * i} for i, t in enumerate(tickers)}
    top15_rows = [
        # Two usable comps at an identical position (labels must not overlap)
        # and one review_exclude far above the usable range (must be pinned).
        {"ticker": "T1", "revenue_ttm_usd_mm": 300.0, "ev_ebitda_actual": 12.0, "tier": "core"},
        {"ticker": "T2", "revenue_ttm_usd_mm": 300.0, "ev_ebitda_actual": 12.0, "tier": "secondary"},
        {"ticker": "T3", "revenue_ttm_usd_mm": 400.0, "ev_ebitda_actual": 90.0, "tier": "review_exclude"},
    ]
    return company_scores, companies_by_ticker, tickers, top15_rows


def test_scatter_y_axis_scales_to_usable_comps_and_pins_outliers():
    company_scores, companies_by_ticker, tickers, top15_rows = _scatter_fixture()
    result = _revenue_multiple_scatter_svg(
        company_scores, companies_by_ticker, tickers, top15_rows, target_revenue=150.0,
    )
    assert result is not None
    assert result["n_top_pinned"] == 1
    assert "T3 (90x)" in result["svg"]
    # The 90x outlier must not set the axis: the top tick stays near the
    # usable comps' 12x (12 * 1.15 padding ≈ 14x), nowhere near 90x.
    tick_values = [int(v) for v in re.findall(r">(\d+)x</text>", result["svg"])]
    assert max(tick_values) < 20


def test_scatter_coincident_labels_do_not_overlap():
    company_scores, companies_by_ticker, tickers, top15_rows = _scatter_fixture()
    result = _revenue_multiple_scatter_svg(
        company_scores, companies_by_ticker, tickers, top15_rows, target_revenue=150.0,
    )
    labels = {}
    for match in re.finditer(r'<text x="([\d.]+)" y="([\d.]+)" text-anchor="(\w+)" fill="#333333">(\w+)</text>', result["svg"]):
        x, y, anchor, text = float(match.group(1)), float(match.group(2)), match.group(3), match.group(4)
        x0 = x if anchor == "start" else x - len(text) * LABEL_CHAR_WIDTH
        labels[text] = (x0, x0 + len(text) * LABEL_CHAR_WIDTH, y)
    assert {"T1", "T2"} <= set(labels)
    assert not _boxes_overlap(labels["T1"], labels["T2"])


def test_scatter_labels_do_not_cover_markers_or_escape_plot_top():
    company_scores, companies_by_ticker, tickers, top15_rows = _scatter_fixture()
    result = _revenue_multiple_scatter_svg(
        company_scores, companies_by_ticker, tickers, top15_rows, target_revenue=150.0,
    )
    svg = result["svg"]
    markers = [
        (float(m.group(1)), float(m.group(2)))
        for m in re.finditer(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="5"', svg)
    ]
    markers += [
        (float(m.group(1)), float(m.group(2)) + 5)  # arrow apex -> center-ish
        for m in re.finditer(r'<path d="M([\d.]+) ([\d.]+) L', svg)
    ]
    plot_top = SCATTER_MARGIN["top"]
    for m in re.finditer(r'<text x="([\d.]+)" y="([\d.]+)" text-anchor="(\w+)" fill="#333333">([^<]+)</text>', svg):
        x, y, anchor, text = float(m.group(1)), float(m.group(2)), m.group(3), m.group(4)
        width = len(text) * LABEL_CHAR_WIDTH
        x0 = x if anchor == "start" else x - width
        assert y >= plot_top + 8, f"label {text} escaped above the plot area"
        for mx, my in markers:
            covers = not (x0 + width < mx - 6 or x0 > mx + 6) and abs(y - my) < LABEL_HEIGHT - 1
            assert not covers, f"label {text} covers a marker at ({mx}, {my})"


def test_scatter_legend_sits_outside_plot_area():
    company_scores, companies_by_ticker, tickers, top15_rows = _scatter_fixture()
    result = _revenue_multiple_scatter_svg(
        company_scores, companies_by_ticker, tickers, top15_rows, target_revenue=150.0,
    )
    legend_labels = ("Core", "Secondary", "Review / Exclude", "Eligible pool")
    for label in legend_labels:
        match = re.search(rf'<text x="[\d.]+" y="(\d+)" fill="#333333">{re.escape(label)}</text>', result["svg"])
        assert match, f"legend label {label} missing"
        assert int(match.group(1)) < SCATTER_MARGIN["top"]
