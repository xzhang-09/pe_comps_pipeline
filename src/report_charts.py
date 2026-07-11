"""Hand-built SVG charts for the HTML report (no charting-library dependency
— the rest of the report generates fine without one, and these are the only
charts).

These are pure data-to-SVG-string functions, with no selection or valuation
logic. reporter.py re-exports the shared helpers so existing callers and
tests can keep addressing reporter.<name>.
"""
import math

import pandas as pd

from src.report_valuation import MIN_VALUES_FOR_DISPERSION

SCATTER_WIDTH = 720
# Top margin holds a horizontal legend strip and the target callout OUTSIDE
# the plot area — the legend used to float over the top-right of the plot,
# where it collided with pool dots and with outlier arrows pinned to the top
# edge. Height grows by the same amount so the plot area itself keeps its size.
SCATTER_HEIGHT = 446
SCATTER_MARGIN = {"left": 70, "right": 30, "top": 56, "bottom": 60}
SCATTER_X_TICK_CANDIDATES_USD_MM = (10, 30, 100, 300, 1000, 3000, 10000)


SCATTER_Y_PADDING_MULTIPLE = 1.15
# Approximate glyph width at font-size 11 (Arial), used for label collision
# boxes. SVG has no text measurement; an over-estimate just spaces labels a
# little more generously.
LABEL_CHAR_WIDTH = 6.5
LABEL_HEIGHT = 12


def _place_label(
    cx: float, cy: float, text: str, placed: list,
    x_bounds: tuple[float, float], y_bounds: tuple[float, float] | None = None,
) -> tuple[float, float, str]:
    """
    Greedy label placement with collision avoidance: try to the right of the
    point, then to the left, then nudged down/up one label-height on either
    side. `placed` accumulates occupied boxes (x0, x1, y) across calls —
    pre-seeded with every foreground marker's own box (see the caller), so a
    label dodges dots and pinned arrows too, not just other labels. Labels
    placed later dodge everything placed before them (points are drawn in
    ranking order, so better-ranked comps keep the preferred spot). If every
    candidate collides, fall back to the right — one overlapping label beats
    a label pushed arbitrarily far from its point.

    Returns (x, y, text-anchor).
    """
    width = len(text) * LABEL_CHAR_WIDTH
    right_x, left_x = cx + 7, cx - 7
    candidates = []
    for dy in (0.0, LABEL_HEIGHT, -LABEL_HEIGHT, 2 * LABEL_HEIGHT):
        candidates.append((right_x, cy + 3 + dy, "start"))
        candidates.append((left_x, cy + 3 + dy, "end"))

    def box(x: float, y: float, anchor: str) -> tuple[float, float, float]:
        x0 = x if anchor == "start" else x - width
        return (x0, x0 + width, y)

    def collides(x0: float, x1: float, y: float) -> bool:
        if x0 < x_bounds[0] or x1 > x_bounds[1]:
            return True
        if y_bounds is not None and not (y_bounds[0] <= y <= y_bounds[1]):
            return True
        return any(not (x1 < px0 or x0 > px1) and abs(y - py) < LABEL_HEIGHT - 1 for px0, px1, py in placed)

    for x, y, anchor in candidates:
        x0, x1, y_box = box(x, y, anchor)
        if not collides(x0, x1, y_box):
            placed.append((x0, x1, y_box))
            return (x, y, anchor)

    x, y, anchor = candidates[0]
    placed.append(box(x, y, anchor))
    return (x, y, anchor)


def _revenue_multiple_scatter_svg(
    company_scores: pd.DataFrame, companies_by_ticker: dict, eligible_candidates: list[str],
    top15_rows: list[dict], target_revenue: float | None,
) -> dict | None:
    """
    Plots EV/EBITDA against revenue, log-scaled on the x-axis since revenue
    spans roughly two orders of magnitude across the eligible pool. This
    turns what the rest of the report only describes in prose ("wide range",
    "scale mismatch", "outlier") into something a reader can see in one
    glance: where the target's own revenue sits relative to the comp
    cloud, and how the Top-N comps map to the report's tier colors. The
    eligible pool renders as faint background dots for context; the Top-N
    are the colored, labeled foreground dots.

    The y-axis scales off the *usable* (core/secondary) Top-N points' own
    EV/EBITDA range (with padding), not the full Top-N's or the pool's — the
    review_exclude tier exists precisely to hold statistical outliers
    (observed: a 62x semiconductor name stretching the axis so the 5-20x
    band every usable comp lives in was squeezed into the bottom quarter).
    Top-N points above the resulting y_max are pinned to the top edge as
    labeled up-arrows, so they stay visible without setting the scale. Pool
    points above y_max are dropped; the count of dropped points is returned
    so the caller can disclose how many aren't shown.
    """
    pool_points = []
    for ticker in eligible_candidates:
        revenue = companies_by_ticker.get(ticker, {}).get("revenue_ttm_usd_mm")
        ev_ebitda = company_scores.loc[ticker, "ev_ebitda_actual"] if ticker in company_scores.index else None
        if revenue and revenue > 0 and ev_ebitda is not None:
            pool_points.append((revenue, float(ev_ebitda)))

    if len(pool_points) < MIN_VALUES_FOR_DISPERSION:
        return None

    top15_points = [
        (row["revenue_ttm_usd_mm"], row["ev_ebitda_actual"], row["ticker"], row.get("tier"))
        for row in top15_rows
        if row.get("revenue_ttm_usd_mm") and row["revenue_ttm_usd_mm"] > 0 and row.get("ev_ebitda_actual") is not None
    ]
    if not top15_points:
        return None

    all_revenues_log = [math.log10(r) for r, _ in pool_points] + ([math.log10(target_revenue)] if target_revenue else [])

    x_min, x_max = min(all_revenues_log), max(all_revenues_log)
    x_pad = (x_max - x_min) * 0.08 or 0.5
    x_min, x_max = x_min - x_pad, x_max + x_pad

    y_min = 0.0
    usable_multiples = [m for _, m, _, tier in top15_points if tier != "review_exclude"]
    scale_multiples = usable_multiples or [m for _, m, *_ in top15_points]
    y_max = max(scale_multiples) * SCATTER_Y_PADDING_MULTIPLE
    n_pool_clipped = sum(1 for _, m in pool_points if m > y_max)
    pool_points = [(r, m) for r, m in pool_points if m <= y_max]
    pinned_top15 = [(r, m, t, tier) for r, m, t, tier in top15_points if m > y_max]
    top15_points = [(r, m, t, tier) for r, m, t, tier in top15_points if m <= y_max]

    left, right = SCATTER_MARGIN["left"], SCATTER_WIDTH - SCATTER_MARGIN["right"]
    top, bottom = SCATTER_MARGIN["top"], SCATTER_HEIGHT - SCATTER_MARGIN["bottom"]

    def x_pos(revenue: float) -> float:
        frac = (math.log10(revenue) - x_min) / (x_max - x_min)
        return left + frac * (right - left)

    def y_pos(multiple: float) -> float:
        frac = (multiple - y_min) / (y_max - y_min) if y_max > y_min else 0.0
        return bottom - frac * (bottom - top)

    parts = [
        f'<svg viewBox="0 0 {SCATTER_WIDTH} {SCATTER_HEIGHT}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="11">',
        f'<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" fill="none" stroke="#dddddd"/>',
    ]

    n_y_ticks = 5
    for i in range(n_y_ticks + 1):
        val = y_min + (y_max - y_min) * i / n_y_ticks
        y = y_pos(val)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#555555">{val:.0f}x</text>')

    for revenue in SCATTER_X_TICK_CANDIDATES_USD_MM:
        log_revenue = math.log10(revenue)
        if x_min <= log_revenue <= x_max:
            x = x_pos(revenue)
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#eeeeee"/>')
            label = f"${revenue}mm" if revenue < 1000 else f"${revenue / 1000:.0f}bn"
            parts.append(f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" fill="#555555">{label}</text>')

    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="{SCATTER_HEIGHT - 10}" text-anchor="middle" '
        f'fill="#333333" font-weight="bold">Revenue (log scale)</text>',
    )
    mid_y = (top + bottom) / 2
    parts.append(
        f'<text x="16" y="{mid_y:.1f}" text-anchor="middle" fill="#333333" font-weight="bold" '
        f'transform="rotate(-90 16 {mid_y:.1f})">EV/EBITDA</text>',
    )

    # Labels share one collision registry across normal points, pinned
    # points, and the target callout, so e.g. a pinned outlier's label also
    # dodges a normal point's label sitting just under the top edge. Points
    # are drawn in ranking order — on a clash the better-ranked comp keeps
    # the preferred right-of-dot spot.
    placed_labels: list[tuple[float, float, float]] = []
    label_bounds = (left - 40.0, float(SCATTER_WIDTH - 4))

    if target_revenue:
        tx = x_pos(target_revenue)
        target_label = f"Target (${target_revenue:.0f}mm)"
        target_half_width = len(target_label) * LABEL_CHAR_WIDTH / 2
        parts.append(f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{bottom}" stroke="#1a3a5c" stroke-width="2" stroke-dasharray="4,3"/>')
        parts.append(f'<text x="{tx:.1f}" y="{top - 8}" text-anchor="middle" fill="#1a3a5c" font-weight="bold">{target_label}</text>')
        placed_labels.append((tx - target_half_width, tx + target_half_width, top - 8))

    for revenue, multiple in pool_points:
        parts.append(f'<circle cx="{x_pos(revenue):.1f}" cy="{y_pos(multiple):.1f}" r="3" fill="#cccccc" opacity="0.6"/>')

    tier_colors = {"core": "#2e7d32", "secondary": "#1a3a5c", "review_exclude": "#c0392b"}
    normal_markers = [
        (x_pos(revenue), y_pos(multiple), ticker, tier) for revenue, multiple, ticker, tier in top15_points
    ]
    # Pinned = Top-N comps whose multiple exceeds the usable-comp axis range
    # (in practice review_exclude outliers): up-arrows at the top edge with
    # the actual multiple in the label, so the reader sees they exist without
    # letting them set the scale.
    pinned_markers = [
        (x_pos(revenue), top + 6.0, f"{ticker} ({multiple:.0f}x)", tier) for revenue, multiple, ticker, tier in pinned_top15
    ]

    # Register every foreground marker's box BEFORE placing any label:
    # otherwise a label placed early can't know about a dot or arrow drawn
    # later and ends up underneath it (observed: a pinned arrow landing on
    # the previous outlier's label text).
    for cx, cy, _, _ in normal_markers + pinned_markers:
        placed_labels.append((cx - 6.0, cx + 6.0, cy))

    # Labels stay inside the plot area vertically — without the bound, a
    # label nudged upward off a crowded top edge escapes into the margin
    # band where the legend and target callout live.
    label_y_bounds = (top + 8.0, bottom - 2.0)

    for cx, cy, ticker, tier in normal_markers:
        color = tier_colors.get(tier, "#1a3a5c")
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}" stroke="#ffffff" stroke-width="1"/>')
        lx, ly, anchor = _place_label(cx, cy, ticker, placed_labels, label_bounds, label_y_bounds)
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" fill="#333333">{ticker}</text>')

    for cx, cy, label, tier in pinned_markers:
        color = tier_colors.get(tier, "#1a3a5c")
        parts.append(
            f'<path d="M{cx:.1f} {cy - 5:.1f} L{cx + 5:.1f} {cy + 4:.1f} L{cx - 5:.1f} {cy + 4:.1f} Z" '
            f'fill="{color}" stroke="#ffffff" stroke-width="1"/>',
        )
        lx, ly, anchor = _place_label(cx, cy + 3, label, placed_labels, label_bounds, label_y_bounds)
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" fill="#333333">{label}</text>')

    # Horizontal legend strip in the top margin, outside the plot area — it
    # must never sit over data (pool dots) or the pinned-outlier band.
    legend_items = (
        ("#2e7d32", "Core"),
        ("#1a3a5c", "Secondary"),
        ("#c0392b", "Review / Exclude"),
        ("#cccccc", "Eligible pool"),
    )
    legend_x = float(left)
    legend_y = 16
    for color, label in legend_items:
        parts.append(f'<circle cx="{legend_x:.1f}" cy="{legend_y}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 9:.1f}" y="{legend_y + 4}" fill="#333333">{label}</text>')
        legend_x += 9 + len(label) * LABEL_CHAR_WIDTH + 22

    parts.append("</svg>")
    return {
        "svg": "".join(parts),
        "n_pool_clipped": n_pool_clipped,
        "n_top_pinned": len(pinned_top15),
    }


FOOTBALL_WIDTH = 720
FOOTBALL_MARGIN = {"left": 185, "right": 110, "top": 22, "bottom": 44}
FOOTBALL_ROW_HEIGHT = 34
FOOTBALL_BAR_HEIGHT = 14


def _football_field_rows(
    implied_valuation: dict, revenue_screened_valuation: dict | None,
    discounted_valuation: dict | None, size_adjusted_valuation: dict | None, size_anchor: dict | None,
) -> list[dict]:
    """Assembles the football-field rows, most decision-relevant first for a
    small private target: the size anchor and the discounted ("working") range
    lead, the raw Top-N comp ranges and the regression point follow. Each range
    row carries low/mid/high (P25/median/P75); each point row a single value.
    Ranges use whichever basis the rest of the report leads with (EV/EBITDA,
    else EV/Revenue) so a bar isn't shown on a different footing than the
    headline."""
    def basis_range(block: dict | None) -> dict | None:
        block = block or {}
        return block.get("by_ebitda") or block.get("by_revenue")

    rows: list[dict] = []
    if size_anchor and size_anchor.get("implied_ev"):
        rows.append({"label": f"Size anchor (n={size_anchor.get('n')})", "kind": "point",
                     "value": size_anchor["implied_ev"], "color": "#b8860b", "emphasis": True})
    discounted_range = basis_range(discounted_valuation)
    if discounted_range:
        pct = round((discounted_valuation.get("discount") or 0) * 100)
        rows.append({"label": f"After {pct}% discount", "kind": "range", "color": "#2e7d32", "emphasis": True,
                     "low": discounted_range["p25"], "mid": discounted_range["median"], "high": discounted_range["p75"]})
    screened_range = basis_range(revenue_screened_valuation)
    if screened_range:
        rows.append({"label": f"Size-comparable (n={revenue_screened_valuation.get('n')})", "kind": "range",
                     "color": "#1a3a5c", "low": screened_range["p25"], "mid": screened_range["median"], "high": screened_range["p75"]})
    by_ebitda = implied_valuation.get("by_ebitda")
    if by_ebitda:
        rows.append({"label": "EV/EBITDA (Top-N)", "kind": "range", "color": "#1a3a5c",
                     "low": by_ebitda["p25"], "mid": by_ebitda["median"], "high": by_ebitda["p75"]})
    by_revenue = implied_valuation.get("by_revenue")
    if by_revenue:
        rows.append({"label": "EV/Revenue (Top-N)", "kind": "range", "color": "#1a3a5c",
                     "low": by_revenue["p25"], "mid": by_revenue["median"], "high": by_revenue["p75"]})
    if (
        size_adjusted_valuation
        and size_adjusted_valuation.get("is_significant")
        and size_adjusted_valuation.get("implied_ev")
    ):
        rows.append({"label": "Size-adjusted (regr.)", "kind": "point",
                     "value": size_adjusted_valuation["implied_ev"], "color": "#888888"})
    return rows


def _football_field_svg(
    implied_valuation: dict, revenue_screened_valuation: dict | None,
    discounted_valuation: dict | None, size_adjusted_valuation: dict | None, size_anchor: dict | None,
) -> str | None:
    """Horizontal 'football field' of implied enterprise value by valuation
    method — the canonical one-glance comps output. Returns None when no comp
    range is available (target revenue/EBITDA missing), so the template can
    skip it."""
    rows = _football_field_rows(
        implied_valuation, revenue_screened_valuation, discounted_valuation, size_adjusted_valuation, size_anchor,
    )
    if not any(r["kind"] == "range" for r in rows):
        return None

    values: list[float] = []
    for r in rows:
        values += [r["low"], r["high"]] if r["kind"] == "range" else [r["value"]]
    x_min, x_max = min(values), max(values)
    if x_max <= x_min:
        x_max = x_min + 1.0
    pad = (x_max - x_min) * 0.08 or 1.0
    x_min, x_max = max(0.0, x_min - pad), x_max + pad

    left = FOOTBALL_MARGIN["left"]
    right = FOOTBALL_WIDTH - FOOTBALL_MARGIN["right"]
    top = FOOTBALL_MARGIN["top"]
    plot_bottom = top + len(rows) * FOOTBALL_ROW_HEIGHT
    height = plot_bottom + FOOTBALL_MARGIN["bottom"]

    def x_pos(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * (right - left)

    parts = [
        f'<svg viewBox="0 0 {FOOTBALL_WIDTH} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="11">',
    ]

    n_ticks = 5
    for i in range(n_ticks + 1):
        v = x_min + (x_max - x_min) * i / n_ticks
        x = x_pos(v)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{plot_bottom:.1f}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{x:.1f}" y="{plot_bottom + 16:.1f}" text-anchor="middle" fill="#555555">${v:.0f}mm</text>')

    for idx, r in enumerate(rows):
        cy = top + idx * FOOTBALL_ROW_HEIGHT + FOOTBALL_ROW_HEIGHT / 2
        weight = "bold" if r.get("emphasis") else "normal"
        parts.append(
            f'<text x="{left - 10}" y="{cy + 4:.1f}" text-anchor="end" fill="#333333" font-weight="{weight}">{r["label"]}</text>',
        )
        if r["kind"] == "range":
            x0, x1, xm = x_pos(r["low"]), x_pos(r["high"]), x_pos(r["mid"])
            bar_top = cy - FOOTBALL_BAR_HEIGHT / 2
            parts.append(
                f'<rect x="{x0:.1f}" y="{bar_top:.1f}" width="{max(x1 - x0, 1):.1f}" height="{FOOTBALL_BAR_HEIGHT}" '
                f'fill="{r["color"]}" opacity="0.30"/>',
            )
            parts.append(
                f'<line x1="{xm:.1f}" y1="{bar_top:.1f}" x2="{xm:.1f}" y2="{bar_top + FOOTBALL_BAR_HEIGHT:.1f}" '
                f'stroke="{r["color"]}" stroke-width="2"/>',
            )
            parts.append(
                f'<text x="{right + 6}" y="{cy + 4:.1f}" text-anchor="start" fill="#333333">'
                f'${r["low"]:.0f}–{r["high"]:.0f}mm</text>',
            )
        else:
            x = x_pos(r["value"])
            d = 5
            parts.append(
                f'<path d="M{x:.1f} {cy - d:.1f} L{x + d:.1f} {cy:.1f} L{x:.1f} {cy + d:.1f} L{x - d:.1f} {cy:.1f} Z" '
                f'fill="{r["color"]}"/>',
            )
            parts.append(
                f'<text x="{right + 6}" y="{cy + 4:.1f}" text-anchor="start" fill="#333333">${r["value"]:.0f}mm</text>',
            )

    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="{height - 6}" text-anchor="middle" '
        f'fill="#333333" font-weight="bold">Implied Enterprise Value ($mm) — bar = P25–P75, tick = median</text>',
    )
    parts.append("</svg>")
    return "".join(parts)
