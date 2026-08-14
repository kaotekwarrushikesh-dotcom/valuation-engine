"""Historical trend analysis and its financial interpretation.

The point of this stage is not the metrics. It is the read on the business that the
forecast has to be consistent with. A forecast of accelerating growth for a company whose
growth has decelerated for five years is not wrong because the arithmetic fails, it is
wrong because nothing in the history supports it. So each driver is classified, and the
classification is what the assumption engine consumes.

Classification uses the first half of the available history against the second half rather
than first-year against last-year. Endpoint comparisons are hostage to a single unusual
year at either end, which for these companies means the 2020 and 2021 distortions land
directly on the conclusion.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Trend:
    """One driver's direction, with the numbers behind the call."""

    metric: str
    label: str
    early: float
    late: float
    change: float
    classification: str
    narrative: str


def cagr(first: float, last: float, years: int) -> float:
    """Compound annual growth rate. Undefined if either endpoint is non-positive."""
    if years <= 0 or pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0:
        return float("nan")
    return (last / first) ** (1.0 / years) - 1.0


def _halves(series: pd.Series) -> tuple[float, float]:
    """Mean of the earlier half and the later half of the non-null values."""
    s = series.dropna()
    if len(s) < 2:
        return float("nan"), float("nan")
    mid = len(s) // 2
    return float(s.iloc[:mid].mean()), float(s.iloc[mid:].mean())


def classify_growth(hist: pd.DataFrame) -> Trend:
    """Is revenue growth accelerating, holding, or fading?"""
    early, late = _halves(hist["revenue_growth"])
    change = late - early

    if pd.isna(change):
        classification, narrative = "unknown", "Too few years of revenue growth to classify."
    elif late < 0:
        classification = "contracting"
        narrative = "Revenue is shrinking, so the forecast must start from decline, not growth."
    elif change > 0.03:
        classification = "accelerating"
        narrative = (
            "Growth has accelerated. Acceleration is rarely permanent, so a forecast should "
            "fade it toward a sustainable rate rather than extend it."
        )
    elif change < -0.03:
        classification = "decelerating"
        narrative = (
            "Growth has decelerated. The forecast should continue fading rather than assume "
            "a recovery the history does not support."
        )
    else:
        classification = "stable"
        narrative = "Growth has been broadly stable, which makes a modest fade the neutral base case."

    return Trend(
        "revenue_growth", "Revenue growth", early, late, change, classification, narrative
    )


def classify_margin(hist: pd.DataFrame, column: str, label: str) -> Trend:
    """Is the margin expanding, holding, or compressing?"""
    early, late = _halves(hist[column])
    change = late - early

    if pd.isna(change):
        classification, narrative = "unknown", f"Too few years of {label.lower()} to classify."
    elif change > 0.02:
        classification = "expanding"
        narrative = (
            f"{label} has expanded by {change:.1%}. Holding the recent level flat is the "
            "defensible base case; assuming further expansion needs a stated reason."
        )
    elif change < -0.02:
        classification = "compressing"
        narrative = (
            f"{label} has compressed by {abs(change):.1%}, which is a direct drag on value "
            "and should not be assumed away in the base case."
        )
    else:
        classification = "stable"
        narrative = f"{label} has been stable, so the recent average is a reasonable forecast level."

    return Trend(column, label, early, late, change, classification, narrative)


def classify_cash_conversion(hist: pd.DataFrame) -> Trend:
    """Is free cash flow improving or deteriorating relative to sales?"""
    early, late = _halves(hist["fcf_margin"])
    change = late - early

    if pd.isna(change):
        classification, narrative = "unknown", "Too few years of free cash flow to classify."
    elif change > 0.02:
        classification = "improving"
        narrative = "Cash conversion has improved, which supports the quality of reported earnings."
    elif change < -0.02:
        classification = "deteriorating"
        narrative = (
            "Cash conversion has deteriorated. Earnings are turning into cash less reliably, "
            "which is a warning worth carrying into the forecast."
        )
    else:
        classification = "stable"
        narrative = "Cash conversion has been stable."

    return Trend("fcf_margin", "FCF margin", early, late, change, classification, narrative)


def capital_intensity(hist: pd.DataFrame) -> Trend:
    """Is the business getting more or less capital hungry?"""
    early, late = _halves(hist["capex_pct_revenue"])
    change = late - early

    if pd.isna(change):
        classification, narrative = "unknown", "Capex is not reported in enough years to classify."
    elif change > 0.015:
        classification = "rising"
        narrative = (
            "Capital intensity is rising, which consumes free cash flow even when margins hold."
        )
    elif change < -0.015:
        classification = "falling"
        narrative = "Capital intensity is falling, which frees up cash at any given margin."
    else:
        classification = "stable"
        narrative = "Capital intensity has been stable."

    return Trend("capex_pct_revenue", "Capex / revenue", early, late, change, classification, narrative)


def growth_rates(hist: pd.DataFrame) -> dict[str, float]:
    """Full-period and recent CAGRs for the headline lines."""
    years = int(hist["fiscal_year"].iloc[-1] - hist["fiscal_year"].iloc[0])
    out: dict[str, float] = {}

    for key, col in [("revenue", "revenue"), ("ebitda", "ebitda"), ("ebit", "ebit"),
                     ("net_income", "net_income"), ("fcf", "fcf_levered")]:
        s = hist[col].dropna()
        if len(s) >= 2:
            span = int(hist["fiscal_year"].iloc[s.index[-1]] - hist["fiscal_year"].iloc[s.index[0]])
            out[f"{key}_cagr"] = cagr(s.iloc[0], s.iloc[-1], span)
        else:
            out[f"{key}_cagr"] = float("nan")

    recent = hist.tail(4)
    r = recent["revenue"].dropna()
    out["revenue_cagr_recent"] = cagr(r.iloc[0], r.iloc[-1], len(r) - 1) if len(r) >= 2 else float("nan")
    out["full_period_years"] = years
    return out


def analyse(hist: pd.DataFrame) -> dict:
    """Run the full historical read."""
    trends = [
        classify_growth(hist),
        classify_margin(hist, "ebitda_margin", "EBITDA margin"),
        classify_margin(hist, "ebit_margin", "EBIT margin"),
        classify_cash_conversion(hist),
        capital_intensity(hist),
    ]

    levels = {
        "ebitda_margin_recent": float(hist["ebitda_margin"].tail(3).mean()),
        "ebit_margin_recent": float(hist["ebit_margin"].tail(3).mean()),
        "da_pct_revenue_median": float(hist["da_pct_revenue"].median()),
        "capex_pct_revenue_median": float(hist["capex_pct_revenue"].median()),
        "nwc_pct_revenue_median": float(hist["nwc_pct_revenue"].median()),
        "effective_tax_rate_median": float(hist["effective_tax_rate"].median()),
    }

    return {"trends": {t.metric: t for t in trends}, "growth": growth_rates(hist), "levels": levels}


def summary_lines(analysis: dict) -> list[str]:
    """The historical read as sentences, which is what goes in front of a reader."""
    g = analysis["growth"]
    lines = [
        f"Revenue compounded at {g['revenue_cagr']:.1%} over {g['full_period_years']} years, "
        f"and at {g['revenue_cagr_recent']:.1%} over the last three.",
    ]
    if not np.isnan(g["ebitda_cagr"]):
        lines.append(f"EBITDA compounded at {g['ebitda_cagr']:.1%}, against revenue at {g['revenue_cagr']:.1%}: "
                     + ("operating leverage is positive." if g["ebitda_cagr"] > g["revenue_cagr"]
                        else "profit has grown more slowly than sales."))
    for t in analysis["trends"].values():
        lines.append(f"{t.label}: {t.classification}. {t.narrative}")
    return lines
