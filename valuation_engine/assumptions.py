"""Derive forecast assumptions from history, with the reasoning attached to each one.

Every assumption here is produced by a stated rule applied to the company's own history,
and carries the rationale that justifies it. Two consequences follow, and both are
deliberate:

  A number you cannot explain is not usable. Each assumption ships with the evidence
  behind it, so "why 6.5%?" has an answer that is not "it looked about right".

  The engine must not be steerable toward a wanted answer. The rules run before anyone
  sees the resulting share price. Overrides are allowed but are recorded as overrides,
  so a hand-set assumption can never be mistaken for a derived one.

The central mechanism is the growth fade. High growth decays: competition arrives, the
base gets larger, and the addressable market fills up. A forecast that holds recent growth
flat for five years and then drops to a terminal rate implies a cliff no business
experiences. So revenue growth is faded geometrically from the starting rate toward the
terminal rate across the forecast horizon, which is both smoother and more conservative.

The terminal growth rate is capped at long-run nominal GDP. A company growing faster than
the economy forever eventually becomes the economy, so any terminal rate above that is an
arithmetic impossibility rather than an optimistic view.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from valuation_engine.terminal_value import estimate_roic, terminal_consistent_capex_ratio

# Long-run nominal GDP growth: roughly 2% real plus a 2% inflation target. This is the
# ceiling on any terminal growth rate, not a forecast of it.
NOMINAL_GDP_GROWTH = 0.04

# Growth above this is treated as exceptional and faded harder, since sustaining it for a
# full forecast horizon is the exception rather than the rule.
EXCEPTIONAL_GROWTH = 0.25


@dataclass
class Assumption:
    """One forecast input, its value, where it came from, and how confident we are."""

    name: str
    value: float
    rationale: str
    source: str  # "derived" or "override"
    confidence: str  # high | medium | low

    def __str__(self) -> str:
        tag = "" if self.source == "derived" else "  [OVERRIDE]"
        return f"{self.name}: {self.value:.4g}  ({self.confidence} confidence){tag}\n    {self.rationale}"


@dataclass
class ForecastAssumptions:
    """The complete assumption set for one company's forecast."""

    ticker: str
    horizon: int
    revenue_growth_path: list[float]
    ebitda_margin_path: list[float]
    da_pct_revenue: float
    capex_pct_revenue_path: list[float]
    nwc_pct_revenue: float
    tax_rate: float
    terminal_growth: float
    terminal_roic: float
    detail: dict[str, Assumption] = field(default_factory=dict)

    @property
    def capex_pct_revenue(self) -> float:
        """The starting-year capex intensity, for callers that want one number.

        The forecast itself uses the full path, which fades toward the terminal-consistent
        level; this is the level history actually supports, kept for anything that reports
        a single figure (a summary table, an override default) rather than the trajectory.
        """
        return self.capex_pct_revenue_path[0]

    def as_frame(self, last_actual_year: int) -> pd.DataFrame:
        years = [last_actual_year + i + 1 for i in range(self.horizon)]
        return pd.DataFrame(
            {
                "year": years,
                "revenue_growth": self.revenue_growth_path,
                "ebitda_margin": self.ebitda_margin_path,
                "da_pct_revenue": self.da_pct_revenue,
                "capex_pct_revenue": self.capex_pct_revenue_path,
                "nwc_pct_revenue": self.nwc_pct_revenue,
                "tax_rate": self.tax_rate,
            }
        )


def fade_path(start: float, end: float, periods: int) -> list[float]:
    """Fade geometrically from `start` toward `end` across `periods` years.

    Geometric rather than linear: growth decays proportionally, so the early years hold
    more of the starting rate and the approach to the terminal rate flattens out. A linear
    fade would cut the first year too hard and the last year not hard enough.
    """
    if periods <= 0:
        return []
    if periods == 1:
        return [end]

    gap = start - end
    return [end + gap * ((periods - 1 - i) / (periods - 1)) ** 1.5 for i in range(periods)]


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def derive(
    hist: pd.DataFrame,
    analysis: dict,
    ticker: str,
    horizon: int = 5,
    overrides: dict[str, float] | None = None,
    nominal_gdp_growth: float = NOMINAL_GDP_GROWTH,
    inflation: float = 0.02,
) -> ForecastAssumptions:
    """Build the assumption set from the company's own history.

    `nominal_gdp_growth` is the terminal-growth ceiling and must match the currency of the
    cash flows. India's nominal GDP grows far faster than the US, so a 4% ceiling applied
    to rupee cash flows would understate terminal value for reasons of geography rather
    than economics.
    """
    overrides = overrides or {}
    detail: dict[str, Assumption] = {}

    def record(name: str, derived_value: float, rationale: str, confidence: str) -> float:
        if name in overrides:
            value = float(overrides[name])
            detail[name] = Assumption(
                name, value,
                f"Set by hand. Derived value would have been {derived_value:.4g}. Original basis: {rationale}",
                "override", confidence,
            )
            return value
        detail[name] = Assumption(name, derived_value, rationale, "derived", confidence)
        return derived_value

    growth_trend = analysis["trends"]["revenue_growth"]
    levels = analysis["levels"]
    g = analysis["growth"]

    # --- Starting revenue growth -----------------------------------------------------
    # The recent three-year CAGR, not the latest single year, which is noisy. Exceptional
    # growth is damped on the way in, before the fade even starts.
    recent = g["revenue_cagr_recent"]
    full = g["revenue_cagr"]
    base_start = recent if not np.isnan(recent) else full

    if np.isnan(base_start):
        base_start = nominal_gdp_growth
        start_note = "No usable growth history, so the forecast starts at long-run nominal GDP."
        start_conf = "low"
    elif base_start > EXCEPTIONAL_GROWTH:
        base_start = EXCEPTIONAL_GROWTH + (base_start - EXCEPTIONAL_GROWTH) * 0.5
        start_note = (
            f"Recent growth of {recent:.1%} is exceptional. Half the excess above "
            f"{EXCEPTIONAL_GROWTH:.0%} is removed at the start, because sustaining that rate "
            "over a full horizon is the exception, and the fade then does the rest."
        )
        start_conf = "low"
    elif abs(recent - full) < 1e-9:
        # Short histories make the recent window and the full period the same span.
        start_note = (
            f"Revenue CAGR of {recent:.1%}. The history is short enough that the recent "
            "window and the full period are the same span, so there is no longer-run rate "
            "to compare against and the forecast rests on fewer observations than it should."
        )
        start_conf = "low"
    else:
        start_note = (
            f"Three-year revenue CAGR of {recent:.1%}, preferred over the full-period "
            f"{full:.1%} because it reflects the current state of the business, and over the "
            "latest single year because one year is noise."
        )
        start_conf = "medium" if abs(recent - full) < 0.10 else "low"

    start_growth = record("revenue_growth_start", round(_clamp(base_start, -0.15, 0.60), 4),
                          start_note, start_conf)

    # --- Terminal growth -------------------------------------------------------------
    # Two ceilings, and the lower one binds.
    #
    #   Long-run nominal GDP. A company growing faster than the economy forever eventually
    #   becomes the economy, so this is arithmetic, not conservatism.
    #
    #   The company's own current growth. If a business is growing at 2%, assuming it
    #   re-accelerates to 4% and holds that in perpetuity is an assumption that has to be
    #   argued for. Without an argument, the forecast should not drift upward into the
    #   terminal value, which is where most of the value sits.
    mature_signal = growth_trend.classification in ("decelerating", "contracting")
    gdp_ceiling = nominal_gdp_growth * (0.625 if mature_signal else 1.0)
    unfloored = min(gdp_ceiling, start_growth)
    terminal_basis = max(unfloored, inflation)

    if terminal_basis > unfloored:
        terminal_note = (
            f"Floored at {inflation:.1%} inflation. Recent growth of {start_growth:.1%} would "
            f"imply the company grows below inflation in perpetuity, which means shrinking in "
            f"real terms forever and eventually to nothing. A few years of weak growth is "
            "evidence about the cycle, not about the next century, and treating a cyclical "
            "window as a structural rate is not conservatism but a different and much stronger "
            "claim. The floor is what stops the terminal value being set by whichever part of "
            "the cycle the last four years happened to fall in."
        )
    elif terminal_basis == start_growth and start_growth < gdp_ceiling:
        terminal_note = (
            f"Set to current growth of {start_growth:.1%}, below the {gdp_ceiling:.1%} GDP ceiling. "
            "The company is growing more slowly than the economy, and assuming it re-accelerates "
            "in perpetuity would push value into the terminal period on no evidence. Growth is "
            "therefore held flat rather than faded upward."
        )
    else:
        terminal_note = (
            f"Capped at long-run nominal GDP of {nominal_gdp_growth:.1%}"
            + (f", reduced to {gdp_ceiling:.1%} because growth is fading. " if mature_signal else ". ")
            + "No company can outgrow the economy in perpetuity, so this is a ceiling rather than a view."
        )

    terminal_growth = record("terminal_growth", round(terminal_basis, 4), terminal_note, "medium")

    growth_path = [round(x, 5) for x in fade_path(start_growth, terminal_growth, horizon)]
    path_str = ", ".join(f"{x:.1%}" for x in growth_path)
    if abs(start_growth - terminal_growth) < 1e-9:
        path_note = (
            f"Held flat at {start_growth:.1%} for all {horizon} years: {path_str}. There is nothing "
            "to fade, because current growth is already at or below the terminal rate the company "
            "can sustain."
        )
    else:
        path_note = (
            f"Fades geometrically from {start_growth:.1%} to {terminal_growth:.1%} over {horizon} "
            f"years: {path_str}. Holding growth flat and then dropping straight to the terminal rate "
            "would imply a cliff that no business experiences."
        )

    detail["revenue_growth_path"] = Assumption(
        "revenue_growth_path", start_growth, path_note, "derived", start_conf
    )

    # --- EBITDA margin ---------------------------------------------------------------
    # Held at the recent average. Assuming expansion is how a DCF gets talked into the
    # answer someone wanted, so expansion has to be argued for, not defaulted to.
    margin_trend = analysis["trends"]["ebitda_margin"]
    recent_margin = levels["ebitda_margin_recent"]
    margin_note = (
        f"Three-year average EBITDA margin of {recent_margin:.1%}, held flat. "
        f"History says the margin is {margin_trend.classification}. "
    )
    if margin_trend.classification == "expanding":
        margin_note += (
            "It has been expanding, but extending expansion into the forecast is the most "
            "common way a DCF is quietly steered upward, so the base case does not."
        )
    elif margin_trend.classification == "compressing":
        margin_note += (
            "It has been compressing. Holding it flat is already generous relative to the "
            "trend; the bear case carries the compression through."
        )
    else:
        margin_note += "A stable history makes the recent average the natural forecast level."

    ebitda_margin = record("ebitda_margin", round(recent_margin, 4), margin_note,
                           "high" if margin_trend.classification == "stable" else "medium")
    margin_path = [ebitda_margin] * horizon

    # --- Tax -------------------------------------------------------------------------
    # Computed before capex because the reinvestment-consistency check below needs it to
    # estimate return on invested capital.
    median_tax = levels["effective_tax_rate_median"]
    tax_floor, tax_cap = 0.10, 0.35
    tax_rate = record(
        "tax_rate", round(_clamp(median_tax, tax_floor, tax_cap), 4),
        (
            f"Median effective tax rate of {median_tax:.1%}, bounded to "
            f"[{tax_floor:.0%}, {tax_cap:.0%}]. The effective rate is used because it is what the "
            "company actually pays. The floor exists because temporary credits and one-off "
            "settlements should not be projected into perpetuity."
        ),
        "high" if tax_floor <= median_tax <= tax_cap else "low",
    )

    # --- Capital and working capital intensity ---------------------------------------
    # Medians, not means: one acquisition-heavy or one deferred-spending year should not
    # set the run rate for five forecast years.
    da_pct = record(
        "da_pct_revenue", round(levels["da_pct_revenue_median"], 4),
        (
            f"Median D&A of {levels['da_pct_revenue_median']:.1%} of revenue. D&A is forecast as a "
            "share of revenue rather than grown independently, so it stays tied to the asset base "
            "that revenue implies. The median avoids letting one impairment set the run rate."
        ),
        "high",
    )

    capex_trend = analysis["trends"]["capex_pct_revenue"]
    capex_start = record(
        "capex_pct_revenue", round(levels["capex_pct_revenue_median"], 4),
        (
            f"Median capex of {levels['capex_pct_revenue_median']:.1%} of revenue, where capital "
            f"intensity is {capex_trend.classification}. Capex is the assumption most often set too "
            "low, because it is the easiest way to manufacture free cash flow."
        ),
        "medium" if capex_trend.classification == "stable" else "low",
    )

    # Capex fades toward a terminal-consistent level, the same way revenue growth fades
    # toward terminal growth. Holding capex flat at its historical share of revenue while
    # growth fades down underneath it charges the company for reinvestment it is never
    # credited with in the explicit years, not only in the terminal year: it is the same
    # incoherence the terminal-year normalisation exists to fix, just left in the five years
    # in front of it. Reliance is the clear case: its forecast reinvested 145% of NOPAT for
    # 6.4% growth every explicit year, a gap the terminal fix alone did not touch.
    terminal_roic = estimate_roic(hist, tax_rate)
    capex_terminal, capex_path_note = terminal_consistent_capex_ratio(
        ebitda_margin=ebitda_margin, da_pct_revenue=da_pct,
        nwc_pct_revenue=float(hist["nwc_pct_revenue"].tail(3).mean()),
        tax_rate=tax_rate, growth=terminal_growth, roic=terminal_roic,
    )

    if np.isnan(capex_terminal):
        capex_path = [capex_start] * horizon
        detail["capex_pct_revenue_path"] = Assumption(
            "capex_pct_revenue_path", capex_start, capex_path_note, "derived",
            "low",
        )
    else:
        capex_path = [round(x, 5) for x in fade_path(capex_start, capex_terminal, horizon)]
        path_str = ", ".join(f"{x:.1%}" for x in capex_path)
        detail["capex_pct_revenue_path"] = Assumption(
            "capex_pct_revenue_path", capex_start,
            f"{capex_path_note} Path: {path_str}.",
            "derived", "medium" if capex_trend.classification == "stable" else "low",
        )

    # Working capital is the one ratio that must anchor on the recent level rather than the
    # full-period median. The forecast bridges from the last actual NWC balance, so the
    # first year's change in working capital is the gap between that balance and whatever
    # ratio is assumed. If the ratio comes from a median spanning a structural shift, that
    # gap is a large phantom cash flow that reflects the choice of average, not the
    # business. Apple is the clear case: NWC moved from +5% of revenue to -10% over the
    # decade, and the median of -2.5% describes no state the company was ever in.
    nwc_recent = float(hist["nwc_pct_revenue"].tail(3).mean())
    nwc_median = levels["nwc_pct_revenue_median"]
    drift = abs(nwc_recent - nwc_median)

    nwc_pct = record(
        "nwc_pct_revenue", round(nwc_recent, 4),
        (
            f"Three-year average non-cash working capital of {nwc_recent:.1%} of revenue"
            + (
                f", against a full-period median of {nwc_median:.1%}. The recent level is used because "
                "the forecast bridges from the last reported balance, and a ratio drawn from a period "
                "the business has moved away from would create a first-year cash flow that is an "
                "artefact of averaging."
                if drift > 0.02
                else f", consistent with the full-period median of {nwc_median:.1%}."
            )
            + " Holding the ratio constant means growth consumes cash in proportion to the revenue it "
            "adds. A negative ratio means suppliers and customers fund the business, so growth "
            "releases cash rather than consuming it."
        ),
        "medium" if drift <= 0.02 else "low",
    )

    return ForecastAssumptions(
        ticker=ticker,
        horizon=horizon,
        revenue_growth_path=growth_path,
        ebitda_margin_path=margin_path,
        da_pct_revenue=da_pct,
        capex_pct_revenue_path=capex_path,
        nwc_pct_revenue=nwc_pct,
        tax_rate=tax_rate,
        terminal_growth=terminal_growth,
        terminal_roic=terminal_roic,
        detail=detail,
    )
