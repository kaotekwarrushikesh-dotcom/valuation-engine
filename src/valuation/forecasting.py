"""Stage 2a: turn the Stage 1 assumptions into a forecast income statement.

The chain is deliberately shallow and every link is visible:

    revenue(t) = revenue(t-1) x (1 + growth(t))
    EBITDA(t)  = revenue(t) x EBITDA margin(t)
    D&A(t)     = revenue(t) x D&A intensity
    EBIT(t)    = EBITDA(t) - D&A(t)

Two choices are worth stating, because the alternatives are common and worse.

**Margins are forecast, not absolute EBITDA.** Projecting an EBITDA figure directly hides
the operating assumption inside a single number. Forecasting the margin makes the claim
explicit: this business converts x% of sales into operating cash profit, and here is
whether that is rising, flat or falling relative to its own history.

**EBIT is derived, not forecast.** EBIT is EBITDA less D&A, so forecasting both EBITDA and
EBIT independently lets them drift apart and implies a depreciation charge nobody chose.
Deriving it means the implied D&A is always the one that was actually assumed.

D&A is tied to revenue rather than grown on its own for the same reason: a depreciation
charge that grows independently of the asset base eventually describes a company that does
not exist. Tying it to revenue keeps it anchored to the scale of operations that the
revenue forecast implies.
"""

from dataclasses import dataclass

import pandas as pd

from src.valuation.assumptions import ForecastAssumptions


@dataclass
class Forecast:
    """A forecast income statement, plus the base year it grew from."""

    ticker: str
    frame: pd.DataFrame
    base_year: int
    base_revenue: float
    base_nwc: float
    assumptions: ForecastAssumptions

    @property
    def horizon(self) -> int:
        return len(self.frame)

    @property
    def terminal_year(self) -> int:
        return int(self.frame["year"].iloc[-1])


def build_forecast(hist: pd.DataFrame, a: ForecastAssumptions, ticker: str | None = None) -> Forecast:
    """Project revenue, EBITDA, D&A and EBIT across the forecast horizon."""
    base_year = int(hist["fiscal_year"].iloc[-1])
    base_revenue = float(hist["revenue"].iloc[-1])
    base_nwc = float(hist["nwc"].iloc[-1])

    rows = []
    revenue = base_revenue

    for i in range(a.horizon):
        growth = a.revenue_growth_path[i]
        margin = a.ebitda_margin_path[i]

        prior_revenue = revenue
        revenue = prior_revenue * (1.0 + growth)

        ebitda = revenue * margin
        dep_amort = revenue * a.da_pct_revenue
        ebit = ebitda - dep_amort

        rows.append({
            "year": base_year + i + 1,
            "revenue": revenue,
            "revenue_growth": growth,
            "revenue_increment": revenue - prior_revenue,
            "ebitda": ebitda,
            "ebitda_margin": margin,
            "dep_amort": dep_amort,
            "ebit": ebit,
            "ebit_margin": ebit / revenue,
        })

    return Forecast(
        ticker=ticker or a.ticker,
        frame=pd.DataFrame(rows),
        base_year=base_year,
        base_revenue=base_revenue,
        base_nwc=base_nwc,
        assumptions=a,
    )


def forecast_vs_history(hist: pd.DataFrame, fc: Forecast) -> list[str]:
    """Sanity checks comparing the forecast against the history it came from.

    A forecast that runs at a different level from the history is not necessarily wrong,
    but it is always a claim, and a claim should be visible rather than buried in a table.
    """
    notes: list[str] = []

    hist_margin = float(hist["ebitda_margin"].tail(3).mean())
    fc_margin = float(fc.frame["ebitda_margin"].mean())
    if abs(fc_margin - hist_margin) > 0.01:
        notes.append(
            f"Forecast EBITDA margin averages {fc_margin:.1%} against a recent history of "
            f"{hist_margin:.1%}, a gap of {abs(fc_margin - hist_margin):.1%}."
        )

    hist_growth = float(hist["revenue_growth"].tail(3).mean())
    fc_growth = float(fc.frame["revenue_growth"].mean())
    if fc_growth > hist_growth + 0.02:
        notes.append(
            f"Forecast growth averages {fc_growth:.1%} against {hist_growth:.1%} recently. "
            "The forecast is more optimistic than the recent record and needs a reason."
        )

    terminal_revenue = float(fc.frame["revenue"].iloc[-1])
    multiple = terminal_revenue / fc.base_revenue
    if multiple > 2.0:
        notes.append(
            f"Revenue {multiple:.1f}x over {fc.horizon} years. Check that the market is large "
            "enough to absorb a business of that size."
        )

    # D&A well below capex means the asset base is being built faster than it is written
    # off, which is normal while investing but cannot continue forever. Capex intensity
    # already fades toward a terminal-consistent level (see assumptions.py), so this checks
    # what is left over after that fade rather than the flat historical ratio.
    da = float(fc.frame["dep_amort"].iloc[-1])
    capex = float(fc.frame["revenue"].iloc[-1] * fc.assumptions.capex_pct_revenue_path[-1])
    if capex > da * 1.5:
        notes.append(
            f"Terminal capex ({capex:,.0f}) still runs {capex/da:.1f}x terminal D&A "
            f"({da:,.0f}) even after fading toward the ROIC-consistent level. In perpetuity "
            "the two should converge, so terminal value from this base may still overstate "
            "the reinvestment the company can sustain."
        )

    return notes
