"""Stage 6b: sensitivity tables.

Two grids, because they test different things. Both are generated from the model, not typed
in by hand, so a table can always be regenerated from the assumptions behind it.

**WACC x terminal growth.** These two sit in the terminal-value formula together
(`FCFF x (1+g) / (WACC-g)`), and the spread between them, not either one alone, drives the
result: a company discounted at 12% growing at 4% forever is valued very differently from
one discounted at 11% growing at 5%, even though both spreads look similar at a glance. This
grid is cheap to build, because the explicit-period cash flows do not depend on WACC or
terminal growth; only the terminal value and the discounting change per cell, so the
forecast is built once and reused across the whole table.

**Revenue growth x EBITDA margin.** These drive the explicit-period cash flows themselves,
not just the terminal value, so each cell requires a full rebuild: a new assumption set, a
new forecast, a new FCFF, and a new discounted value. This grid is the expensive one and is
kept smaller as a result.

Every cell is an implied share price, so the tables read directly against the current price
without a second lookup. Cells are the raw arithmetic result, not floored at zero the way a
single headline valuation is: a grid exists to show the gradient and magnitude of the
sensitivity, and flattening a whole region of stressed cells to zero would hide exactly the
shape the table is meant to reveal.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.valuation import assumptions as asmp
from src.valuation.dcf import run_dcf
from src.valuation.fcff import FCFFResult, build_fcff
from src.valuation.forecasting import build_forecast

# Default grids: +/- 2 points around the base case in both dimensions, which is wide enough
# to show the shape of the sensitivity without so wide that most cells are implausible.
DEFAULT_WACC_STEPS = 5
DEFAULT_GROWTH_STEPS = 4


def wacc_terminal_growth_grid(
    fcff: FCFFResult, base_wacc: float, base_terminal_growth: float, roic: float,
    net_debt: float, shares_outstanding: float,
    wacc_steps: int = DEFAULT_WACC_STEPS, growth_steps: int = DEFAULT_GROWTH_STEPS,
    wacc_step_size: float = 0.005, growth_step_size: float = 0.005,
) -> pd.DataFrame:
    """Implied share price across a grid of discount rates and terminal growth rates.

    The forecast cash flows are fixed and passed in once; only the terminal value and
    discounting are recomputed per cell, which is what makes this grid cheap.
    """
    wacc_lo = base_wacc - (wacc_steps // 2) * wacc_step_size
    waccs = [round(wacc_lo + i * wacc_step_size, 5) for i in range(wacc_steps)]

    growth_lo = base_terminal_growth - (growth_steps // 2) * growth_step_size
    growths = [round(growth_lo + i * growth_step_size, 5) for i in range(growth_steps)]

    rows = []
    for wacc in waccs:
        row = {"wacc": wacc}
        for g in growths:
            try:
                result = run_dcf(
                    ticker="grid", fcff=fcff, wacc=wacc, terminal_growth=g,
                    net_debt=net_debt, shares_outstanding=shares_outstanding,
                    current_share_price=0.0, roic=roic,
                )
                row[f"g={g:.2%}"] = result.implied_share_price
            except ValueError:
                row[f"g={g:.2%}"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows).set_index("wacc")


def growth_margin_grid(
    hist: pd.DataFrame, analysis: dict, ticker: str,
    base: asmp.ForecastAssumptions, base_wacc: float,
    net_debt: float, shares_outstanding: float, currency: str,
    nominal_gdp_growth: float, inflation: float, horizon: int,
    growth_steps: int = DEFAULT_GROWTH_STEPS, margin_steps: int = DEFAULT_GROWTH_STEPS,
    growth_step_size: float = 0.02, margin_step_size: float = 0.02,
) -> pd.DataFrame:
    """Implied share price across a grid of starting revenue growth and EBITDA margin.

    Each cell rebuilds the whole forecast, because these drivers move the explicit-period
    cash flows, not only the terminal value. WACC is held at the base case throughout, so
    the grid isolates the operating drivers rather than mixing them with the discount rate.
    """
    base_growth = base.revenue_growth_path[0]
    base_margin = base.ebitda_margin_path[0]

    growth_lo = base_growth - (growth_steps // 2) * growth_step_size
    growths = [round(growth_lo + i * growth_step_size, 5) for i in range(growth_steps)]

    margin_lo = base_margin - (margin_steps // 2) * margin_step_size
    margins = [round(max(margin_lo + i * margin_step_size, 0.01), 5) for i in range(margin_steps)]

    rows = []
    for g in growths:
        row = {"revenue_growth": g}
        for m in margins:
            overrides = {"revenue_growth_start": max(min(g, 0.60), -0.15), "ebitda_margin": m}
            try:
                a = asmp.derive(hist, analysis, ticker, horizon=horizon, overrides=overrides,
                                nominal_gdp_growth=nominal_gdp_growth, inflation=inflation)
                fc = build_forecast(hist, a, ticker)
                fcff = build_fcff(fc)
                result = run_dcf(
                    ticker=ticker, fcff=fcff, wacc=base_wacc, terminal_growth=a.terminal_growth,
                    net_debt=net_debt, shares_outstanding=shares_outstanding,
                    current_share_price=0.0, currency=currency, roic=a.terminal_roic,
                )
                row[f"margin={m:.1%}"] = result.implied_share_price
            except ValueError:
                row[f"margin={m:.1%}"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows).set_index("revenue_growth")


def driver_sensitivity(
    hist: pd.DataFrame, analysis: dict, ticker: str,
    base: asmp.ForecastAssumptions, base_wacc: float, base_price: float,
    net_debt: float, shares_outstanding: float, currency: str,
    nominal_gdp_growth: float, inflation: float, horizon: int,
    bump: float = 0.01,
) -> pd.DataFrame:
    """The impact of a one-point bump to each driver, holding everything else fixed.

    Answers "what assumption is the valuation most sensitive to" directly, by bumping one
    driver at a time and reporting the resulting change in implied equity value. Drivers are
    bumped in the direction that plausibly improves them (growth and margin up, WACC and
    capex down), so every row reads as "this much better on this one input is worth this
    much", which is the form the question is actually asked in.
    """
    def value_at(overrides: dict, wacc: float) -> float:
        a = asmp.derive(hist, analysis, ticker, horizon=horizon, overrides=overrides,
                        nominal_gdp_growth=nominal_gdp_growth, inflation=inflation)
        fc = build_forecast(hist, a, ticker)
        fcff = build_fcff(fc)
        result = run_dcf(ticker=ticker, fcff=fcff, wacc=wacc, terminal_growth=a.terminal_growth,
                         net_debt=net_debt, shares_outstanding=shares_outstanding,
                         current_share_price=0.0, currency=currency, roic=a.terminal_roic)
        return result.implied_share_price

    base_growth = base.revenue_growth_path[0]
    base_margin = base.ebitda_margin_path[0]
    base_capex = base.capex_pct_revenue_path[0]

    drivers = [
        ("Revenue growth", {"revenue_growth_start": base_growth + bump}, base_wacc, f"+{bump:.0%}pt"),
        ("EBITDA margin", {"ebitda_margin": base_margin + bump}, base_wacc, f"+{bump:.0%}pt"),
        ("WACC", {}, base_wacc - bump, f"-{bump:.0%}pt"),
        ("Terminal growth", {"terminal_growth": base.terminal_growth + bump}, base_wacc, f"+{bump:.0%}pt"),
        ("Capex / revenue", {"capex_pct_revenue": max(base_capex - bump, 0.0)}, base_wacc, f"-{bump:.0%}pt"),
    ]

    rows = []
    for label, overrides, wacc, direction in drivers:
        try:
            new_price = value_at(overrides, wacc)
            impact = new_price / base_price - 1.0 if base_price > 0 else float("nan")
        except ValueError:
            new_price, impact = float("nan"), float("nan")
        rows.append({"driver": label, "shift": direction, "implied_price": new_price,
                    "impact_pct": impact})

    df = pd.DataFrame(rows)
    return df.reindex(df["impact_pct"].abs().sort_values(ascending=False).index).reset_index(drop=True)
