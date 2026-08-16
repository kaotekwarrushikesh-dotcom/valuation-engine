"""Stage 4: discounting, the enterprise-to-equity bridge, and a value per share.

    PV of forecast FCFF
    + PV of terminal value
    = enterprise value
    - net debt
    = equity value
    / shares outstanding
    = implied share price

**Mid-year convention is not used, and that is a choice.** Cash arrives through the year
rather than all on the last day, so discounting at the full year end understates value by
roughly half a year's discounting. The end-year convention is used because it is the plain
reading of the arithmetic and errs low. The effect is stated so it can be added back by
anyone who prefers the mid-year treatment.

**Net debt bridges enterprise value to equity value.** Enterprise value is what the whole
business is worth to everyone who funded it. Debt holders are paid first, and cash is an
asset the equity holders own, so equity value is enterprise value less debt plus cash, which
is enterprise value less net debt. A net-cash company therefore has equity worth *more* than
its business, which is correct and often surprises people.

**Terminal value share is reported prominently.** When it exceeds about three quarters of
enterprise value, the DCF is mostly an opinion about the world after the forecast, and the
five years of careful modelling in front of it are close to decoration. That is a fact about
the valuation, not a bug, and hiding it would be the real failure.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.valuation.fcff import FCFFResult
from src.valuation.terminal_value import (
    TerminalCashFlow,
    TerminalValue,
    exit_multiple,
    implied_exit_multiple,
    implied_growth,
    normalise_terminal_cash_flow,
    perpetual_growth,
)


@dataclass
class DCFResult:
    """A complete valuation, with every intermediate step retained."""

    ticker: str
    wacc: float
    terminal_growth: float
    schedule: pd.DataFrame
    pv_forecast: float
    terminal: TerminalValue
    pv_terminal: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    shares_outstanding: float
    implied_share_price: float
    current_share_price: float
    currency: str
    alternative_terminal: TerminalValue | None = None
    terminal_cash_flow: TerminalCashFlow | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def implied_share_price_floored(self) -> float:
        """The implied price, floored at zero.

        Equity cannot trade below zero under limited liability, so when net debt exceeds
        enterprise value the honest arithmetic result is negative but the honest economic
        reading is "the model considers the equity worthless on these assumptions," not a
        literal negative price. The raw, unfloored value stays on `implied_share_price` for
        anyone who wants to see how far underwater the business is; this is what should be
        shown as a price and used to compute downside, since an investor's loss is capped at
        the amount invested.
        """
        return max(self.implied_share_price, 0.0)

    @property
    def upside(self) -> float:
        if self.current_share_price <= 0:
            return float("nan")
        return self.implied_share_price_floored / self.current_share_price - 1.0

    @property
    def terminal_share(self) -> float:
        if self.enterprise_value == 0:
            return float("nan")
        return self.pv_terminal / self.enterprise_value

    def bridge_table(self) -> pd.DataFrame:
        """Enterprise value to a price per share, one line at a time."""
        rows = [
            ("PV of forecast FCFF", self.pv_forecast),
            ("PV of terminal value", self.pv_terminal),
            ("Enterprise value", self.enterprise_value),
            ("Less: net debt", -self.net_debt),
            ("Equity value", self.equity_value),
        ]
        frame = pd.DataFrame(rows, columns=["Line", f"{self.currency} m"])
        return frame


def discount_factors(years: int, wacc: float) -> np.ndarray:
    """1 / (1 + WACC)^t for t = 1..years, end-year convention."""
    t = np.arange(1, years + 1)
    return 1.0 / (1.0 + wacc) ** t


def run_dcf(
    ticker: str,
    fcff: FCFFResult,
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares_outstanding: float,
    current_share_price: float,
    currency: str = "INR",
    exit_ev_ebitda: float | None = None,
    roic: float | None = None,
) -> DCFResult:
    """Discount the forecast cash flows and bridge to a value per share."""
    f = fcff.frame
    notes: list[str] = []
    warnings: list[str] = list(fcff.warnings)

    if wacc <= terminal_growth:
        raise ValueError(
            f"{ticker}: WACC {wacc:.2%} does not exceed terminal growth {terminal_growth:.2%}; "
            "no perpetuity built on these inputs is meaningful"
        )

    n = len(f)
    factors = discount_factors(n, wacc)

    schedule = pd.DataFrame({
        "year": f["year"].astype(int),
        "fcff": f["fcff"].to_numpy(),
        "discount_factor": factors,
        "pv_fcff": f["fcff"].to_numpy() * factors,
    })
    pv_forecast = float(schedule["pv_fcff"].sum())

    terminal_ebitda = float(f["ebitda"].iloc[-1])

    # Reinvestment in the terminal year must match the growth being paid for. Carrying the
    # forecast's investment-phase capex into perpetuity charges the company for growth it is
    # never credited with, which is the largest single error an unadjusted DCF makes.
    terminal_cash_flow = normalise_terminal_cash_flow(
        terminal_nopat=float(f["nopat"].iloc[-1]),
        terminal_fcff=float(f["fcff"].iloc[-1]),
        growth=terminal_growth,
        roic=roic if roic is not None else float("nan"),
        wacc=wacc,
    )
    terminal_fcff = terminal_cash_flow.normalised_fcff
    if terminal_cash_flow.note:
        notes.append(terminal_cash_flow.note)

    primary = perpetual_growth(terminal_fcff, wacc, terminal_growth)
    alternative = (
        exit_multiple(terminal_ebitda, exit_ev_ebitda) if exit_ev_ebitda is not None else None
    )

    if not primary.usable:
        if alternative is not None and alternative.usable:
            notes.append(
                f"Perpetual growth is not usable here ({primary.note}) so the exit multiple "
                "is used as the primary terminal value."
            )
            primary, alternative = alternative, primary
        else:
            raise ValueError(f"{ticker}: no usable terminal value. {primary.note}")

    if primary.note:
        warnings.append(primary.note)

    # The terminal value sits at the end of the final forecast year, so it discounts at that
    # year's factor rather than one year further out.
    pv_terminal = primary.value * factors[-1]

    enterprise_value = pv_forecast + pv_terminal
    equity_value = enterprise_value - net_debt

    if shares_outstanding <= 0:
        raise ValueError(f"{ticker}: share count is not positive")
    implied_share_price = equity_value / shares_outstanding

    if net_debt < 0:
        notes.append(
            f"Net cash of {abs(net_debt):,.0f}, so equity value exceeds enterprise value. "
            "The cash belongs to shareholders on top of the business itself."
        )

    result = DCFResult(
        ticker=ticker, wacc=wacc, terminal_growth=terminal_growth, schedule=schedule,
        pv_forecast=pv_forecast, terminal=primary, pv_terminal=pv_terminal,
        enterprise_value=enterprise_value, net_debt=net_debt, equity_value=equity_value,
        shares_outstanding=shares_outstanding, implied_share_price=implied_share_price,
        current_share_price=current_share_price, currency=currency,
        alternative_terminal=alternative, terminal_cash_flow=terminal_cash_flow,
        notes=notes, warnings=warnings,
    )

    share = result.terminal_share
    if share > 0.75:
        warnings.append(
            f"The terminal value is {share:.0%} of enterprise value, so this valuation is "
            "mostly a claim about the world after the forecast rather than about the "
            "forecast itself."
        )
    elif share < 0.35:
        notes.append(
            f"The terminal value is only {share:.0%} of enterprise value, which is unusually "
            "low and means the answer rests mainly on the explicit forecast."
        )

    if equity_value <= 0:
        warnings.append(
            "Equity value is not positive: the business does not cover its net debt on these "
            "assumptions."
        )

    return result


def cross_checks(result: DCFResult, terminal_ebitda: float) -> list[str]:
    """Independent reads on whether the terminal assumption is reasonable."""
    checks: list[str] = []

    multiple = implied_exit_multiple(result.terminal.value, terminal_ebitda)
    if not np.isnan(multiple):
        checks.append(
            f"The terminal value implies an exit multiple of {multiple:.1f}x EV/EBITDA. "
            + ("That is high, so the growth assumption is doing a lot of work."
               if multiple > 20 else
               "That is low, which makes the terminal assumption conservative."
               if multiple < 6 else
               "That sits in a normal range for a mature business.")
        )

    if result.alternative_terminal is not None and result.alternative_terminal.usable:
        gap = result.alternative_terminal.value / result.terminal.value - 1.0
        checks.append(
            f"The two terminal methods differ by {gap:+.0%}. "
            + ("They broadly agree, so the answer does not hinge on the choice."
               if abs(gap) < 0.20 else
               "They disagree materially, which means the growth assumption and what the "
               "market currently pays are telling different stories.")
        )

    terminal_fcff = (result.terminal_cash_flow.normalised_fcff
                     if result.terminal_cash_flow else float(result.schedule["fcff"].iloc[-1]))
    g = implied_growth(result.terminal.value, terminal_fcff, result.wacc)
    if not np.isnan(g):
        checks.append(f"The terminal value implies perpetual growth of {g:.2%}.")

    return checks


def reverse_dcf(
    fcff: FCFFResult,
    market_cap: float,
    net_debt: float,
    terminal_growth: float,
    roic: float,
    low: float = 0.04,
    high: float = 0.40,
) -> float:
    """The discount rate at which the model would agree with today's share price.

    When a valuation disagrees with the market, there are two possible readings and the
    difference matters: either the market is wrong, or the model's assumptions are. A
    reverse DCF refuses to take sides. It solves for the cost of capital that makes the
    model's equity value equal the market capitalisation, which converts a verdict into a
    question that can be argued about with evidence.

    If the answer comes back at 11% against a 14% modelled WACC, the market is not being
    irrational, it is applying a lower required return than this model assumes, and the
    argument is about the equity risk premium rather than about the company. That is a far
    more useful place for a disagreement to sit.

    Solved by bisection because equity value falls monotonically as the discount rate rises,
    which makes the root unique wherever one exists.
    """
    target = market_cap + net_debt  # the enterprise value the market implies

    def enterprise_value_at(rate: float) -> float:
        if rate <= terminal_growth:
            return float("inf")
        f = fcff.frame
        factors = discount_factors(len(f), rate)
        pv = float((f["fcff"].to_numpy() * factors).sum())

        tcf = normalise_terminal_cash_flow(
            terminal_nopat=float(f["nopat"].iloc[-1]),
            terminal_fcff=float(f["fcff"].iloc[-1]),
            growth=terminal_growth, roic=roic, wacc=rate,
        )
        tv = perpetual_growth(tcf.normalised_fcff, rate, terminal_growth)
        if not tv.usable:
            return float("nan")
        return pv + tv.value * factors[-1]

    lo, hi = max(low, terminal_growth + 0.001), high
    if enterprise_value_at(hi) > target:
        return float("nan")  # even a punitive rate cannot justify the price

    for _ in range(200):
        mid = (lo + hi) / 2.0
        value = enterprise_value_at(mid)
        if np.isnan(value):
            return float("nan")
        if value > target:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0
