"""Terminal value: everything the business is worth after the forecast ends.

This is usually the largest single number in a DCF, often more than two thirds of
enterprise value, which is uncomfortable because it rests on the assumptions we know least
about. Both standard methods are computed so they can be compared, and the report says how
much of the answer depends on it.

**Perpetual growth (Gordon growth).**

    TV = FCFF(terminal) x (1 + g) / (WACC - g)

The next year's cash flow capitalised at the spread between the discount rate and the
growth rate. Its weakness is that the spread sits in the denominator: at a 12% WACC, moving
g from 3% to 5% lifts the terminal value by around 29%, so an assumption nobody can verify
swings the answer. It also requires FCFF to be positive and g strictly below WACC.

**Exit multiple.**

    TV = EBITDA(terminal) x multiple

Values the business at what a comparable company trades for. It grounds the answer in the
market rather than in a growth assumption, but it imports whatever the market is currently
paying, so a DCF cross-checked against a bubble multiple is not independent of the bubble.

The two disagreeing is information, not a defect. A large gap means the growth assumption
and the market's current pricing are telling different stories, and that is worth saying.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A company earning less on new capital than its cost of capital destroys value by growing.
# Where measured ROIC is implausible the estimate is bounded rather than propagated.
MIN_TERMINAL_ROIC = 0.05
MAX_TERMINAL_ROIC = 0.60


@dataclass(frozen=True)
class TerminalCashFlow:
    """The terminal cash flow after reinvestment is made consistent with growth."""

    raw_fcff: float
    normalised_fcff: float
    nopat: float
    roic: float
    reinvestment_rate: float
    note: str


def estimate_roic(hist: pd.DataFrame, tax_rate: float) -> float:
    """Return on invested capital, from the company's own history.

        ROIC = NOPAT / (total debt + equity - cash)

    Invested capital is what the providers of capital have actually put into the operating
    business, which is why cash is removed: idle cash earns a treasury return, not the
    business return, and leaving it in understates how productively the company invests.

    The median across available years is used, since one unusual year would otherwise set
    the reinvestment the terminal value depends on.
    """
    invested = hist["total_debt"] + hist["equity"] - hist["cash"]
    nopat = hist["ebit"] * (1.0 - tax_rate)
    roic = (nopat / invested).where(invested > 0)

    value = float(roic.median()) if roic.notna().any() else float("nan")
    if np.isnan(value):
        return float("nan")
    return float(min(max(value, MIN_TERMINAL_ROIC), MAX_TERMINAL_ROIC))


def normalise_terminal_cash_flow(
    terminal_nopat: float,
    terminal_fcff: float,
    growth: float,
    roic: float,
    wacc: float,
) -> TerminalCashFlow:
    """Make terminal reinvestment consistent with terminal growth.

    This is the single most consequential correction in a DCF. The forecast holds capital
    spending at its historical share of revenue, which for a company in an investment phase
    is far above what merely sustaining the business needs. Carrying that into perpetuity
    charges the company for growth it is never credited with: Reliance's forecast reinvests
    145% of NOPAT forever, enough to fund roughly 9.7% growth at its own returns, while the
    terminal value only ever pays it for 6.4%. Discounted forever, that gap is enormous, and
    it is why an unadjusted DCF can value a heavily investing company at a small fraction of
    what it is worth.

    The identity that fixes it is the one that ties growth to reinvestment:

        growth = reinvestment rate x ROIC        so        reinvestment rate = growth / ROIC

    A company growing at 6% while earning 12% on new capital must plough back half its
    NOPAT, no more and no less. Terminal FCFF is therefore NOPAT x (1 - g/ROIC), which is
    internally consistent by construction.

    This raises most valuations, so it deserves scepticism rather than acceptance. The
    defence is that the alternative is not conservative, it is incoherent: it assumes
    spending that buys nothing.
    """
    if np.isnan(roic) or roic <= 0 or terminal_nopat <= 0:
        return TerminalCashFlow(
            terminal_fcff, terminal_fcff, terminal_nopat, roic, float("nan"),
            "ROIC could not be estimated, so the forecast cash flow is carried into the "
            "terminal value unadjusted. Where capex exceeds D&A that understates value.",
        )

    if roic <= growth:
        # Growing faster than the return on capital consumes cash without limit, so the
        # growth assumption is the thing that has to give.
        capped = max(roic - 0.01, 0.0)
        rate = capped / roic if roic > 0 else 1.0
        normalised = terminal_nopat * (1.0 - rate)
        return TerminalCashFlow(
            terminal_fcff, normalised, terminal_nopat, roic, rate,
            f"Terminal growth of {growth:.1%} is at or above the {roic:.1%} return on "
            f"invested capital, which cannot be sustained: growth funded at a return below "
            f"its cost destroys value. Growth is capped at {capped:.1%} for the terminal "
            "calculation.",
        )

    rate = growth / roic
    normalised = terminal_nopat * (1.0 - rate)

    note = (
        f"Terminal reinvestment set to {rate:.0%} of NOPAT, the amount needed to sustain "
        f"{growth:.1%} growth at a {roic:.1%} return on invested capital. The forecast year "
        f"reinvests a different amount, which is right while the company is building but "
        f"incoherent in perpetuity."
    )
    if roic < wacc:
        note += (
            f" Note that ROIC of {roic:.1%} is below the {wacc:.1%} cost of capital, so on "
            "these numbers growth destroys value rather than creating it."
        )

    return TerminalCashFlow(terminal_fcff, normalised, terminal_nopat, roic, rate, note)


@dataclass(frozen=True)
class TerminalValue:
    method: str
    value: float
    basis: str
    usable: bool
    note: str = ""


def perpetual_growth(terminal_fcff: float, wacc: float, growth: float) -> TerminalValue:
    """Gordon growth terminal value."""
    if wacc <= growth:
        return TerminalValue(
            "perpetual growth", float("nan"),
            f"FCFF {terminal_fcff:,.0f} x (1 + {growth:.2%}) / ({wacc:.2%} - {growth:.2%})",
            usable=False,
            note=("WACC does not exceed terminal growth, so the denominator is zero or "
                  "negative and the formula returns infinity or a negative value that looks "
                  "like an answer."),
        )

    if terminal_fcff <= 0:
        return TerminalValue(
            "perpetual growth", float("nan"),
            f"terminal FCFF is {terminal_fcff:,.0f}", usable=False,
            note=("Terminal free cash flow is not positive. Capitalising it in perpetuity "
                  "would value the company at a negative number for continuing to exist, so "
                  "the exit multiple is the only defensible route here."),
        )

    value = terminal_fcff * (1.0 + growth) / (wacc - growth)
    note = ""
    if wacc - growth < 0.03:
        note = (f"The spread between WACC and growth is only {wacc - growth:.2%}. It sits in "
                "the denominator, so the terminal value is extremely sensitive to both and "
                "small changes move the whole valuation.")

    return TerminalValue(
        "perpetual growth", value,
        f"{terminal_fcff:,.0f} x (1 + {growth:.2%}) / ({wacc:.2%} - {growth:.2%})",
        usable=True, note=note,
    )


def exit_multiple(terminal_ebitda: float, multiple: float) -> TerminalValue:
    """Terminal value from a comparable EV/EBITDA multiple."""
    if terminal_ebitda <= 0 or np.isnan(multiple) or multiple <= 0:
        return TerminalValue(
            "exit multiple", float("nan"),
            f"EBITDA {terminal_ebitda:,.0f} x {multiple}", usable=False,
            note="Terminal EBITDA or the multiple is not usable.",
        )

    return TerminalValue(
        "exit multiple", terminal_ebitda * multiple,
        f"terminal EBITDA {terminal_ebitda:,.0f} x {multiple:.1f}x", usable=True,
    )


def implied_exit_multiple(terminal_value: float, terminal_ebitda: float) -> float:
    """What EV/EBITDA a perpetuity terminal value implies.

    The strongest single cross-check on a DCF. If a growth assumption implies the company
    will be worth 40x EBITDA forever, the growth assumption is the problem, and the implied
    multiple exposes that far more clearly than the growth rate does on its own.
    """
    if terminal_ebitda <= 0 or np.isnan(terminal_value):
        return float("nan")
    return terminal_value / terminal_ebitda


def implied_growth(terminal_value: float, terminal_fcff: float, wacc: float) -> float:
    """The growth rate an exit multiple implies, running Gordon growth backwards."""
    if terminal_value <= 0 or terminal_fcff <= 0:
        return float("nan")
    # TV = FCFF(1+g)/(WACC-g)  =>  g = (TV x WACC - FCFF) / (TV + FCFF)
    return (terminal_value * wacc - terminal_fcff) / (terminal_value + terminal_fcff)
