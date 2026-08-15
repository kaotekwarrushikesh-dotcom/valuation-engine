"""WACC: the blended return every provider of capital requires.

    WACC = We x Ke + Wd x Kd x (1 - t)

    Ke = risk-free + beta x equity risk premium      (CAPM)
    Kd = interest expense / average total debt        (what the company actually pays)

Four choices carry the finance:

**Weights use market value of equity, not book.** Book equity is an accounting residual and
for a company that has bought back stock it can be negative, which would make the weights
meaningless. Market capitalisation is what the equity is worth today, which is what a
provider of capital is actually committing. Debt is taken at book value, which is standard:
the market value of corporate debt is rarely observable and, for investment-grade borrowers
trading near par, book is a close enough proxy.

**Cost of debt is what the company pays, not a rating table.** Interest expense over average
debt gives the effective borrowing rate directly from the statements. It is checked against
the risk-free rate, since a company cannot borrow far below the sovereign, and falls back to
a spread over the government yield where the company carries too little debt to measure.

**The tax shield sits here and only here.** Interest is deductible, so the after-tax cost of
debt is Kd x (1 - t). This is the only place in the model that benefit appears: FCFF is
struck on unlevered EBIT precisely so that it is not counted twice.

**WACC must exceed terminal growth.** A perpetuity with g >= WACC divides by zero or by a
negative number and produces either infinity or a negative value that looks like a real
answer. This is checked and blocks the valuation rather than being quietly clamped.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.valuation.beta import BetaEstimate

# A company cannot borrow far below its own government. Where the measured rate implies it,
# the measurement is wrong rather than the company being a better credit than the sovereign.
MIN_SPREAD_OVER_RISK_FREE = 0.0
# Used only where a company carries too little debt for the effective rate to be measurable.
DEFAULT_CREDIT_SPREAD = 0.02
# Below this share of assets, debt is immaterial: the ratio of a full-year interest charge
# to a tiny balance is dominated by lease interest and one-off items, and produces rates
# like Hindustan Unilever's 26% that describe the measurement rather than the borrowing.
MATERIAL_DEBT_RATIO = 0.05
# A listed large-cap paying more than this over the sovereign is in distress, not borrowing
# normally, so a rate above it points at the same measurement problem from the other side.
MAX_SPREAD_OVER_RISK_FREE = 0.08


@dataclass
class WACCResult:
    """A WACC and every component behind it."""

    ticker: str
    risk_free: float
    equity_risk_premium: float
    beta_used: float
    cost_of_equity: float
    pre_tax_cost_of_debt: float
    tax_rate: float
    after_tax_cost_of_debt: float
    market_value_equity: float
    book_value_debt: float
    weight_equity: float
    weight_debt: float
    wacc: float
    beta: BetaEstimate | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def component_table(self) -> pd.DataFrame:
        """The build, line by line, so any input can be challenged on its own."""
        rows = [
            ("Risk-free rate", f"{self.risk_free:.2%}", "10Y government bond"),
            ("Equity risk premium", f"{self.equity_risk_premium:.2%}", "assumption, includes country risk"),
            ("Beta (adjusted)", f"{self.beta_used:.2f}", "regression vs index, Blume adjusted"),
            ("Cost of equity", f"{self.cost_of_equity:.2%}", "risk-free + beta x ERP"),
            ("Pre-tax cost of debt", f"{self.pre_tax_cost_of_debt:.2%}", "interest / average debt"),
            ("Tax rate", f"{self.tax_rate:.2%}", "effective rate"),
            ("After-tax cost of debt", f"{self.after_tax_cost_of_debt:.2%}", "Kd x (1 - t)"),
            ("Market value of equity", f"{self.market_value_equity:,.0f}", "market capitalisation"),
            ("Book value of debt", f"{self.book_value_debt:,.0f}", "total debt"),
            ("Weight of equity", f"{self.weight_equity:.1%}", "E / (D + E)"),
            ("Weight of debt", f"{self.weight_debt:.1%}", "D / (D + E)"),
            ("WACC", f"{self.wacc:.2%}", "We x Ke + Wd x Kd x (1 - t)"),
        ]
        return pd.DataFrame(rows, columns=["Component", "Value", "Basis"])


def cost_of_equity(risk_free: float, beta: float, equity_risk_premium: float) -> float:
    """CAPM. The return equity holders require for bearing this company's market risk."""
    return risk_free + beta * equity_risk_premium


def cost_of_debt(hist: pd.DataFrame, risk_free: float) -> tuple[float, str]:
    """Effective borrowing rate from the statements, with a stated fallback.

    Interest expense is a full-year flow while debt is a point-in-time balance, so the
    average of opening and closing debt is the right denominator. Using the closing balance
    alone would understate the rate for a company that borrowed during the year.
    """
    debt = hist["total_debt"]
    interest = hist["interest_expense"]

    if len(hist) >= 2 and pd.notna(interest.iloc[-1]):
        average_debt = (debt.iloc[-1] + debt.iloc[-2]) / 2.0
        assets = hist["total_assets"].iloc[-1]

        if pd.notna(average_debt) and average_debt > assets * MATERIAL_DEBT_RATIO:
            measured = float(interest.iloc[-1] / average_debt)
            if measured > risk_free + MAX_SPREAD_OVER_RISK_FREE:
                return (
                    risk_free + DEFAULT_CREDIT_SPREAD,
                    f"the measured rate of {measured:.1%} is higher than a solvent listed "
                    "company borrows at, which happens when a full-year interest charge that "
                    "includes lease and other financing costs is divided by a small debt "
                    "balance. A spread over the government bond is used instead",
                )

            if measured >= risk_free + MIN_SPREAD_OVER_RISK_FREE:
                return measured, "interest expense over average total debt"
            return (
                risk_free + DEFAULT_CREDIT_SPREAD,
                f"the measured rate of {measured:.1%} sits below the {risk_free:.1%} sovereign "
                "yield, which usually means part of the debt is borrowed in a foreign currency "
                "at a lower coupon. That coupon is cheaper only because the borrower carries "
                "the currency risk, so discounting rupee cash flows at it would price the debt "
                "and ignore the exposure that comes with it. A rupee spread over the government "
                "bond is used instead, which raises WACC and is the conservative direction",
            )

    return (
        risk_free + DEFAULT_CREDIT_SPREAD,
        "debt is immaterial, so the effective rate is not measurable and a spread over the "
        "government bond is assumed; the weight on this term is small enough that it barely "
        "moves the result",
    )


def build_wacc(
    ticker: str,
    hist: pd.DataFrame,
    beta: BetaEstimate,
    market_cap: float,
    risk_free: float,
    equity_risk_premium: float,
    tax_rate: float,
    peer_beta: float | None = None,
) -> WACCResult:
    """Assemble WACC from its components."""
    notes: list[str] = []
    warnings: list[str] = list(beta.warnings)

    beta_used = beta.adjusted
    if np.isnan(beta_used):
        if peer_beta is not None and not np.isnan(peer_beta):
            beta_used = peer_beta
            notes.append(
                f"The company's own regression failed, so the sector median beta of "
                f"{peer_beta:.2f} is used in its place."
            )
        else:
            beta_used = 1.0
            warnings.append(
                "no usable beta and no peer set, so a market beta of 1.0 is assumed, which "
                "says nothing about this company's actual risk"
            )
    elif peer_beta is not None and not np.isnan(peer_beta) and abs(beta_used - peer_beta) > 0.5:
        notes.append(
            f"Beta of {beta_used:.2f} sits well away from the sector median of {peer_beta:.2f}. "
            "That is worth explaining rather than accepting."
        )

    ke = cost_of_equity(risk_free, beta_used, equity_risk_premium)
    kd, kd_basis = cost_of_debt(hist, risk_free)
    notes.append(f"Cost of debt: {kd_basis}.")

    after_tax_kd = kd * (1.0 - tax_rate)

    debt = float(hist["total_debt"].iloc[-1])
    if pd.isna(debt) or debt < 0:
        debt = 0.0
        warnings.append("total debt is unavailable, so the structure is treated as all equity")

    capital = market_cap + debt
    if capital <= 0:
        raise ValueError(f"{ticker}: capital base is not positive, WACC cannot be computed")

    we = market_cap / capital
    wd = debt / capital

    wacc = we * ke + wd * after_tax_kd

    if wd > 0.7:
        notes.append(
            f"Debt is {wd:.0%} of capital. At that level the capital structure is doing much "
            "of the work in the discount rate, and small changes to it move the valuation."
        )

    return WACCResult(
        ticker=ticker,
        risk_free=risk_free,
        equity_risk_premium=equity_risk_premium,
        beta_used=beta_used,
        cost_of_equity=ke,
        pre_tax_cost_of_debt=kd,
        tax_rate=tax_rate,
        after_tax_cost_of_debt=after_tax_kd,
        market_value_equity=market_cap,
        book_value_debt=debt,
        weight_equity=we,
        weight_debt=wd,
        wacc=wacc,
        beta=beta,
        notes=notes,
        warnings=warnings,
    )


def validate_wacc(result: WACCResult, terminal_growth: float) -> list[str]:
    """Checks that must pass before these numbers are used to discount anything."""
    errors: list[str] = []

    if np.isnan(result.wacc):
        errors.append("WACC is not a number")
        return errors

    if result.wacc <= terminal_growth:
        errors.append(
            f"WACC of {result.wacc:.2%} does not exceed terminal growth of "
            f"{terminal_growth:.2%}. A perpetuity on those inputs divides by zero or by a "
            "negative number, so no terminal value from them is meaningful."
        )

    if result.wacc <= 0:
        errors.append(f"WACC of {result.wacc:.2%} is not positive")

    if not 0.03 <= result.wacc <= 0.35:
        errors.append(
            f"WACC of {result.wacc:.2%} falls outside a plausible range for a listed "
            "company, which points at a bad input rather than an unusual company"
        )

    if abs(result.weight_equity + result.weight_debt - 1.0) > 1e-9:
        errors.append("capital structure weights do not sum to 1")

    return errors
