"""Stage 2b: free cash flow to the firm.

    EBIT
    - taxes on EBIT
    = NOPAT
    + D&A
    - capex
    - change in working capital
    = FCFF

Every line is a column, never folded into one expression, because the whole point of a
valuation model is that someone can disagree with a specific number rather than with a
black box.

**Why taxes on EBIT and not the reported tax charge.** FCFF is the cash available to every
provider of capital, debt and equity alike, before any financing decision. The reported tax
charge is computed after interest is deducted, so it already contains the tax shield on
debt. Using it here would put the financing benefit into the cash flow, and WACC puts it
into the discount rate as well, valuing the same shield twice. So tax is recomputed on
unlevered EBIT and interest never appears in this statement at all. This is the single
place a DCF most often double-counts, and it always flatters the answer.

**Why D&A is added back.** It was subtracted to get EBIT but no cash left the business.
Capex is the actual cash spent on assets and is subtracted separately. Treating D&A as the
cost of capital assets rather than capex would value a company on an accounting convention
rather than on what it spends.

**Why the change in working capital and not its level.** Only the movement consumes or
releases cash. A company can carry a large receivables balance forever and use no cash at
all, provided the balance is not growing. The change is computed as the assumed working
capital intensity applied to the revenue increment, which is what "growth consumes cash in
proportion to the revenue it adds" means arithmetically. It also avoids a step in the first
forecast year: bridging from the last reported balance to a ratio-implied one would produce
a one-off cash flow that reflects the choice of ratio rather than the business.

A negative FCFF is not an error. A company growing fast and investing heavily genuinely
consumes cash, and the model should show that rather than smooth it away.
"""

from dataclasses import dataclass

import pandas as pd

from src.valuation.forecasting import Forecast

# FCFF lines in statement order, so the output reads top to bottom like a statement.
# Labels are neutral rather than "Add"/"Less", because the working capital line can go
# either way: a business with negative working capital releases cash as it grows, and
# printing "Less" above a positive number would misstate the direction. Every figure is a
# cash impact, so a negative number is always an outflow.
FCFF_LINES = [
    ("ebit", "EBIT"),
    ("tax_on_ebit", "Tax on EBIT"),
    ("nopat", "NOPAT"),
    ("dep_amort", "Depreciation and amortisation"),
    ("capex", "Capital expenditure"),
    ("change_in_nwc", "Change in working capital"),
    ("fcff", "FCFF"),
]


@dataclass
class FCFFResult:
    ticker: str
    frame: pd.DataFrame
    warnings: list[str]

    @property
    def terminal_fcff(self) -> float:
        return float(self.frame["fcff"].iloc[-1])


def build_fcff(fc: Forecast) -> FCFFResult:
    """Compute FCFF for each forecast year, keeping every component as its own column."""
    a = fc.assumptions
    f = fc.frame.copy()

    f["tax_rate"] = a.tax_rate
    f["tax_on_ebit"] = f["ebit"] * a.tax_rate
    f["nopat"] = f["ebit"] - f["tax_on_ebit"]

    f["capex"] = f["revenue"] * a.capex_pct_revenue
    f["capex_pct_revenue"] = a.capex_pct_revenue

    # Working capital scales with revenue at the assumed intensity, so the cash effect is
    # that intensity applied to the revenue added in the year.
    f["change_in_nwc"] = f["revenue_increment"] * a.nwc_pct_revenue
    f["implied_nwc"] = f["revenue"] * a.nwc_pct_revenue

    f["fcff"] = f["nopat"] + f["dep_amort"] - f["capex"] - f["change_in_nwc"]
    f["fcff_margin"] = f["fcff"] / f["revenue"]

    return FCFFResult(ticker=fc.ticker, frame=f, warnings=review_fcff(fc, f))


def review_fcff(fc: Forecast, f: pd.DataFrame) -> list[str]:
    """Flag results that are legitimate but need saying out loud."""
    notes: list[str] = []

    if (f["fcff"] < 0).all():
        notes.append(
            "FCFF is negative in every forecast year. The business consumes cash throughout, "
            "so essentially all of its value will sit in the terminal period, where the "
            "assumptions are weakest."
        )
    elif (f["fcff"] < 0).any():
        years = ", ".join(str(int(y)) for y in f.loc[f["fcff"] < 0, "year"])
        notes.append(f"FCFF is negative in {years}, so the company consumes cash in those years.")

    if f["fcff"].iloc[-1] <= 0:
        notes.append(
            "Terminal-year FCFF is not positive. A perpetuity growth terminal value built on "
            "it would be meaningless, so the exit-multiple approach is the only defensible "
            "one for this company."
        )

    # Capex below D&A means the company is not replacing what it consumes.
    if (f["capex"] < f["dep_amort"]).all():
        notes.append(
            "Capex runs below D&A in every year, so the asset base shrinks in real terms. "
            "That is sustainable for a while and not in perpetuity, which inflates FCFF."
        )

    reinvestment = (f["capex"] + f["change_in_nwc"]) / f["nopat"].where(f["nopat"] > 0)
    if reinvestment.notna().any() and float(reinvestment.mean()) > 1.0:
        notes.append(
            f"Reinvestment averages {float(reinvestment.mean()):.0%} of NOPAT, so the company "
            "is investing more than it earns after tax and is funding growth externally."
        )

    return notes


def validate_fcff(f: pd.DataFrame, tolerance: float = 1e-6) -> list[str]:
    """Arithmetic checks. These test the model, not the company.

    Running is not the same as being correct, so each identity in the build is re-derived
    independently and compared.
    """
    errors: list[str] = []

    if f[[k for k, _ in FCFF_LINES]].isna().any().any():
        errors.append("FCFF contains missing values")

    rebuilt_nopat = f["ebit"] - f["tax_on_ebit"]
    if not ((rebuilt_nopat - f["nopat"]).abs() < tolerance).all():
        errors.append("NOPAT does not equal EBIT less taxes on EBIT")

    rebuilt_fcff = f["nopat"] + f["dep_amort"] - f["capex"] - f["change_in_nwc"]
    if not ((rebuilt_fcff - f["fcff"]).abs() < tolerance).all():
        errors.append("FCFF does not reconcile to its components")

    rebuilt_ebit = f["ebitda"] - f["dep_amort"]
    if not ((rebuilt_ebit - f["ebit"]).abs() < tolerance).all():
        errors.append("EBIT does not equal EBITDA less D&A")

    if (f["year"].diff().dropna() != 1).any():
        errors.append("forecast years are not consecutive")

    if (f["revenue"] <= 0).any():
        errors.append("forecast revenue is not positive")

    return errors


def statement_frame(result: FCFFResult) -> pd.DataFrame:
    """The FCFF build as a statement: lines down, years across."""
    f = result.frame
    data = {}
    for key, label in FCFF_LINES:
        # Capex and the working capital movement are deductions, shown as negatives so the
        # column reads as an addition down the page.
        sign = -1.0 if key in ("tax_on_ebit", "capex", "change_in_nwc") else 1.0
        data[label] = (f[key] * sign).values

    return pd.DataFrame(data, index=[f"{int(y)}E" for y in f["year"]]).T
