"""Bridge from Module 1's cleaned statements to the inputs a valuation needs.

Module 1 produces one CSV per company of filed statement lines. A valuation needs a
different set of quantities, all derived from those lines rather than re-sourced:

  D&A                  EBITDA - EBIT, or Module 1's dep_amort column where present
  Non-cash working     (current assets - cash) - (current liabilities - short-term debt)
  capital              Cash and debt are financing, not operating, items. Leaving them in
                       would make working capital move with the cash balance, and cash
                       movements are the output of the model, not an input to it.
  Change in NWC        The year-over-year movement. This is what consumes cash, not the
                       level. A company can carry a large working capital balance and use
                       no cash at all as long as the balance is not growing.
  Net debt             Total debt - cash. Used to bridge enterprise value to equity value.
  Effective tax rate   Tax expense / pretax income, where pretax income is rebuilt as
                       net income + tax expense. This is the cash-relevant rate actually
                       paid, not the statutory rate.
  Unlevered FCF        Reported CFO - capex is levered (CFO is after interest). It is kept
                       here for trend analysis only; the DCF builds FCFF from EBIT instead,
                       so that the discount rate carries the whole financing effect.

Nothing here forecasts. This module only restates history.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Where Module 1 writes its cleaned per-company CSVs.
DEFAULT_MODULE1_DATA = Path.home() / "financial_statement_intelligence" / "data"

REQUIRED = [
    "fiscal_year",
    "revenue",
    "ebitda",
    "ebit",
    "net_income",
    "tax_expense",
    "cash",
    "current_assets",
    "current_liabilities",
    "total_debt",
    "equity",
    "cfo",
    "capex",
]


class DataQualityError(Exception):
    """Raised when history is too incomplete to value the company honestly."""


def module1_path(ticker: str, data_dir: Path | None = None) -> Path:
    return (Path(data_dir) if data_dir else DEFAULT_MODULE1_DATA) / f"{ticker.upper()}_financials.csv"


def load_history(ticker: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load one company's history from Module 1 and derive the valuation inputs."""
    path = module1_path(ticker, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No Module 1 data for {ticker} at {path}. Run fetch_data.py in Module 1 first."
        )

    df = pd.read_csv(path)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise DataQualityError(f"{ticker}: Module 1 file is missing columns {sorted(missing)}")

    df = df.sort_values("fiscal_year").reset_index(drop=True)
    return derive_valuation_inputs(df)


def derive_valuation_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived quantities a DCF consumes. Pure restatement of history."""
    d = df.copy()

    # D&A: prefer the reported line, fall back to the EBITDA-to-EBIT difference.
    reported_da = d["dep_amort"] if "dep_amort" in d.columns else pd.Series(np.nan, index=d.index)
    d["dep_amort"] = reported_da.fillna(d["ebitda"] - d["ebit"])

    # Short-term debt is needed to strip financing out of current liabilities. Where the
    # split is unavailable, fall back to total current liabilities and flag it downstream
    # rather than guessing a split.
    std = d["short_term_debt"] if "short_term_debt" in d.columns else pd.Series(np.nan, index=d.index)
    d["short_term_debt"] = std

    operating_ca = d["current_assets"] - d["cash"]
    operating_cl = d["current_liabilities"] - d["short_term_debt"].fillna(0.0)
    d["nwc"] = operating_ca - operating_cl
    d["change_in_nwc"] = d["nwc"].diff()

    d["net_debt"] = d["total_debt"] - d["cash"]

    pretax_income = d["net_income"] + d["tax_expense"]
    d["pretax_income"] = pretax_income
    d["effective_tax_rate"] = (d["tax_expense"] / pretax_income).where(pretax_income > 0)

    d["fcf_levered"] = d["cfo"] - d["capex"]

    # Ratios that become the forecast drivers.
    d["revenue_growth"] = d["revenue"].pct_change()
    d["ebitda_margin"] = d["ebitda"] / d["revenue"]
    d["ebit_margin"] = d["ebit"] / d["revenue"]
    d["da_pct_revenue"] = d["dep_amort"] / d["revenue"]
    d["capex_pct_revenue"] = d["capex"] / d["revenue"]
    d["nwc_pct_revenue"] = d["nwc"] / d["revenue"]
    d["fcf_margin"] = d["fcf_levered"] / d["revenue"]

    return d


def data_quality_report(hist: pd.DataFrame, ticker: str) -> dict:
    """Assess whether the history supports a defensible forecast.

    A valuation is only as good as the drivers behind it, so this checks the specific
    series the forecast reads rather than the file as a whole.
    """
    drivers = ["revenue", "ebitda_margin", "da_pct_revenue", "capex_pct_revenue", "nwc_pct_revenue"]
    n_years = len(hist)

    coverage = {}
    for col in drivers:
        available = int(hist[col].notna().sum())
        coverage[col] = {"available": available, "of": n_years}

    warnings: list[str] = []
    blocking: list[str] = []

    for col, cov in coverage.items():
        if cov["available"] == 0:
            blocking.append(f"{col} is entirely missing, so it cannot be forecast")
        elif cov["available"] < 3:
            blocking.append(f"{col} has only {cov['available']} usable years, too few to trend")
        elif cov["available"] < n_years:
            warnings.append(
                f"{col} covers {cov['available']} of {n_years} years; "
                "assumptions are drawn from the years that exist"
            )

    if hist["short_term_debt"].isna().any():
        warnings.append(
            "short-term debt is missing in some years, so working capital there includes "
            "financing items and its level is overstated"
        )

    if n_years < 5:
        warnings.append(f"only {n_years} years of history, which is thin for a trend-based forecast")

    return {
        "ticker": ticker,
        "years": n_years,
        "first_year": int(hist["fiscal_year"].iloc[0]),
        "last_year": int(hist["fiscal_year"].iloc[-1]),
        "coverage": coverage,
        "warnings": warnings,
        "blocking": blocking,
        "usable": not blocking,
    }
