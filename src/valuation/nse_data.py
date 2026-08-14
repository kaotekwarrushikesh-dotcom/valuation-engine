"""Fetch Nifty company financials into Module 1's cleaned-statement schema.

Module 1 sources US filers from SEC EDGAR. Indian companies do not file with the SEC, so
that path cannot reach them. The architecture principle still holds, because what Module 2
actually depends on is Module 1's *schema*, not its source: one cleaned table, one row per
fiscal year, the same column names. This module is a second loader that satisfies that same
contract for NSE-listed companies, and every downstream stage (data bridge, historical
analysis, assumptions, and later the DCF) runs unchanged on the output.

Two differences from the EDGAR path are real and worth stating rather than hiding:

  History is shorter. EDGAR gives ten years. The India source gives four to five, which is
  at the floor of what a trend-based forecast can stand on. Assumption confidence is
  downgraded accordingly rather than pretending the evidence is as strong.

  Units are INR millions. Indian filings quote crore (1 crore = 10 million). Storing
  millions keeps a single code path with the USD side; ratios are unit-free, and only the
  per-share bridge needs the unit to be consistent, which it is.

Banks and non-bank financial companies are excluded on purpose. Their balance sheets are
unclassified and, more fundamentally, FCFF is not the right cash flow for a bank: debt is
raw material rather than financing, so enterprise value has no clean meaning. Valuing them
needs an excess-return or FCFE model, which is a different engine, not a looser one.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# yfinance row labels vary a little by company, so each field lists candidates in priority
# order and the first one present wins.
INCOME_MAP = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cogs": ["Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "ebit": ["EBIT", "Operating Income", "Total Operating Income As Reported"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "tax_expense": ["Tax Provision"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "eps": ["Diluted EPS", "Basic EPS"],
}

BALANCE_MAP = {
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "current_assets": ["Current Assets"],
    "total_assets": ["Total Assets"],
    "current_liabilities": ["Current Liabilities"],
    "total_debt": ["Total Debt"],
    "short_term_debt": ["Current Debt", "Current Debt And Capital Lease Obligation"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "equity": ["Stockholders Equity", "Common Stock Equity"],
}

CASHFLOW_MAP = {
    "cfo": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure", "Purchase Of PPE"],
    "financing_cash_flow": ["Financing Cash Flow"],
    "investing_cash_flow": ["Investing Cash Flow"],
    "dep_amort": ["Depreciation And Amortization", "Depreciation Amortization Depletion"],
}

# Column order matching Module 1's output exactly.
SCHEMA = [
    "fiscal_year", "revenue", "cogs", "gross_profit", "ebitda", "ebit", "dep_amort",
    "interest_expense", "tax_expense", "net_income", "eps", "cash", "current_assets",
    "total_assets", "current_liabilities", "total_debt", "short_term_debt",
    "long_term_debt", "equity", "cfo", "capex", "financing_cash_flow", "investing_cash_flow",
]

INR_MILLION = 1e6


def _pick(frame: pd.DataFrame, candidates: list[str], column) -> float:
    """First candidate row that carries a value for this period."""
    for name in candidates:
        if name in frame.index:
            value = frame.loc[name, column]
            if pd.notna(value):
                return float(value)
    return float("nan")


def fiscal_year_of(period) -> int:
    """Label an Indian fiscal year by the calendar year holding most of it.

    Indian companies close on 31 March, so FY ending March 2026 ran April 2025 to March
    2026 and is labelled 2025. This matches Module 1's convention (a period ending before
    June belongs to the prior calendar year), which keeps the two datasets comparable.
    """
    ts = pd.Timestamp(period)
    return ts.year if ts.month >= 6 else ts.year - 1


def build_company_frame(ticker: str) -> pd.DataFrame:
    """Fetch one NSE company and return it in Module 1's schema, in INR millions."""
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = yf.Ticker(ticker)
        income, balance, cash = t.financials, t.balance_sheet, t.cashflow

    if income.empty or balance.empty or cash.empty:
        raise ValueError(f"{ticker}: no financial statements returned")

    periods = sorted(set(income.columns) & set(balance.columns) & set(cash.columns))
    if not periods:
        raise ValueError(f"{ticker}: no periods common to all three statements")

    rows = []
    for period in periods:
        row: dict[str, float] = {"fiscal_year": fiscal_year_of(period)}
        for field, names in INCOME_MAP.items():
            row[field] = _pick(income, names, period)
        for field, names in BALANCE_MAP.items():
            row[field] = _pick(balance, names, period)
        for field, names in CASHFLOW_MAP.items():
            row[field] = _pick(cash, names, period)
        rows.append(row)

    df = pd.DataFrame(rows)

    # yfinance reports capital expenditure as a negative cash outflow; the schema stores it
    # as a positive amount spent, matching Module 1.
    df["capex"] = df["capex"].abs()

    # Derive what is missing rather than dropping the year.
    df["dep_amort"] = df["dep_amort"].fillna(df["ebitda"] - df["ebit"])
    df["ebitda"] = df["ebitda"].fillna(df["ebit"] + df["dep_amort"])
    df["gross_profit"] = df["gross_profit"].fillna(df["revenue"] - df["cogs"])
    df["cogs"] = df["cogs"].fillna(df["revenue"] - df["gross_profit"])

    both_debt_missing = df["long_term_debt"].isna() & df["short_term_debt"].isna()
    summed = df["long_term_debt"].fillna(0.0) + df["short_term_debt"].fillna(0.0)
    df["total_debt"] = df["total_debt"].fillna(summed.where(~both_debt_missing))

    for col in SCHEMA:
        if col not in df.columns:
            df[col] = np.nan
    df = df[SCHEMA]

    money = [c for c in SCHEMA if c not in ("fiscal_year", "eps")]
    df[money] = df[money] / INR_MILLION

    df["fiscal_year"] = df["fiscal_year"].astype(int)
    df = df.sort_values("fiscal_year").reset_index(drop=True)

    # A gap would make year-over-year changes and CAGR silently wrong, so keep only the
    # unbroken run ending at the most recent year.
    years = df["fiscal_year"].tolist()
    start = years[-1]
    for earlier, later in zip(reversed(years[:-1]), reversed(years[1:])):
        if later - earlier != 1:
            break
        start = earlier
    df = df[df["fiscal_year"] >= start].reset_index(drop=True)

    core = ["revenue", "ebit", "net_income", "total_assets", "equity", "cfo"]
    df = df.dropna(subset=core).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{ticker}: no years with complete core data")

    return df


def fetch_universe(tickers: dict[str, tuple[str, str]], data_dir: Path) -> tuple[list, list]:
    """Fetch every company, writing one CSV per company. Returns (ok, failed)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []

    for ticker, (name, sector) in tickers.items():
        try:
            df = build_company_frame(ticker)
            out = data_dir / f"{ticker.replace('.NS', '')}_financials.csv"
            df.to_csv(out, index=False)
            span = f"FY{int(df.fiscal_year.min())}-FY{int(df.fiscal_year.max())}"
            ok.append((ticker, name, len(df), span))
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            failed.append((ticker, str(exc)))

    return ok, failed
