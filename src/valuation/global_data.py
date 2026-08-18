"""Live statement fetching for any company, not only the curated 33.

Module 2 is deployed as its own standalone repository, so it cannot reach into Module 1's
files at runtime the way local development can; the deployed app only has what is in this
repo. Rather than reimplement currency handling from scratch, this module ports the
already-verified logic from Module 1's `src/providers/yahoo.py` (checked live against
Shell's pence quotation, Infosys's USD-reporting-in-INR-quoted case, and HCL Technologies'
mislabelled currency flag) as a self-contained copy, so the deployed app does not depend on
a sibling repository that will not exist on the server it runs on.

The curated universe exists because comparable-company valuation needs real sector peers,
and a hand-picked list is what makes Stage 5 trustworthy (see universe.py and the README's
Calibration section, where the curated peer set is what proved the DCF's terminal-value
method, not the underlying data, was the source of a systematic gap). Nothing here tries to
solve peer discovery for an arbitrary company; that is a genuinely different, harder problem,
and pretending an auto-discovered peer group is as reliable as a curated one would quietly
break the property that made this engine's own diagnosis trustworthy in the first place. A
company fetched through this module gets a DCF and scenarios; it does not get comparables.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

INCOME_MAP = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "ebit": ["EBIT", "Operating Income", "Total Operating Income As Reported"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "tax_expense": ["Tax Provision"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
}

BALANCE_MAP = {
    "cash": ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"],
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
    "dep_amort": ["Depreciation And Amortization", "Depreciation Amortization Depletion"],
}

SCHEMA = [
    "fiscal_year", "revenue", "ebitda", "ebit", "dep_amort", "interest_expense",
    "tax_expense", "net_income", "cash", "current_assets", "total_assets",
    "current_liabilities", "total_debt", "short_term_debt", "long_term_debt", "equity",
    "cfo", "capex",
]

MILLION = 1e6
PLAUSIBLE_PRICE_TO_SALES = (0.05, 50.0)
MINOR_UNIT_CODES = {"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS"}
MINOR_UNIT_DIVISOR = {"GBP": 100.0, "ZAR": 100.0, "ILS": 100.0}


@dataclass(frozen=True)
class GlobalCompany:
    ticker: str
    name: str
    frame: pd.DataFrame
    currency: str
    share_price: float
    shares_outstanding: float
    market_cap: float
    notes: list[str]


def _pick(frame: pd.DataFrame, candidates: list[str], column) -> float:
    for name in candidates:
        if name in frame.index:
            value = frame.loc[name, column]
            if pd.notna(value):
                return float(value)
    return float("nan")


def _fiscal_year_of(period) -> int:
    ts = pd.Timestamp(period)
    return ts.year if ts.month >= 6 else ts.year - 1


def _normalise_quote(price: float, market_cap: float, currency: str) -> tuple[float, float, str, list[str]]:
    code = MINOR_UNIT_CODES.get(currency)
    if code is None:
        return price, market_cap, (currency or "").upper(), []
    divisor = MINOR_UNIT_DIVISOR.get(code, 100.0)
    note = (f"Quoted in {currency} (minor unit), converted to {code} by dividing by "
           f"{divisor:.0f}, or the price would read a hundredfold too high.")
    return price / divisor, market_cap / divisor, code, [note]


def _fx_rate(from_currency: str, to_currency: str) -> float:
    if from_currency == to_currency:
        return 1.0
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist = yf.Ticker(f"{from_currency}{to_currency}=X").history(period="5d")
    if hist.empty:
        raise ValueError(f"no {from_currency}{to_currency} exchange rate available")
    return float(hist["Close"].iloc[-1])


def _resolve_statement_currency(declared: str, quote_currency: str, latest_revenue: float,
                                market_cap: float) -> tuple[str, float, list[str]]:
    declared = (declared or quote_currency or "").upper()
    if not declared or declared == quote_currency or latest_revenue <= 0 or market_cap <= 0:
        return quote_currency, 1.0, []

    try:
        rate = _fx_rate(declared, quote_currency)
    except Exception:  # noqa: BLE001
        return quote_currency, 1.0, [f"statements appear to be in {declared} but no exchange rate was available"]

    # Currencies close in size (e.g. USD vs GBP) leave the plausibility test unable to
    # discriminate, so the source's own tag is trusted rather than second-guessed there.
    if 0.2 <= rate <= 5.0:
        return declared, rate, [
            f"Statements reported in {declared}, quoted in {quote_currency}, converted at {rate:.4g}."
        ]

    low, high = PLAUSIBLE_PRICE_TO_SALES
    as_quote = market_cap / latest_revenue
    as_declared = market_cap / (latest_revenue * rate)
    if low <= as_declared <= high and not (low <= as_quote <= high):
        return declared, rate, [
            f"Statements reported in {declared} while the share quotes in {quote_currency}, "
            f"converted at {rate:.4g}."
        ]
    if low <= as_quote <= high and not (low <= as_declared <= high):
        return quote_currency, 1.0, [
            f"The data source tags these statements {declared}, but the implied valuation only "
            f"makes sense if they are already in {quote_currency}, so the tag is ignored."
        ]
    return quote_currency, 1.0, []


def fetch(ticker: str) -> GlobalCompany:
    """Fetch one company from Yahoo Finance, any market, into the schema data_bridge needs."""
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = yf.Ticker(ticker)
        income, balance, cash = t.financials, t.balance_sheet, t.cashflow
        fast = t.fast_info
        try:
            info = t.info
        except Exception:  # noqa: BLE001
            info = {}

    if income.empty or balance.empty or cash.empty:
        raise ValueError(f"{ticker}: no financial statements available")

    periods = sorted(set(income.columns) & set(balance.columns) & set(cash.columns))
    if not periods:
        raise ValueError(f"{ticker}: no periods common to all three statements")

    rows = []
    for period in periods:
        row: dict[str, float] = {"fiscal_year": _fiscal_year_of(period)}
        for field, names in INCOME_MAP.items():
            row[field] = _pick(income, names, period)
        for field, names in BALANCE_MAP.items():
            row[field] = _pick(balance, names, period)
        for field, names in CASHFLOW_MAP.items():
            row[field] = _pick(cash, names, period)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["capex"] = df["capex"].abs()
    df["dep_amort"] = df["dep_amort"].fillna(df["ebitda"] - df["ebit"])
    df["ebitda"] = df["ebitda"].fillna(df["ebit"] + df["dep_amort"])

    both_missing = df["long_term_debt"].isna() & df["short_term_debt"].isna()
    summed = df["long_term_debt"].fillna(0.0) + df["short_term_debt"].fillna(0.0)
    df["total_debt"] = df["total_debt"].fillna(summed.where(~both_missing))

    for col in SCHEMA:
        if col not in df.columns:
            df[col] = np.nan
    df = df[SCHEMA].copy()
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    df = df.sort_values("fiscal_year").reset_index(drop=True)

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
        raise ValueError(f"{ticker}: no fiscal years with complete core data")

    raw_price = float(fast.get("lastPrice") or 0.0)
    raw_cap = float(fast.get("marketCap") or 0.0)
    raw_currency = fast.get("currency") or info.get("currency") or ""
    price, market_cap, quote_currency, notes = _normalise_quote(raw_price, raw_cap, raw_currency)

    latest_revenue = float(df["revenue"].iloc[-1])
    currency, rate, currency_notes = _resolve_statement_currency(
        info.get("financialCurrency", ""), quote_currency, latest_revenue, market_cap)
    notes.extend(currency_notes)

    money = [c for c in SCHEMA if c != "fiscal_year"]
    df[money] = df[money] * rate / MILLION

    if len(df) < 5:
        notes.append(f"Only {len(df)} years of history against the curated universe's 4-5.")

    return GlobalCompany(
        ticker=ticker.upper(),
        name=info.get("longName") or info.get("shortName") or ticker.upper(),
        frame=df, currency=quote_currency or "USD",
        share_price=price, shares_outstanding=float(fast.get("shares") or 0.0),
        market_cap=market_cap, notes=notes,
    )
