"""Market data: prices, share counts, and the risk-free rate.

Module 1 covers the filings. Nothing in a filing tells you what the market thinks the
company is worth today, and a valuation needs that for three separate purposes:

  Share count and price   To turn equity value into a per-share number and compare it
                          with the traded price.
  Price history           To regress company returns against the market and estimate beta.
  Risk-free rate          The base of the cost of equity under CAPM.

Sources and why:
  Prices and shares  yfinance. Free and no key. It is an unofficial interface, so every
                     call is treated as capable of failing or returning nothing.
  Risk-free rate     FRED series DGS10, the 10-year US Treasury constant maturity yield.
                     Ten years is used rather than 3-month because the cash flows being
                     discounted are long-dated, and the discount rate should match the
                     horizon of what it discounts.

Everything fetched is cached to disk. A valuation that silently changes because a quote
moved between two runs is not reproducible, and reproducibility matters more here than
freshness.
"""

import json
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "market_cache"

# The equity risk premium cannot be looked up from a public feed the way a Treasury yield
# can; it is an estimate. This is the widely used implied-ERP level for mature US equity
# (Damodaran's published series has run in the 4.0-4.6% range). It is stated here as a
# named, overridable assumption rather than buried inside the WACC calculation.
DEFAULT_EQUITY_RISK_PREMIUM = 0.045


@dataclass(frozen=True)
class MarketSnapshot:
    """What the market says about the company, as at a fixed date."""

    ticker: str
    as_of: str
    share_price: float
    shares_outstanding: float
    market_cap: float
    currency: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def _cache_path(cache_dir: Path | None, name: str) -> Path:
    d = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def risk_free_rate(cache_dir: Path | None = None, series: str = "DGS10") -> tuple[float, str]:
    """Latest 10-year Treasury yield as a decimal, with the observation date.

    Falls back to the most recent cached value if FRED is unreachable, because a stale
    risk-free rate is far less damaging than a missing one.
    """
    cache = _cache_path(cache_dir, f"fred_{series}.csv")

    try:
        resp = requests.get(FRED_CSV.format(series=series), timeout=30)
        resp.raise_for_status()
        cache.write_text(resp.text)
    except Exception:
        if not cache.exists():
            raise

    df = pd.read_csv(cache)
    value_col = df.columns[-1]
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        raise ValueError(f"FRED series {series} returned no usable observations")

    last = df.iloc[-1]
    return float(last[value_col]) / 100.0, str(last[df.columns[0]])


def fetch_snapshot(ticker: str, cache_dir: Path | None = None, refresh: bool = False) -> MarketSnapshot:
    """Current price, share count and market cap, cached so runs are reproducible."""
    cache = _cache_path(cache_dir, f"snapshot_{ticker.upper()}.json")

    if cache.exists() and not refresh:
        return MarketSnapshot(**json.loads(cache.read_text()))

    import yfinance as yf

    info = yf.Ticker(ticker).fast_info
    price = info.get("lastPrice")
    shares = info.get("shares")
    market_cap = info.get("marketCap")

    if not price or not shares:
        raise ValueError(
            f"{ticker}: market data unavailable (price={price}, shares={shares}). "
            "Per-share valuation cannot proceed without both."
        )

    snap = MarketSnapshot(
        ticker=ticker.upper(),
        as_of=datetime.now().strftime("%Y-%m-%d %H:%M"),
        share_price=float(price),
        shares_outstanding=float(shares),
        market_cap=float(market_cap) if market_cap else float(price) * float(shares),
        currency=info.get("currency") or "USD",
        source="yfinance fast_info",
    )
    cache.write_text(json.dumps(snap.to_dict(), indent=2))
    return snap


def fetch_price_history(
    ticker: str,
    years: int = 5,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Monthly close prices, used for the beta regression in a later stage.

    Monthly rather than daily: daily returns for a single stock against an index are noisy
    and pick up non-synchronous trading effects, and the standard practice for a beta used
    in CAPM is monthly observations over roughly five years.
    """
    cache = _cache_path(cache_dir, f"prices_{ticker.upper()}_{years}y.csv")

    if cache.exists() and not refresh:
        df = pd.read_csv(cache, parse_dates=["date"])
        return df.set_index("date")

    import yfinance as yf

    raw = yf.Ticker(ticker).history(period=f"{years}y", interval="1mo", auto_adjust=True)
    if raw.empty:
        raise ValueError(f"{ticker}: no price history returned")

    df = pd.DataFrame({"close": raw["Close"]})
    df.index.name = "date"
    df.index = df.index.tz_localize(None)
    df.to_csv(cache)
    return df
