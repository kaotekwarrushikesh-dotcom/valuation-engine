"""Stage 1: Module 1 history to forecast assumptions.

Run:  .venv/bin/python stage1_historical.py --ticker AAPL
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.valuation import assumptions as asmp
from src.valuation import historical
from src.valuation.data_bridge import DataQualityError, data_quality_report, load_history
from src.valuation.market_data import DEFAULT_EQUITY_RISK_PREMIUM, fetch_snapshot, risk_free_rate
from src.valuation.universe import INDIA_MARKET, NIFTY_UNIVERSE

RULE = "=" * 78

NIFTY_DATA = Path(__file__).parent / "data" / "nifty"

# A rupee cash flow needs a rupee discount rate and a rupee GDP ceiling, so the market
# profile travels with the company rather than being a global constant.
US_MARKET = {
    "currency": "USD",
    "index_ticker": "^GSPC",
    "index_name": "S&P 500",
    "risk_free_series": "DGS10",
    "equity_risk_premium": DEFAULT_EQUITY_RISK_PREMIUM,
    "nominal_gdp_growth": 0.04,
    "inflation": 0.02,
}


def resolve_market(ticker: str, data_dir: str | None) -> tuple[str, Path, dict]:
    """Work out whether this is an NSE or a US company, and where its data lives."""
    nse_ticker = ticker if ticker.endswith(".NS") else f"{ticker}.NS"

    if nse_ticker in NIFTY_UNIVERSE:
        base = ticker.replace(".NS", "")
        return base, Path(data_dir) if data_dir else NIFTY_DATA, INDIA_MARKET

    return ticker, Path(data_dir) if data_dir else None, US_MARKET


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def fmt_pct(v: float) -> str:
    return "n/a" if pd.isna(v) else f"{v:.1%}"


def main() -> int:
    p = argparse.ArgumentParser(description="Historical analysis and forecast assumptions")
    p.add_argument("--ticker", default="RELIANCE")
    p.add_argument("--horizon", type=int, default=5, help="forecast years (default 5)")
    p.add_argument("--data-dir", default=None, help="override the data directory")
    p.add_argument("--no-market", action="store_true", help="skip live market data")
    args = p.parse_args()

    ticker, data_dir, market = resolve_market(args.ticker.upper(), args.data_dir)

    # --- Load and validate ----------------------------------------------------------
    try:
        hist = load_history(ticker, data_dir)
    except (FileNotFoundError, DataQualityError) as exc:
        print(f"error: {exc}")
        return 1

    quality = data_quality_report(hist, ticker)
    name = NIFTY_UNIVERSE.get(f"{ticker}.NS", (ticker, ""))[0]

    section(f"{ticker}  |  STAGE 1: HISTORICAL ANALYSIS AND FORECAST ASSUMPTIONS")
    print(f"{name}  |  {market['currency']} millions  |  discount market: {market['index_name']}")
    print(f"Source: cleaned statements in Module 1 schema, FY{quality['first_year']} to "
          f"FY{quality['last_year']} ({quality['years']} years)")

    if quality["blocking"]:
        print("\nCannot proceed. The history does not support a defensible forecast:")
        for b in quality["blocking"]:
            print(f"  - {b}")
        return 1

    if quality["warnings"]:
        print("\nData quality warnings:")
        for w in quality["warnings"]:
            print(f"  - {w}")

    # --- Historical statements -------------------------------------------------------
    section("1. HISTORICAL FINANCIALS (currency millions)")
    view = hist[["fiscal_year", "revenue", "ebitda", "ebit", "dep_amort", "capex",
                 "nwc", "change_in_nwc", "net_debt", "fcf_levered"]].copy()
    view.columns = ["FY", "Revenue", "EBITDA", "EBIT", "D&A", "Capex",
                    "NWC", "dNWC", "Net debt", "FCF"]
    print(view.to_string(index=False, float_format=lambda x: f"{x:,.0f}", na_rep="n/a"))

    # --- Driver ratios ---------------------------------------------------------------
    section("2. DRIVER RATIOS")
    ratios = hist[["fiscal_year", "revenue_growth", "ebitda_margin", "ebit_margin",
                   "da_pct_revenue", "capex_pct_revenue", "nwc_pct_revenue",
                   "effective_tax_rate", "fcf_margin"]].copy()
    ratios.columns = ["FY", "Rev growth", "EBITDA mgn", "EBIT mgn", "D&A/rev",
                      "Capex/rev", "NWC/rev", "Tax rate", "FCF mgn"]
    print(ratios.to_string(
        index=False, na_rep="n/a",
        formatters={c: (lambda x: "n/a" if pd.isna(x) else f"{x:.1%}")
                    for c in ratios.columns if c != "FY"},
    ))

    # --- Interpretation --------------------------------------------------------------
    analysis = historical.analyse(hist)
    section("3. WHAT THE HISTORY SAYS")
    for line in historical.summary_lines(analysis):
        print(f"  - {line}")

    # --- Market context --------------------------------------------------------------
    if not args.no_market:
        section("4. MARKET CONTEXT")
        try:
            quote_ticker = f"{ticker}.NS" if f"{ticker}.NS" in NIFTY_UNIVERSE else ticker
            snap = fetch_snapshot(quote_ticker)
            rf, rf_date = risk_free_rate(series=market["risk_free_series"])
            cur = snap.currency
            print(f"  Share price          {cur} {snap.share_price:,.2f}   (as at {snap.as_of})")
            print(f"  Shares outstanding   {snap.shares_outstanding/1e9:,.3f} bn")
            print(f"  Market cap           {cur} {snap.market_cap/1e9:,.1f} bn")
            latest_net_debt = hist["net_debt"].iloc[-1]
            print(f"  Net debt (last FY)   {cur} {latest_net_debt/1e3:,.1f} bn")
            print(f"  Implied EV           {cur} {(snap.market_cap/1e6 + latest_net_debt)/1e3:,.1f} bn")
            print(f"  Risk-free rate       {rf:.2%}   (10Y govt bond, FRED {market['risk_free_series']}, {rf_date})")
            print(f"  Equity risk premium  {market['equity_risk_premium']:.2%}   (assumption, includes country risk)")
            print(f"  Cost of equity at    {rf + market['equity_risk_premium']:.2%} for a beta of 1.0")
        except Exception as exc:
            print(f"  Market data unavailable: {exc}")
            print("  Stage 1 does not depend on it; the DCF stage will.")

    # --- Assumptions -----------------------------------------------------------------
    a = asmp.derive(hist, analysis, ticker, horizon=args.horizon,
                    nominal_gdp_growth=market["nominal_gdp_growth"],
                    inflation=market.get("inflation", 0.02))
    section(f"5. FORECAST ASSUMPTIONS ({args.horizon} years, every one derived from the history above)")

    order = ["revenue_growth_start", "revenue_growth_path", "ebitda_margin", "da_pct_revenue",
             "capex_pct_revenue", "nwc_pct_revenue", "tax_rate", "terminal_growth"]
    for name in order:
        if name in a.detail:
            print(f"\n  {a.detail[name]}")

    section("6. FORECAST DRIVER TABLE")
    frame = a.as_frame(int(hist["fiscal_year"].iloc[-1]))
    display = frame.copy()
    display["year"] = display["year"].astype(str) + "E"
    print(display.to_string(
        index=False,
        formatters={c: (lambda x: f"{x:.1%}") for c in display.columns if c != "year"},
    ))

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    frame.to_csv(out_dir / f"{ticker}_stage1_assumptions.csv", index=False)
    hist.to_csv(out_dir / f"{ticker}_stage1_historical.csv", index=False)
    print(f"\nWrote outputs/{ticker}_stage1_assumptions.csv and outputs/{ticker}_stage1_historical.csv")

    section("NEXT")
    print("  Stage 2 builds the forecast and FCFF from these drivers, then WACC, then the DCF.")
    print("  Nothing above discounts anything yet. These are the inputs a valuation stands on,")
    print("  and they are worth arguing with before any share price is produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
