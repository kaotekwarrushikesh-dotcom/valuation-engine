"""Stage 3: the discount rate.

Run:  .venv/bin/python stage3_wacc.py --ticker RELIANCE
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from valuation_engine import assumptions as asmp
from valuation_engine import historical
from valuation_engine.beta import estimate_beta, peer_median_beta
from valuation_engine.data_bridge import DataQualityError, load_history
from valuation_engine.market_data import fetch_price_history, fetch_snapshot, risk_free_rate
from valuation_engine.universe import NIFTY_UNIVERSE, peers_for
from valuation_engine.wacc import build_wacc, validate_wacc
from stage1_historical import resolve_market

RULE = "=" * 78


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def quote_ticker(ticker: str) -> str:
    return f"{ticker}.NS" if f"{ticker}.NS" in NIFTY_UNIVERSE else ticker


def main() -> int:
    p = argparse.ArgumentParser(description="WACC")
    p.add_argument("--ticker", default="RELIANCE")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--no-peers", action="store_true", help="skip the peer beta cross-check")
    args = p.parse_args()

    ticker, data_dir, market = resolve_market(args.ticker.upper(), args.data_dir)

    try:
        hist = load_history(ticker, data_dir)
    except (FileNotFoundError, DataQualityError) as exc:
        print(f"error: {exc}")
        return 1

    name = NIFTY_UNIVERSE.get(f"{ticker}.NS", (ticker, ""))[0]
    analysis = historical.analyse(hist)
    a = asmp.derive(hist, analysis, ticker, nominal_gdp_growth=market["nominal_gdp_growth"],
                    inflation=market.get("inflation", 0.02))

    section(f"{ticker}  |  STAGE 3: WEIGHTED AVERAGE COST OF CAPITAL")
    print(f"{name}  |  {market['currency']}  |  market: {market['index_name']}")

    # --- Inputs ----------------------------------------------------------------------
    rf, rf_date = risk_free_rate(series=market["risk_free_series"])
    snap = fetch_snapshot(quote_ticker(ticker))
    erp = market["equity_risk_premium"]

    # --- Beta ------------------------------------------------------------------------
    prices = fetch_price_history(quote_ticker(ticker), years=5)
    index_prices = fetch_price_history(market["index_ticker"], years=5)
    b = estimate_beta(prices, index_prices, market["index_name"])

    peer_beta = None
    if not args.no_peers:
        estimates = {}
        for peer in peers_for(f"{ticker}.NS"):
            try:
                estimates[peer] = estimate_beta(
                    fetch_price_history(peer, years=5), index_prices, market["index_name"])
            except Exception:  # noqa: BLE001 - a missing peer must not stop the valuation
                continue
        if estimates:
            peer_beta = peer_median_beta(estimates)

    section("1. BETA")
    print(f"  Regression of {ticker} monthly returns on {market['index_name']}, 5 years")
    print(f"  Raw beta             {b.raw:.3f}")
    print(f"  Adjusted beta        {b.adjusted:.3f}   (Blume: 0.33 + 0.67 x raw, for mean reversion)")
    print(f"  R-squared            {b.r_squared:.2f}     (share of variation the index explains)")
    print(f"  Standard error       {b.standard_error:.3f}")
    print(f"  Observations         {b.observations} monthly returns")
    print(f"  Confidence           {b.confidence}")
    if peer_beta is not None:
        print(f"  Sector median beta   {peer_beta:.3f}   (adjusted, across the peer set)")
    for w in b.warnings:
        print(f"  warning: {w}")

    # --- WACC ------------------------------------------------------------------------
    result = build_wacc(
        ticker=ticker, hist=hist, beta=b, market_cap=snap.market_cap / 1e6,
        risk_free=rf, equity_risk_premium=erp, tax_rate=a.tax_rate, peer_beta=peer_beta,
        sovereign_default_spread=market.get("sovereign_default_spread", 0.0),
    )

    section("2. WACC BUILD")
    print(result.component_table().to_string(index=False))
    print(f"\n  Risk-free from FRED {market['risk_free_series']} as at {rf_date}.")
    print("  Equity is at market value and debt at book. Book equity is an accounting residual")
    print("  and can be negative after buybacks, which would make the weights meaningless.")

    for n in result.notes:
        print(f"\n  {n}")
    for w in result.warnings:
        print(f"\n  warning: {w}")

    # --- Validation ------------------------------------------------------------------
    errors = validate_wacc(result, a.terminal_growth)
    section("3. VALIDATION")
    if errors:
        print("  FAILED. This discount rate must not be used:")
        for e in errors:
            print(f"    - {e}")
        return 1

    print(f"  WACC {result.wacc:.2%} exceeds terminal growth {a.terminal_growth:.2%}, so a")
    print("  perpetuity is well defined. Weights sum to 1. WACC is within a plausible range.")
    print(f"\n  Spread over terminal growth: {result.wacc - a.terminal_growth:.2%}. The narrower")
    print("  this is, the more of the valuation sits in the terminal value and the more")
    print("  sensitive the answer becomes to both assumptions.")

    section("NEXT")
    print("  Stage 4 discounts the Stage 2 cash flows at this rate, adds a terminal value,")
    print("  bridges enterprise value to equity value through net debt, and divides by shares")
    print("  to produce a fair value per share.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
