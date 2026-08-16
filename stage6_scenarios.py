"""Stage 6: bull, base and bear scenarios, plus sensitivity tables.

Run:  .venv/bin/python stage6_scenarios.py --ticker RELIANCE
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.valuation import assumptions as asmp
from src.valuation import historical
from src.valuation.beta import estimate_beta
from src.valuation.data_bridge import DataQualityError, data_quality_report, load_history
from src.valuation.market_data import fetch_price_history, fetch_snapshot, risk_free_rate
from src.valuation.scenarios import run_all_scenarios
from src.valuation.sensitivity import driver_sensitivity, growth_margin_grid, wacc_terminal_growth_grid
from src.valuation.universe import NIFTY_UNIVERSE
from src.valuation.wacc import build_wacc, validate_wacc
from stage1_historical import resolve_market

RULE = "=" * 84


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> int:
    p = argparse.ArgumentParser(description="Scenarios and sensitivity")
    p.add_argument("--ticker", default="RELIANCE")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()

    ticker, data_dir, market = resolve_market(args.ticker.upper(), args.data_dir)
    full_ticker = f"{ticker}.NS" if f"{ticker}.NS" in NIFTY_UNIVERSE else ticker

    try:
        hist = load_history(ticker, data_dir)
    except (FileNotFoundError, DataQualityError) as exc:
        print(f"error: {exc}")
        return 1

    if data_quality_report(hist, ticker)["blocking"]:
        print("Cannot value this company: insufficient history.")
        return 1

    name = NIFTY_UNIVERSE.get(full_ticker, (ticker, ""))[0]
    analysis = historical.analyse(hist)
    base_assumptions = asmp.derive(hist, analysis, ticker, horizon=args.horizon,
                                   nominal_gdp_growth=market["nominal_gdp_growth"],
                                   inflation=market.get("inflation", 0.02))

    rf, _ = risk_free_rate(series=market["risk_free_series"])
    snap = fetch_snapshot(full_ticker)
    beta = estimate_beta(fetch_price_history(full_ticker, years=5),
                         fetch_price_history(market["index_ticker"], years=5),
                         market["index_name"])
    w = build_wacc(ticker, hist, beta, snap.market_cap / 1e6, rf,
                   market["equity_risk_premium"], base_assumptions.tax_rate)

    if validate_wacc(w, base_assumptions.terminal_growth):
        print("WACC failed validation; cannot proceed.")
        return 1

    net_debt = float(hist["net_debt"].iloc[-1])
    shares = snap.shares_outstanding / 1e6
    cur = market["currency"]

    section(f"{ticker}  |  STAGE 6: SCENARIOS AND SENSITIVITY")
    print(f"{name}  |  {cur} millions  |  base WACC {w.wacc:.2%}  |  "
          f"current price {cur} {snap.share_price:,.2f}")

    # --- Scenarios ---------------------------------------------------------------------
    scenarios = run_all_scenarios(
        hist, analysis, ticker, base_assumptions, w.wacc, net_debt, shares,
        snap.share_price, cur, market["nominal_gdp_growth"], market.get("inflation", 0.02),
        args.horizon,
    )

    section("1. BULL / BASE / BEAR")
    for key in ["bear", "base", "bull"]:
        s = scenarios[key]
        print(f"\n  {s.name.upper()}")
        print(f"    {s.adjustments.rationale}")
        if s.error:
            print(f"    Not usable: {s.error}")
            continue
        r = s.dcf
        print(f"    Revenue growth (yr 1): {s.assumptions.revenue_growth_path[0]:>7.1%}   "
              f"EBITDA margin: {s.assumptions.ebitda_margin_path[0]:>6.1%}   "
              f"Terminal growth: {s.assumptions.terminal_growth:>6.1%}   WACC: {s.wacc:>6.2%}")
        print(f"    Implied share price: {cur} {r.implied_share_price_floored:>10,.2f}   "
              f"Upside/downside: {r.upside:+.1%}")
        if r.equity_value <= 0:
            print("    Equity value is not positive on these assumptions: net debt exceeds")
            print("    enterprise value, so the business does not cover its debt and the")
            print("    model reads the equity as worthless, floored at zero rather than")
            print("    shown as a negative price.")

    if all(scenarios[k].dcf is not None for k in scenarios):
        bear_p = scenarios["bear"].dcf.implied_share_price_floored
        base_p = scenarios["base"].dcf.implied_share_price_floored
        bull_p = scenarios["bull"].dcf.implied_share_price_floored
        section("2. SCENARIO SUMMARY")
        print(f"  Bear   {cur} {bear_p:>10,.2f}")
        print(f"  Base   {cur} {base_p:>10,.2f}")
        print(f"  Bull   {cur} {bull_p:>10,.2f}")
        if base_p > 0:
            print(f"\n  Range as a share of the base case: {(bull_p - bear_p) / base_p:.0%}")
            print("  Bear and bull are fixed, stated shifts from the base case, not fitted to")
            print("  produce a target spread. A very wide range says more about how uncertain")
            print("  the base assumptions are than about the company.")
        else:
            print("\n  Base case equity value is not positive, so a range as a percentage of it")
            print("  is not meaningful. All three scenarios reading close to worthless says the")
            print("  company's debt load, not the growth or margin assumptions, dominates the")
            print("  valuation here.")

    # --- Sensitivity ---------------------------------------------------------------------
    section("3. SENSITIVITY: WACC vs TERMINAL GROWTH")
    base_result = scenarios["base"].dcf
    if base_result is not None:
        grid = wacc_terminal_growth_grid(
            scenarios["base"].fcff, w.wacc, base_assumptions.terminal_growth,
            base_assumptions.terminal_roic, net_debt, shares,
        )
        print(grid.to_string(float_format=lambda v: f"{v:,.0f}" if v == v else "n/a",
                             formatters={grid.index.name: lambda v: f"{v:.2%}"}))
        print("\n  Cheap to build: the explicit-period cash flows do not depend on WACC or")
        print("  terminal growth, only the terminal value and discounting do, so one forecast")
        print("  is reused across the whole table.")

    section("4. SENSITIVITY: REVENUE GROWTH vs EBITDA MARGIN")
    gm_grid = growth_margin_grid(
        hist, analysis, ticker, base_assumptions, w.wacc, net_debt, shares, cur,
        market["nominal_gdp_growth"], market.get("inflation", 0.02), args.horizon,
    )
    print(gm_grid.to_string(float_format=lambda v: f"{v:,.0f}" if v == v else "n/a",
                            formatters={gm_grid.index.name: lambda v: f"{v:.1%}"}))
    print("\n  Expensive to build: growth and margin move the explicit-period cash flows")
    print("  themselves, so every cell rebuilds the full forecast. WACC is held at the base")
    print("  case throughout, to isolate the operating drivers from the discount rate.")

    section("5. WHAT DRIVES THE VALUATION MOST")
    if base_result is not None:
        driv = driver_sensitivity(
            hist, analysis, ticker, base_assumptions, w.wacc, base_result.implied_share_price_floored,
            net_debt, shares, cur, market["nominal_gdp_growth"], market.get("inflation", 0.02),
            args.horizon,
        )
        for _, row in driv.iterrows():
            impact = "n/a" if row["impact_pct"] != row["impact_pct"] else f"{row['impact_pct']:+.1%}"
            print(f"  {row['driver']:<18} {row['shift']:>7}  ->  {cur} {row['implied_price']:>10,.2f}"
                  f"   ({impact})")
        if base_result.equity_value > 0:
            print("\n  Ranked by size of impact. A one-point move in the driver at the top changes")
            print("  the valuation more than the same move in any other single assumption.")
        else:
            print("\n  Base equity value is not positive, so a percentage impact is not")
            print("  meaningful (n/a above); the implied prices are shown unfloored here to")
            print("  preserve their true magnitude even though they read as negative.")

    section("NEXT")
    print("  Stage 7 adds Monte Carlo (treating these same drivers as distributions rather")
    print("  than fixed points) and a final summary blending the DCF with Stage 5's")
    print("  comparable-company valuation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
