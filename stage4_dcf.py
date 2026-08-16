"""Stage 4: the discounted cash flow valuation.

Run:  .venv/bin/python stage4_dcf.py --ticker RELIANCE
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
from src.valuation.dcf import cross_checks, reverse_dcf, run_dcf
from src.valuation.fcff import build_fcff
from src.valuation.forecasting import build_forecast
from src.valuation.market_data import fetch_price_history, fetch_snapshot, risk_free_rate
from src.valuation.universe import NIFTY_UNIVERSE
from src.valuation.wacc import build_wacc, validate_wacc
from stage1_historical import resolve_market

RULE = "=" * 78


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> int:
    p = argparse.ArgumentParser(description="DCF valuation")
    p.add_argument("--ticker", default="RELIANCE")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--exit-multiple", type=float, default=None,
                   help="terminal EV/EBITDA for the cross-check")
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()

    ticker, data_dir, market = resolve_market(args.ticker.upper(), args.data_dir)
    quote = f"{ticker}.NS" if f"{ticker}.NS" in NIFTY_UNIVERSE else ticker

    try:
        hist = load_history(ticker, data_dir)
    except (FileNotFoundError, DataQualityError) as exc:
        print(f"error: {exc}")
        return 1

    quality = data_quality_report(hist, ticker)
    if quality["blocking"]:
        print("Cannot value this company:")
        for b in quality["blocking"]:
            print(f"  - {b}")
        return 1

    name = NIFTY_UNIVERSE.get(f"{ticker}.NS", (ticker, ""))[0]
    analysis = historical.analyse(hist)
    a = asmp.derive(hist, analysis, ticker, horizon=args.horizon,
                    nominal_gdp_growth=market["nominal_gdp_growth"],
                    inflation=market.get("inflation", 0.02))
    fc = build_forecast(hist, a, ticker)
    fcff = build_fcff(fc)

    rf, rf_date = risk_free_rate(series=market["risk_free_series"])
    snap = fetch_snapshot(quote)
    beta = estimate_beta(fetch_price_history(quote, years=5),
                         fetch_price_history(market["index_ticker"], years=5),
                         market["index_name"])

    w = build_wacc(ticker, hist, beta, snap.market_cap / 1e6, rf,
                   market["equity_risk_premium"], a.tax_rate)

    wacc_errors = validate_wacc(w, a.terminal_growth)
    if wacc_errors:
        print("Discount rate failed validation, so no valuation is produced:")
        for e in wacc_errors:
            print(f"  - {e}")
        return 1

    net_debt = float(hist["net_debt"].iloc[-1])
    roic = a.terminal_roic
    shares = snap.shares_outstanding / 1e6  # millions, to match the cash flows

    try:
        result = run_dcf(
            ticker=ticker, fcff=fcff, wacc=w.wacc, terminal_growth=a.terminal_growth,
            net_debt=net_debt, shares_outstanding=shares,
            current_share_price=snap.share_price, currency=market["currency"],
            exit_ev_ebitda=args.exit_multiple, roic=roic,
        )
    except ValueError as exc:
        print(f"Valuation not possible: {exc}")
        return 1

    cur = market["currency"]

    section(f"{ticker}  |  STAGE 4: DISCOUNTED CASH FLOW VALUATION")
    print(f"{name}  |  {cur} millions  |  base FY{fc.base_year}  |  {fc.horizon}-year forecast")
    print(f"Discount rate {w.wacc:.2%}  |  terminal growth {a.terminal_growth:.2%}  |  "
          f"risk-free {rf:.2%} ({rf_date})")

    section("1. DISCOUNTING THE FORECAST CASH FLOWS")
    s = result.schedule.copy()
    s["Year"] = s["year"].astype(str) + "E"
    print(s[["Year", "fcff", "discount_factor", "pv_fcff"]].rename(columns={
        "fcff": "FCFF", "discount_factor": "Discount factor", "pv_fcff": "PV of FCFF"}
    ).to_string(index=False, formatters={
        "FCFF": lambda v: f"{v:,.0f}",
        "Discount factor": lambda v: f"{v:.4f}",
        "PV of FCFF": lambda v: f"{v:,.0f}"}))
    print(f"\n  Sum of PV of forecast FCFF: {result.pv_forecast:,.0f}")
    print("  End-year discounting is used. Cash actually arrives through the year, so this")
    print("  understates value by roughly half a year of discounting and errs low.")

    section("2. TERMINAL VALUE")
    tcf = result.terminal_cash_flow
    if tcf is not None and not pd.isna(tcf.reinvestment_rate):
        print(f"  Terminal NOPAT             {tcf.nopat:,.0f}")
        print(f"  Return on invested capital {tcf.roic:.1%}   (median of history)")
        print(f"  Reinvestment rate          {tcf.reinvestment_rate:.0%}   (growth / ROIC)")
        print(f"  Forecast-year FCFF         {tcf.raw_fcff:,.0f}")
        print(f"  Normalised terminal FCFF   {tcf.normalised_fcff:,.0f}")
        print()
    print(f"  Method    {result.terminal.method}")
    print(f"  Basis     {result.terminal.basis}")
    print(f"  Value     {result.terminal.value:,.0f}")
    print(f"  PV        {result.pv_terminal:,.0f}")
    print(f"  Share of enterprise value: {result.terminal_share:.0%}")
    for c in cross_checks(result, float(fcff.frame["ebitda"].iloc[-1])):
        print(f"\n  {c}")

    section("3. ENTERPRISE VALUE TO EQUITY VALUE")
    print(result.bridge_table().to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    print(f"\n  Net debt is total debt less cash, taken from FY{int(hist.fiscal_year.iloc[-1])}.")

    section("4. VALUATION")
    print(f"  Implied share price   {cur} {result.implied_share_price:,.2f}")
    print(f"  Current share price   {cur} {result.current_share_price:,.2f}   (as at {snap.as_of})")
    print(f"  Upside / downside     {result.upside:+.1%}")
    print(f"\n  Equity value          {cur} {result.equity_value/1e6:,.2f} tn")
    print(f"  Market cap            {cur} {snap.market_cap/1e12:,.2f} tn")

    verdict = ("appears undervalued on these assumptions" if result.upside > 0.15
               else "appears overvalued on these assumptions" if result.upside < -0.15
               else "sits close to fair value on these assumptions")
    print(f"\n  On this model the share {verdict}.")
    print("  That is a statement about the assumptions, not a recommendation. Change the")
    print("  growth or margin inputs and the conclusion changes with them.")

    section("5. REVERSE DCF: WHAT WOULD JUSTIFY TODAY'S PRICE")
    implied_wacc = reverse_dcf(fcff, snap.market_cap / 1e6, net_debt, a.terminal_growth, roic)
    if pd.isna(implied_wacc):
        print("  No discount rate in a plausible range reproduces the market price, so the gap")
        print("  is not explained by the cost of capital alone.")
    else:
        print(f"  Modelled WACC        {w.wacc:.2%}")
        print(f"  Market-implied WACC  {implied_wacc:.2%}   (the rate at which this model agrees with the price)")
        print(f"  Difference           {w.wacc - implied_wacc:+.2%}")
        print()
        if implied_wacc < w.wacc:
            print("  The market is applying a lower required return than this model assumes. That is")
            print("  a disagreement about the equity risk premium, not about the company, and it is")
            print("  a more useful place for the argument to sit than a verdict on the share price.")
        else:
            print("  The market is applying a higher required return than this model assumes.")

    if result.notes or result.warnings:
        section("6. WHAT TO ARGUE WITH")
        for n in result.notes:
            print(f"  - {n}")
        for n in result.warnings:
            print(f"  - {n}")
        for n in beta.warnings:
            print(f"  - beta: {n}")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    result.schedule.to_csv(out / f"{ticker}_stage4_dcf.csv", index=False)
    print(f"\nWrote outputs/{ticker}_stage4_dcf.csv")

    section("NEXT")
    print("  Stage 5 values the company against its sector peers on trading multiples, which")
    print("  is an independent check on this number rather than a refinement of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
