"""Stage 2: forecast and free cash flow to the firm.

Run:  .venv/bin/python stage2_forecast.py --ticker RELIANCE
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from valuation_engine import assumptions as asmp
from valuation_engine import historical
from valuation_engine.data_bridge import DataQualityError, data_quality_report, load_history
from valuation_engine.fcff import build_fcff, statement_frame, validate_fcff
from valuation_engine.forecasting import build_forecast, forecast_vs_history
from valuation_engine.universe import NIFTY_UNIVERSE
from stage1_historical import resolve_market

RULE = "=" * 84


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> int:
    p = argparse.ArgumentParser(description="Forecast and FCFF")
    p.add_argument("--ticker", default="RELIANCE")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()

    ticker, data_dir, market = resolve_market(args.ticker.upper(), args.data_dir)

    try:
        hist = load_history(ticker, data_dir)
    except (FileNotFoundError, DataQualityError) as exc:
        print(f"error: {exc}")
        return 1

    quality = data_quality_report(hist, ticker)
    if not quality["blocking"] is not None and quality["blocking"]:
        print("Cannot proceed:")
        for b in quality["blocking"]:
            print(f"  - {b}")
        return 1

    name = NIFTY_UNIVERSE.get(f"{ticker}.NS", (ticker, ""))[0]
    analysis = historical.analyse(hist)
    a = asmp.derive(hist, analysis, ticker, horizon=args.horizon,
                    nominal_gdp_growth=market["nominal_gdp_growth"],
                    inflation=market.get("inflation", 0.02))

    fc = build_forecast(hist, a, ticker)
    result = build_fcff(fc)

    section(f"{ticker}  |  STAGE 2: FORECAST AND FCFF")
    print(f"{name}  |  {market['currency']} millions  |  "
          f"base year FY{fc.base_year}  |  horizon {fc.horizon} years")

    # --- Arithmetic validation, before anything is read as a result ------------------
    errors = validate_fcff(result.frame)
    if errors:
        print("\nMODEL FAILED VALIDATION. Results below are not trustworthy:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nValidation: all identities reconcile "
          "(EBIT = EBITDA - D&A, NOPAT = EBIT - tax, FCFF = NOPAT + D&A - capex - dNWC).")

    # --- Forecast income statement ---------------------------------------------------
    section("1. FORECAST INCOME STATEMENT")
    f = result.frame
    inc = pd.DataFrame({
        "Year": [f"{int(y)}E" for y in f["year"]],
        "Revenue": f["revenue"].map(lambda v: f"{v:,.0f}"),
        "Growth": f["revenue_growth"].map(lambda v: f"{v:.1%}"),
        "EBITDA": f["ebitda"].map(lambda v: f"{v:,.0f}"),
        "Margin": f["ebitda_margin"].map(lambda v: f"{v:.1%}"),
        "D&A": f["dep_amort"].map(lambda v: f"{v:,.0f}"),
        "EBIT": f["ebit"].map(lambda v: f"{v:,.0f}"),
        "EBIT margin": f["ebit_margin"].map(lambda v: f"{v:.1%}"),
    })
    print(inc.to_string(index=False))
    print(f"\n  Base FY{fc.base_year} revenue {fc.base_revenue:,.0f}, "
          f"terminal FY{fc.terminal_year} revenue {f['revenue'].iloc[-1]:,.0f} "
          f"({f['revenue'].iloc[-1]/fc.base_revenue:.2f}x over {fc.horizon} years)")

    # --- FCFF build ------------------------------------------------------------------
    section("2. FREE CASH FLOW TO THE FIRM")
    print(statement_frame(result).to_string(
        float_format=lambda v: f"{v:,.0f}", justify="right"))
    print("\n  Every figure is a cash impact, so a negative number is an outflow.")
    print(f"  Tax on EBIT at {a.tax_rate:.1%}, applied to unlevered EBIT. Interest is absent")
    print("  on purpose: the financing effect belongs in WACC, and counting it here as well")
    print("  would value the debt tax shield twice.")

    nwc_pct = a.nwc_pct_revenue
    if nwc_pct < 0:
        print(f"\n  Working capital runs at {nwc_pct:.1%} of revenue, meaning suppliers and")
        print("  customers fund the business. Growth therefore releases cash rather than")
        print("  consuming it, which is why that line is a source rather than a use.")

    # --- Cash flow drivers -----------------------------------------------------------
    section("3. WHAT DRIVES THE CASH FLOW")
    drv = pd.DataFrame({
        "Year": [f"{int(y)}E" for y in f["year"]],
        "FCFF": f["fcff"].map(lambda v: f"{v:,.0f}"),
        "FCFF margin": f["fcff_margin"].map(lambda v: f"{v:.1%}"),
        "Capex/rev": f["capex_pct_revenue"].map(lambda v: f"{v:.1%}"),
        "dNWC": f["change_in_nwc"].map(lambda v: f"{v:,.0f}"),
        "Reinvest/NOPAT": ((f["capex"] + f["change_in_nwc"]) / f["nopat"]).map(
            lambda v: "n/a" if pd.isna(v) else f"{v:.0%}"),
    })
    print(drv.to_string(index=False))

    hist_fcf_margin = hist["fcf_margin"].tail(3).mean()
    print(f"\n  Forecast FCFF margin averages {f['fcff_margin'].mean():.1%}. "
          f"Reported levered FCF margin ran {hist_fcf_margin:.1%} recently.")
    print("  The two are not the same measure: reported FCF is after interest and after the")
    print("  tax shield, while FCFF is before both, so a gap is expected rather than an error.")

    # --- Judgement -------------------------------------------------------------------
    notes = forecast_vs_history(hist, fc) + result.warnings
    if notes:
        section("4. WHAT TO ARGUE WITH")
        for n in notes:
            print(f"  - {n}")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    f.to_csv(out / f"{ticker}_stage2_fcff.csv", index=False)
    print(f"\nWrote outputs/{ticker}_stage2_fcff.csv")

    section("NEXT")
    print("  Stage 3 builds WACC: CAPM cost of equity, beta by regression against the Nifty 50,")
    print("  after-tax cost of debt, and market-value capital structure weights. Only then can")
    print("  these cash flows be discounted. Nothing here has been discounted yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
