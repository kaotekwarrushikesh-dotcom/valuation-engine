"""Stage 7: Monte Carlo over the driver distributions, and the blended summary.

Run:  .venv/bin/python stage7_monte_carlo.py --ticker RELIANCE
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from valuation_engine import assumptions as asmp
from valuation_engine import historical
from valuation_engine.beta import estimate_beta
from valuation_engine.blended import build_blended
from valuation_engine.comparables import compute_multiples, run_comparables
from valuation_engine.data_bridge import DataQualityError, data_quality_report, load_history
from valuation_engine.dcf import run_dcf
from valuation_engine.fcff import build_fcff
from valuation_engine.forecasting import build_forecast
from valuation_engine.market_data import fetch_price_history, fetch_snapshot, risk_free_rate
from valuation_engine.monte_carlo import run_monte_carlo
from valuation_engine.universe import NIFTY_UNIVERSE, peers_for
from valuation_engine.wacc import build_wacc, validate_wacc
from stage1_historical import resolve_market

RULE = "=" * 84
MIN_PEERS = 2


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def peer_multiples(full_ticker: str, data_dir):
    ticker = full_ticker.replace(".NS", "")
    name, sector = NIFTY_UNIVERSE.get(full_ticker, (ticker, ""))
    try:
        hist = load_history(ticker, data_dir)
        snap = fetch_snapshot(full_ticker)
        return compute_multiples(ticker, name, sector, hist, snap.share_price,
                                 snap.shares_outstanding / 1e6, snap.market_cap / 1e6)
    except Exception:  # noqa: BLE001 - one bad peer must not stop the run
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Monte Carlo and blended valuation")
    p.add_argument("--ticker", default="RELIANCE")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--seed", type=int, default=12345)
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
    a = asmp.derive(hist, analysis, ticker, horizon=args.horizon,
                    nominal_gdp_growth=market["nominal_gdp_growth"],
                    inflation=market.get("inflation", 0.02))

    rf, _ = risk_free_rate(series=market["risk_free_series"])
    snap = fetch_snapshot(full_ticker)
    beta = estimate_beta(fetch_price_history(full_ticker, years=5),
                         fetch_price_history(market["index_ticker"], years=5),
                         market["index_name"])
    w = build_wacc(ticker, hist, beta, snap.market_cap / 1e6, rf,
                   market["equity_risk_premium"], a.tax_rate,
                   sovereign_default_spread=market.get("sovereign_default_spread", 0.0))

    if validate_wacc(w, a.terminal_growth):
        print("WACC failed validation; cannot proceed.")
        return 1

    net_debt = float(hist["net_debt"].iloc[-1])
    shares = snap.shares_outstanding / 1e6
    cur = market["currency"]

    section(f"{ticker}  |  STAGE 7: MONTE CARLO AND BLENDED VALUATION")
    print(f"{name}  |  {cur} millions  |  WACC {w.wacc:.2%}  |  "
          f"current price {cur} {snap.share_price:,.2f}")

    # --- The point-estimate DCF, for reference -------------------------------------------
    fcff = build_fcff(build_forecast(hist, a, ticker))
    dcf = run_dcf(ticker=ticker, fcff=fcff, wacc=w.wacc, terminal_growth=a.terminal_growth,
                  net_debt=net_debt, shares_outstanding=shares,
                  current_share_price=snap.share_price, currency=cur, roic=a.terminal_roic)

    # --- Monte Carlo ---------------------------------------------------------------------
    section("1. DRIVER DISTRIBUTIONS")
    mc = run_monte_carlo(
        hist, analysis, ticker, a, w.wacc, beta.standard_error,
        market["equity_risk_premium"], w.weight_equity, net_debt, shares,
        snap.share_price, cur, market["nominal_gdp_growth"],
        market.get("inflation", 0.02), args.horizon,
        trials=args.trials, seed=args.seed,
    )
    print(mc.distributions.table().to_string(index=False))
    for note in mc.distributions.notes:
        print(f"\n  note: {note}")
    print("\n  Every width above is measured from the company's own record except terminal")
    print("  growth, which describes a period that has not happened and is therefore drawn")
    print("  uniformly between the inflation floor and the GDP ceiling rather than given an")
    print("  invented mean and variance.")

    section("2. DISTRIBUTION OF FAIR VALUES")
    print(mc.summary_table().to_string(index=False,
                                       float_format=lambda v: f"{v:,.2f}"))
    print(f"\n  Completed {mc.completed} of {mc.trials} trials "
          f"({mc.failure_rate:.1%} failed, where the drawn terminal growth met or exceeded")
    print("  the drawn WACC and the perpetuity has no finite value). Failed trials are")
    print("  counted rather than dropped: discarding them would remove exactly the bad draws")
    print("  and bias the reported distribution upward.")
    print(f"\n  Probability the company is worth more than its current price: "
          f"{mc.probability_above_market:.1%}")
    print("  This is what a Monte Carlo is for. A point estimate says more or less; this")
    print("  says how often, across the plausible range of the company's own inputs.")
    if mc.completed:
        skew = mc.percentile(95) - mc.median, mc.median - mc.percentile(5)
        if skew[0] > skew[1] * 1.5:
            print("\n  The upper tail is much longer than the lower one. That is the Gordon-growth")
            print("  denominator at work: as drawn terminal growth approaches the drawn WACC the")
            print("  spread between them shrinks toward zero and the terminal value rises without")
            print("  bound, so the mean is a poor summary here and the percentiles are the honest")
            print("  reading.")

    # --- Comparables, for the blend --------------------------------------------------------
    comparables_price = None
    if full_ticker in NIFTY_UNIVERSE:
        target = peer_multiples(full_ticker, data_dir)
        peers = [m for m in (peer_multiples(p, data_dir) for p in peers_for(full_ticker))
                 if m is not None]
        if target is not None and len(peers) >= MIN_PEERS:
            comps = run_comparables(target, peers)
            if comps.blended_share_price == comps.blended_share_price:
                comparables_price = comps.blended_share_price

    section("3. BLENDED SUMMARY")
    blended = build_blended(
        ticker=ticker, currency=cur, current_share_price=snap.share_price,
        dcf_share_price=dcf.implied_share_price_floored,
        comparables_share_price=comparables_price,
        monte_carlo_median=mc.median,
    )
    for e in blended.estimates:
        print(f"  {e.name:<24} {cur} {e.share_price:>10,.2f}   {e.basis}")
    print()
    if blended.usable:
        print(f"  Range          {cur} {blended.low:,.2f} to {cur} {blended.high:,.2f}")
        if len(blended.usable) > 1:
            print(f"  Spread         {blended.dispersion:.0%} of the current price")
        print(f"  Central        {cur} {blended.central:,.2f}  ({blended.upside:+.1%})")
    print(f"\n  {blended.verdict}")
    for note in blended.notes:
        print(f"\n  note: {note}")

    section("DONE")
    print("  Stages 1 to 7 complete. The engine reports a range across independent methods")
    print("  and the probability attached to it, rather than a single number that would")
    print("  imply a precision none of the methods individually support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
