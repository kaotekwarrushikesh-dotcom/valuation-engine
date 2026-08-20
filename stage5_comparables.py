"""Stage 5: comparable company valuation.

Run:  .venv/bin/python stage5_comparables.py --ticker RELIANCE
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from valuation_engine.comparables import (
    MIN_PEERS,
    MULTIPLE_LABELS,
    compute_multiples,
    run_comparables,
)
from valuation_engine.data_bridge import DataQualityError, load_history
from valuation_engine.market_data import fetch_snapshot
from valuation_engine.universe import NIFTY_UNIVERSE, peers_for
from stage1_historical import resolve_market

RULE = "=" * 78


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def fmt(v: float, suffix: str = "x") -> str:
    return "n/a" if v is None or v != v else f"{v:.2f}{suffix}"


def load_multiples(full_ticker: str, data_dir) -> object | None:
    """Load one company's multiples, or None if its data is not usable."""
    ticker = full_ticker.replace(".NS", "")
    name = NIFTY_UNIVERSE.get(full_ticker, (ticker, ""))[0]
    sector = NIFTY_UNIVERSE.get(full_ticker, (ticker, ""))[1]
    try:
        hist = load_history(ticker, data_dir)
        snap = fetch_snapshot(full_ticker)
        return compute_multiples(
            ticker, name, sector, hist,
            snap.share_price, snap.shares_outstanding / 1e6, snap.market_cap / 1e6,
        )
    except Exception:  # noqa: BLE001 - a bad peer must not stop the run
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Comparable company valuation")
    p.add_argument("--ticker", default="RELIANCE")
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()

    ticker, data_dir, market = resolve_market(args.ticker.upper(), args.data_dir)
    full_ticker = f"{ticker}.NS" if f"{ticker}.NS" in NIFTY_UNIVERSE else ticker

    try:
        hist = load_history(ticker, data_dir)
    except (FileNotFoundError, DataQualityError) as exc:
        print(f"error: {exc}")
        return 1

    name, sector = NIFTY_UNIVERSE.get(full_ticker, (ticker, "Unknown"))
    snap = fetch_snapshot(full_ticker)
    target = compute_multiples(
        ticker, name, sector, hist,
        snap.share_price, snap.shares_outstanding / 1e6, snap.market_cap / 1e6,
    )

    peer_tickers = peers_for(full_ticker)
    section(f"{ticker}  |  STAGE 5: COMPARABLE COMPANY VALUATION")
    print(f"{name}  |  {sector}  |  {market['currency']} millions")
    print(f"\nPeer selection: same sector as classified in universe.py, fixed independently")
    print(f"of this run, so the peer group cannot be adjusted to produce a wanted answer.")
    print(f"Candidate peers ({len(peer_tickers)}): {', '.join(t.replace('.NS','') for t in peer_tickers)}")

    peers = [m for m in (load_multiples(t, data_dir) for t in peer_tickers) if m is not None]
    failed = len(peer_tickers) - len(peers)
    if failed:
        print(f"{failed} peer(s) had unusable data and were dropped.")

    if len(peers) < MIN_PEERS:
        print(f"\nOnly {len(peers)} usable peers, below the minimum of {MIN_PEERS}.")
        print("A peer group this thin is not a peer group. No comparable valuation is produced.")
        return 1

    section("1. TARGET AND PEER MULTIPLES")
    header = f"{'Company':<14}{'EV/EBITDA':>11}{'EV/Rev':>9}{'P/E':>9}{'P/B':>9}"
    print(header)
    print(f"{target.ticker:<14}{fmt(target.ev_ebitda):>11}{fmt(target.ev_revenue):>9}"
          f"{fmt(target.pe):>9}{fmt(target.pb):>9}   <- target")
    for peer in peers:
        print(f"{peer.ticker:<14}{fmt(peer.ev_ebitda):>11}{fmt(peer.ev_revenue):>9}"
              f"{fmt(peer.pe):>9}{fmt(peer.pb):>9}")

    if target.excluded:
        print("\nExcluded for the target:")
        for m, reason in target.excluded.items():
            print(f"  {MULTIPLE_LABELS[m]}: {reason}")

    result = run_comparables(target, peers)

    section("2. PEER STATISTICS AND PREMIUM / DISCOUNT")
    print(f"{'Multiple':<12}{'N':>3}{'Mean':>9}{'Median':>9}{'Min':>9}{'Max':>9}"
          f"{'Target':>9}{'Prem/Disc':>11}")
    for m in MULTIPLE_LABELS:
        stat = result.stats.get(m)
        if stat is None:
            print(f"{MULTIPLE_LABELS[m]:<12}  fewer than {MIN_PEERS} usable peers")
            continue
        pd_ = result.premium_discount(m)
        pd_str = "n/a" if pd_ != pd_ else f"{pd_:+.0%}"
        print(f"{MULTIPLE_LABELS[m]:<12}{stat.n:>3}{stat.mean:>9.2f}{stat.median:>9.2f}"
              f"{stat.min:>9.2f}{stat.max:>9.2f}{fmt(getattr(target, m)):>9}{pd_str:>11}")
        if stat.outliers:
            print(f"    outlier(s) excluded from the median's influence: {', '.join(stat.outliers)}")

    section("3. IMPLIED VALUATION BY MULTIPLE")
    print(f"{'Multiple':<12}{'Peer median':>12}{'Target metric':>15}{'Implied EV':>13}"
          f"{'Implied equity':>15}{'Implied price':>14}")
    for m, imp in result.implied.items():
        ev_str = "n/a" if imp.implied_enterprise_value is None else f"{imp.implied_enterprise_value:,.0f}"
        print(f"{MULTIPLE_LABELS[m]:<12}{imp.peer_median:>12.2f}{imp.target_metric:>15,.2f}"
              f"{ev_str:>13}{imp.implied_equity_value:>15,.0f}{imp.implied_share_price:>14,.2f}")

    section("4. BLENDED COMPARABLE VALUATION")
    print(f"  Weighted across {', '.join(MULTIPLE_LABELS[m] for m in result.blended_multiples_used)}")
    print("  (enterprise multiples weighted above equity ones, since EV/EBITDA and")
    print("  EV/Revenue are unaffected by the target's own leverage and P/E and P/B are not)")
    print(f"\n  Blended comparable value   {market['currency']} {result.blended_share_price:,.2f}")
    print(f"  Current share price        {market['currency']} {result.current_share_price:,.2f}")
    print(f"  Upside / downside          {result.upside:+.1%}")

    print("\n  This is independent of the DCF: it prices the company off what the market")
    print("  currently pays for its peers, not off a forecast of its own cash flows. Where")
    print("  the two disagree, the disagreement is informative on its own, particularly")
    print("  since Stage 4's terminal-value mechanism has a known bias against high-quality,")
    print("  low-growth compounders (see the README's Calibration section).")

    section("NEXT")
    print("  Stage 6 builds bull/base/bear scenarios and sensitivity tables around the DCF.")
    print("  Stage 7 adds Monte Carlo and a final blended summary combining both methods.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
