"""The valuation pipeline, separated from the app that displays it.

`app.py` used to hold these three functions behind Streamlit's `@st.cache_data`, which made
them unreachable to anything that is not a Streamlit session: importing them pulled in the
whole app module, ran `st.set_page_config`, and tied the caching policy of a library to the
lifetime of a browser tab. Module 4's AI analyst needs to call a valuation as an ordinary
Python function, so the calculation lives here and the caching stays in the app, which is
where a caching policy belongs.

Nothing about the valuation changed in the move. `app.py` imports these and wraps each in
its own `@st.cache_data`, so the app behaves exactly as before.

Two entry points, for the reason set out in the README: `run_pipeline` is the full workflow
for the curated Nifty universe, where real sector peers exist and comparables can act as a
cross-check on the DCF; `run_quick_pipeline` fetches any listed company live and returns a
DCF and scenarios without comparables, because an auto-discovered peer group would be lower
quality without saying so.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from valuation_engine import assumptions as asmp
from valuation_engine import historical
from valuation_engine.beta import estimate_beta
from valuation_engine.blended import build_blended
from valuation_engine.comparables import MIN_PEERS, compute_multiples, run_comparables
from valuation_engine.data_bridge import (
    data_quality_report,
    load_history,
    load_history_from_frame,
)
from valuation_engine.dcf import cross_checks, reverse_dcf, run_dcf
from valuation_engine.fcff import build_fcff
from valuation_engine.forecasting import build_forecast
from valuation_engine.global_data import fetch as fetch_global
from valuation_engine.global_search import search as search_global
from valuation_engine.market_data import fetch_price_history, fetch_snapshot, risk_free_rate
from valuation_engine.monte_carlo import run_monte_carlo
from valuation_engine.scenarios import run_all_scenarios
from valuation_engine.universe import (
    INDIA_MARKET,
    NIFTY_UNIVERSE,
    peers_for,
    resolve_market_for_currency,
)
from valuation_engine.wacc import build_wacc, validate_wacc

# The curated CSVs live in the repo, not in the installed package: they are Module 1 output
# checked in for reproducibility, not library code. A consumer that pip-installs this engine
# gets `run_quick_pipeline`, which fetches live and needs no local data, while
# `run_pipeline` needs the files and says so plainly if they are not there.
DEFAULT_NIFTY_DATA = Path(__file__).resolve().parents[1] / "data" / "nifty"


def _nifty_data(data_dir: Path | None) -> Path:
    resolved = Path(data_dir) if data_dir is not None else DEFAULT_NIFTY_DATA
    if not resolved.exists():
        raise FileNotFoundError(
            f"The curated Nifty data is not at {resolved}. The full workflow reads Module 1 "
            "CSVs checked into this repo, so it only runs from a checkout; use "
            "run_quick_pipeline() for any company fetched live."
        )
    return resolved


def load_peer_multiples(full_ticker: str, data_dir: Path | None = None):
    """One peer's multiples, or None if its data is not usable. Cached per peer, since
    peers repeat across many targets in the same sector and each involves live fetches."""
    ticker = full_ticker.replace(".NS", "")
    name, sector = NIFTY_UNIVERSE.get(full_ticker, (ticker, ""))
    try:
        hist = load_history(ticker, _nifty_data(data_dir))
        snap = fetch_snapshot(full_ticker)
        return compute_multiples(
            ticker, name, sector, hist,
            snap.share_price, snap.shares_outstanding / 1e6, snap.market_cap / 1e6,
        )
    except Exception:  # noqa: BLE001 - a bad peer must not stop the run
        return None



def run_pipeline(full_ticker: str, horizon: int = 5, data_dir: Path | None = None):
    """Stages 1 through 7 for one company. Cached 30 minutes: market data moves, the
    underlying financial statements do not, on any timescale that matters here."""
    ticker = full_ticker.replace(".NS", "")
    name, sector = NIFTY_UNIVERSE[full_ticker]

    hist = load_history(ticker, _nifty_data(data_dir))
    quality = data_quality_report(hist, ticker)
    if quality["blocking"]:
        raise ValueError("; ".join(quality["blocking"]))

    analysis = historical.analyse(hist)
    a = asmp.derive(hist, analysis, ticker, horizon=horizon,
                    nominal_gdp_growth=INDIA_MARKET["nominal_gdp_growth"],
                    inflation=INDIA_MARKET["inflation"])
    fc = build_forecast(hist, a, ticker)
    fcff = build_fcff(fc)

    cur = INDIA_MARKET["currency"]
    rf, rf_date = risk_free_rate(series=INDIA_MARKET["risk_free_series"])
    snap = fetch_snapshot(full_ticker)
    beta = estimate_beta(
        fetch_price_history(full_ticker, years=5),
        fetch_price_history(INDIA_MARKET["index_ticker"], years=5),
        INDIA_MARKET["index_name"],
    )
    wacc_result = build_wacc(ticker, hist, beta, snap.market_cap / 1e6, rf,
                             INDIA_MARKET["equity_risk_premium"], a.tax_rate,
                             sovereign_default_spread=INDIA_MARKET["sovereign_default_spread"])
    wacc_errors = validate_wacc(wacc_result, a.terminal_growth)

    net_debt = float(hist["net_debt"].iloc[-1])
    shares = snap.shares_outstanding / 1e6
    roic = a.terminal_roic

    dcf_result, dcf_error, dcf_checks = None, None, []
    if not wacc_errors:
        try:
            dcf_result = run_dcf(
                ticker=ticker, fcff=fcff, wacc=wacc_result.wacc, terminal_growth=a.terminal_growth,
                net_debt=net_debt, shares_outstanding=shares, current_share_price=snap.share_price,
                currency=cur, roic=roic,
            )
            dcf_checks = cross_checks(dcf_result, float(fcff.frame["ebitda"].iloc[-1]))
        except ValueError as exc:
            dcf_error = str(exc)

    implied_wacc = None
    if dcf_result is not None:
        implied_wacc = reverse_dcf(fcff, snap.market_cap / 1e6, net_debt, a.terminal_growth, roic)

    # Comparables
    target = compute_multiples(ticker, name, sector, hist, snap.share_price, shares,
                               snap.market_cap / 1e6)
    peer_tickers = peers_for(full_ticker)
    peers = [p for p in (load_peer_multiples(t, data_dir) for t in peer_tickers) if p is not None]
    comparables = run_comparables(target, peers) if len(peers) >= MIN_PEERS else None

    # Scenarios (only if the base WACC is usable)
    scenarios = None
    monte_carlo = None
    if not wacc_errors:
        scenarios = run_all_scenarios(
            hist, analysis, ticker, a, wacc_result.wacc, net_debt, shares, snap.share_price,
            cur, INDIA_MARKET["nominal_gdp_growth"], INDIA_MARKET["inflation"], horizon,
        )
        monte_carlo = run_monte_carlo(
            hist, analysis, ticker, a, wacc_result.wacc, beta.standard_error,
            INDIA_MARKET["equity_risk_premium"], wacc_result.weight_equity, net_debt, shares,
            snap.share_price, cur, INDIA_MARKET["nominal_gdp_growth"],
            INDIA_MARKET["inflation"], horizon,
        )

    blended = build_blended(
        ticker=ticker, currency=cur, current_share_price=snap.share_price,
        dcf_share_price=dcf_result.implied_share_price_floored if dcf_result else None,
        comparables_share_price=comparables.blended_share_price if comparables else None,
        monte_carlo_median=monte_carlo.median if monte_carlo else None,
    )

    return {
        "mode": "curated", "currency": cur,
        "risk_free_series": INDIA_MARKET["risk_free_series"], "index_name": INDIA_MARKET["index_name"],
        "ticker": ticker, "name": name, "sector": sector, "hist": hist, "quality": quality,
        "share_price": snap.share_price, "market_cap": snap.market_cap,
        "analysis": analysis, "assumptions": a, "forecast": fc, "fcff": fcff,
        "rf": rf, "rf_date": rf_date, "beta": beta,
        "wacc": wacc_result, "wacc_errors": wacc_errors,
        "dcf": dcf_result, "dcf_error": dcf_error, "dcf_checks": dcf_checks,
        "implied_wacc": implied_wacc, "net_debt": net_debt,
        "target_multiples": target, "peers": peers, "comparables": comparables,
        "scenarios": scenarios, "monte_carlo": monte_carlo, "blended": blended,
    }



def run_quick_pipeline(query: str, horizon: int = 5):
    """Stages 1, 2, 3, 4, 6 and 7 for any company, fetched live. No comparables: real sector
    peers only exist for the curated universe (see universe.py), and an auto-discovered
    peer group would be lower quality without saying so, which defeats the point of having
    comparables as a cross-check in the first place."""
    matches = search_global(query, limit=1)
    if not matches:
        raise ValueError(f"No listed company found for '{query}'")
    ticker = matches[0].ticker

    company = fetch_global(ticker)
    hist = load_history_from_frame(company.frame, ticker)
    quality = data_quality_report(hist, ticker)
    if quality["blocking"]:
        raise ValueError("; ".join(quality["blocking"]))

    market, calibrated = resolve_market_for_currency(company.currency)
    cur = company.currency

    analysis = historical.analyse(hist)
    a = asmp.derive(hist, analysis, ticker, horizon=horizon,
                    nominal_gdp_growth=market["nominal_gdp_growth"], inflation=market["inflation"])
    fc = build_forecast(hist, a, ticker)
    fcff = build_fcff(fc)

    rf, rf_date = risk_free_rate(series=market["risk_free_series"])
    beta = estimate_beta(
        fetch_price_history(ticker, years=5),
        fetch_price_history(market["index_ticker"], years=5),
        market["index_name"],
    )
    wacc_result = build_wacc(ticker, hist, beta, company.market_cap / 1e6, rf,
                             market["equity_risk_premium"], a.tax_rate,
                             sovereign_default_spread=market.get("sovereign_default_spread", 0.0))
    wacc_errors = validate_wacc(wacc_result, a.terminal_growth)

    net_debt = float(hist["net_debt"].iloc[-1])
    shares = company.shares_outstanding / 1e6
    roic = a.terminal_roic

    dcf_result, dcf_error, dcf_checks = None, None, []
    if not wacc_errors:
        try:
            dcf_result = run_dcf(
                ticker=ticker, fcff=fcff, wacc=wacc_result.wacc, terminal_growth=a.terminal_growth,
                net_debt=net_debt, shares_outstanding=shares, current_share_price=company.share_price,
                currency=cur, roic=roic,
            )
            dcf_checks = cross_checks(dcf_result, float(fcff.frame["ebitda"].iloc[-1]))
        except ValueError as exc:
            dcf_error = str(exc)

    implied_wacc = None
    if dcf_result is not None:
        implied_wacc = reverse_dcf(fcff, company.market_cap / 1e6, net_debt, a.terminal_growth, roic)

    scenarios = None
    monte_carlo = None
    if not wacc_errors:
        scenarios = run_all_scenarios(
            hist, analysis, ticker, a, wacc_result.wacc, net_debt, shares, company.share_price,
            cur, market["nominal_gdp_growth"], market["inflation"], horizon,
        )
        monte_carlo = run_monte_carlo(
            hist, analysis, ticker, a, wacc_result.wacc, beta.standard_error,
            market["equity_risk_premium"], wacc_result.weight_equity, net_debt, shares,
            company.share_price, cur, market["nominal_gdp_growth"], market["inflation"], horizon,
        )

    blended = build_blended(
        ticker=ticker, currency=cur, current_share_price=company.share_price,
        dcf_share_price=dcf_result.implied_share_price_floored if dcf_result else None,
        comparables_share_price=None,
        monte_carlo_median=monte_carlo.median if monte_carlo else None,
    )

    return {
        "mode": "quick", "currency": cur, "market_calibrated": calibrated,
        "risk_free_series": market["risk_free_series"], "index_name": market["index_name"],
        "ticker": ticker, "name": company.name, "sector": None, "hist": hist, "quality": quality,
        "share_price": company.share_price, "market_cap": company.market_cap,
        "fetch_notes": company.notes,
        "analysis": analysis, "assumptions": a, "forecast": fc, "fcff": fcff,
        "rf": rf, "rf_date": rf_date, "beta": beta,
        "wacc": wacc_result, "wacc_errors": wacc_errors,
        "dcf": dcf_result, "dcf_error": dcf_error, "dcf_checks": dcf_checks,
        "implied_wacc": implied_wacc, "net_debt": net_debt,
        "target_multiples": None, "peers": [], "comparables": None,
        "scenarios": scenarios, "monte_carlo": monte_carlo, "blended": blended,
    }


# --- Layout --------------------------------------------------------------------------------

