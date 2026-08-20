"""Institutional-Style Valuation Engine, as an app.

Pick a Nifty company, get its full valuation workflow: historical assumptions, forecast and
FCFF, WACC, a discounted cash flow, comparable-company multiples, scenarios and a Monte
Carlo, side by side.

The DCF and comparables are shown together deliberately, never the DCF alone. The DCF still
reads below market across this universe after a country-risk double count in the cost of
equity was found and fixed (see the README's Calibration section), and comparables do not
share that machinery, so pairing them is what stops one number from being read as a verdict
when it is actually one model's opinion among two that disagree. Where they disagree widely
the summary reports the range rather than averaging them into a midpoint no method supports.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.valuation import assumptions as asmp
from src.valuation import historical
from src.valuation.beta import estimate_beta
from src.valuation.blended import build_blended
from src.valuation.comparables import (
    MIN_PEERS,
    MULTIPLE_LABELS,
    compute_multiples,
    run_comparables,
)
from src.valuation.data_bridge import (
    DataQualityError,
    data_quality_report,
    load_history,
    load_history_from_frame,
)
from src.valuation.dcf import cross_checks, reverse_dcf, run_dcf
from src.valuation.fcff import build_fcff
from src.valuation.forecasting import build_forecast
from src.valuation.market_data import fetch_price_history, fetch_snapshot, risk_free_rate
from src.valuation.monte_carlo import run_monte_carlo
from src.valuation.global_data import fetch as fetch_global
from src.valuation.global_search import search as search_global
from src.valuation.scenarios import run_all_scenarios
from src.valuation.universe import INDIA_MARKET, NIFTY_UNIVERSE, peers_for, resolve_market_for_currency
from src.valuation.wacc import build_wacc, validate_wacc

st.set_page_config(page_title="Institutional-Style Valuation Engine", page_icon="📐", layout="wide")

NIFTY_DATA = Path(__file__).parent / "data" / "nifty"


def pct(v, dp=1):
    return "n/a" if v is None or pd.isna(v) else f"{v:.{dp}%}"


def money(v, dp=0):
    return "n/a" if v is None or pd.isna(v) else f"{v:,.{dp}f}"


def fmt_x(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.2f}x"


@st.cache_data(ttl=3600, show_spinner=False)
def load_peer_multiples(full_ticker: str):
    """One peer's multiples, or None if its data is not usable. Cached per peer, since
    peers repeat across many targets in the same sector and each involves live fetches."""
    ticker = full_ticker.replace(".NS", "")
    name, sector = NIFTY_UNIVERSE.get(full_ticker, (ticker, ""))
    try:
        hist = load_history(ticker, NIFTY_DATA)
        snap = fetch_snapshot(full_ticker)
        return compute_multiples(
            ticker, name, sector, hist,
            snap.share_price, snap.shares_outstanding / 1e6, snap.market_cap / 1e6,
        )
    except Exception:  # noqa: BLE001 - a bad peer must not stop the run
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def run_pipeline(full_ticker: str, horizon: int = 5):
    """Stages 1 through 7 for one company. Cached 30 minutes: market data moves, the
    underlying financial statements do not, on any timescale that matters here."""
    ticker = full_ticker.replace(".NS", "")
    name, sector = NIFTY_UNIVERSE[full_ticker]

    hist = load_history(ticker, NIFTY_DATA)
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
    peers = [p for p in (load_peer_multiples(t) for t in peer_tickers) if p is not None]
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


@st.cache_data(ttl=1800, show_spinner=False)
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

st.title("Institutional-Style Valuation Engine")
st.caption(
    "Historical assumptions → forecast → FCFF → WACC → DCF, with comparable-company "
    "multiples and full sector context for a curated Nifty universe, or a quick DCF for "
    "any listed company. Built progressively, stage by stage, with every assumption shown "
    "alongside its reasoning."
)

mode = st.radio(
    "Mode", ["Curated Nifty universe (full workflow)", "Quick DCF (any company)"],
    horizontal=True, label_visibility="collapsed",
)

if mode.startswith("Curated"):
    labels = {f"{name} ({t.replace('.NS','')})": t for t, (name, sec) in sorted(NIFTY_UNIVERSE.items())}
    choice = st.selectbox("Company", list(labels.keys()), index=list(labels.keys()).index(
        next(k for k, v in labels.items() if v == "RELIANCE.NS")))
    full_ticker = labels[choice]

    with st.spinner(f"Running the valuation pipeline for {full_ticker.replace('.NS','')}..."):
        try:
            r = run_pipeline(full_ticker)
        except Exception as exc:  # noqa: BLE001 - surface the reason, not a stack trace
            st.error(f"Could not value this company: {exc}")
            st.stop()
else:
    st.info(
        "**Quick DCF**: any listed company, fetched live. There is no comparable-company "
        "cross-check here, only for the curated universe on the other tab, because real "
        "sector peers do not exist for an arbitrary company the way they do for the "
        "hand-picked 33 (see the README's Calibration section for why that peer set "
        "matters). Market assumptions (risk-free rate, equity risk premium, terminal "
        "growth ceiling) are only calibrated for USD and INR companies; anything else uses "
        "a disclosed generic fallback, which is a real simplification, not a footnote."
    )
    query = st.text_input("Company name or ticker", placeholder="Apple, ASML, SAP, Shell, Nestle")
    if not query:
        st.stop()
    with st.spinner(f"Fetching and valuing '{query}'..."):
        try:
            r = run_quick_pipeline(query)
        except Exception as exc:  # noqa: BLE001 - surface the reason, not a stack trace
            st.error(f"Could not value this company: {exc}")
            st.stop()

    if not r["market_calibrated"]:
        st.warning(
            f"**{r['currency']} is not a calibrated market in this engine.** The risk-free "
            f"rate, equity risk premium and terminal-growth ceiling use a generic "
            f"developed-market fallback rather than {r['currency']}-specific data, which "
            "misprices the discount rate by roughly the currency's own rate differential "
            "against the US. Treat this DCF as indicative only, more so than the "
            "already-documented DCF limitations that apply everywhere in this engine."
        )
    for note in r.get("fetch_notes", []):
        st.caption(f"ℹ️ {note}")

cur = r["currency"]
sector_label = f"  ·  {r['sector']}" if r["sector"] else ""
st.markdown(f"### {r['name']}  ·  `{r['ticker']}`{sector_label}")
top = st.columns(5)
top[0].metric("Current price", f"{cur} {r['share_price']:,.2f}")
top[1].metric("Market cap", f"{cur} {r['market_cap']/1e12:,.2f} tn")
top[2].metric("History", f"{r['quality']['years']} years")
top[3].metric("WACC", pct(r["wacc"].wacc))
top[4].metric("Beta (adj.)", f"{r['beta'].adjusted:.2f}")

if r["quality"]["warnings"]:
    with st.expander("Data quality notes", expanded=False):
        for w in r["quality"]["warnings"]:
            st.markdown(f"- {w}")

st.divider()

# --- Calibration banner: never let the DCF stand alone -------------------------------------
st.warning(
    "**Read the DCF and comparables together, not the DCF alone.** The DCF still reads below "
    "market across this universe (median -59.9%, against comparables at +2.9% on the same "
    "statements). A country-risk double count in the cost of equity was found and fixed, "
    "which closed about 15 points of that gap; the rest sits mostly in capital-intensive "
    "companies earning less on capital than their cost of capital, where a DCF is arguably "
    "the wrong instrument, and in richly-rated consumer franchises. The remaining gap is "
    "reported rather than tuned away. Full evidence and methodology: "
    "[README, Calibration section](https://github.com/kaotekwarrushikesh-dotcom/valuation-engine"
    "#calibration-the-model-read-low-what-was-wrong-and-what-still-is)."
)

tabs = st.tabs(["Valuation summary", "Historical & assumptions", "Forecast & FCFF", "WACC",
                "Scenarios", "Monte Carlo", "Methodology"])

# --- Tab 1: Valuation summary ----------------------------------------------------------------
with tabs[0]:
    bl = r["blended"]
    if bl.usable:
        st.subheader("Across methods")
        b1, b2, b3 = st.columns(3)
        b1.metric("Range low", f"{cur} {bl.low:,.2f}")
        b2.metric("Central", f"{cur} {bl.central:,.2f}",
                  None if bl.methods_disagree else f"{bl.upside:+.1%}")
        b3.metric("Range high", f"{cur} {bl.high:,.2f}")
        (st.warning if bl.methods_disagree else st.success)(bl.verdict)
        for note in bl.notes:
            st.caption(f"ℹ️ {note}")
        st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("DCF")
        if r["dcf_error"]:
            st.error(r["dcf_error"])
        elif r["dcf"] is None:
            st.error("WACC failed validation; no DCF produced. " + "; ".join(r["wacc_errors"]))
        else:
            d = r["dcf"]
            st.metric("Implied share price", f"{cur} {d.implied_share_price_floored:,.2f}",
                     f"{d.upside:+.1%} vs current")
            st.caption(f"Terminal value is {d.terminal_share:.0%} of enterprise value  ·  "
                      f"terminal growth {d.terminal_growth:.2%}  ·  WACC {d.wacc:.2%}")
            if r["implied_wacc"] is not None and not pd.isna(r["implied_wacc"]):
                st.caption(
                    f"Reverse DCF: the market price implies a **{r['implied_wacc']:.2%}** "
                    f"discount rate, against this model's **{d.wacc:.2%}**. The gap is a "
                    "question about the growth/terminal assumptions, not a verdict."
                )
            for note in (r["dcf"].notes + r["dcf"].warnings)[:4]:
                st.caption(f"⚠️ {note}")

    with right:
        st.subheader("Comparable companies")
        if r["mode"] == "quick":
            st.info(
                "Not available in Quick DCF mode. Real sector peers only exist for the "
                "curated Nifty universe; an auto-discovered peer group for an arbitrary "
                "company would be lower quality without saying so, which defeats the point "
                "of comparables as a cross-check. Switch to the curated universe tab for a "
                "company with a real comparable valuation."
            )
        elif r["comparables"] is None:
            st.info(
                f"Fewer than {MIN_PEERS} usable sector peers for {r['sector']} in this "
                "universe, so no comparable valuation is produced rather than one built off "
                "a peer group too thin to mean anything."
            )
        else:
            c = r["comparables"]
            st.metric("Blended comparable value", f"{cur} {c.blended_share_price:,.2f}",
                     f"{c.upside:+.1%} vs current")
            st.caption("Weighted across: " + ", ".join(MULTIPLE_LABELS[m] for m in c.blended_multiples_used))
            rows = []
            for m in MULTIPLE_LABELS:
                if m in c.implied:
                    rows.append({"Multiple": MULTIPLE_LABELS[m],
                                "Peer median": fmt_x(c.stats[m].median),
                                "Implied price": money(c.implied[m].implied_share_price, 2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()
    if r["scenarios"] and all(r["scenarios"][k].dcf is not None for k in ("bear", "base", "bull")):
        st.subheader("Bear / base / bull (DCF)")
        s = r["scenarios"]
        cols = st.columns(3)
        for col, key, label in zip(cols, ["bear", "base", "bull"], ["Bear", "Base", "Bull"]):
            price = s[key].dcf.implied_share_price_floored
            col.metric(label, f"{cur} {price:,.2f}", f"{s[key].dcf.upside:+.1%}")

# --- Tab 2: Historical & assumptions ----------------------------------------------------------
with tabs[1]:
    st.subheader("What the history says")
    for line in historical.summary_lines(r["analysis"]):
        st.markdown(f"- {line}")

    st.subheader("Forecast assumptions, and why")
    order = ["revenue_growth_start", "revenue_growth_path", "ebitda_margin", "da_pct_revenue",
             "tax_rate", "capex_pct_revenue", "capex_pct_revenue_path", "nwc_pct_revenue",
             "terminal_growth"]
    for name in order:
        item = r["assumptions"].detail.get(name)
        if not item:
            continue
        tag = " `[OVERRIDE]`" if item.source == "override" else ""
        st.markdown(f"**{name.replace('_',' ')}**: {item.value:.4g}  "
                   f"({item.confidence} confidence){tag}")
        st.caption(item.rationale)

    st.subheader("Historical financials")
    view_cols = ["fiscal_year", "revenue", "ebitda", "ebit", "net_income", "cfo", "capex",
                "net_debt"]
    hv = r["hist"][view_cols].copy()
    hv.columns = ["FY", "Revenue", "EBITDA", "EBIT", "Net income", "CFO", "Capex", "Net debt"]
    st.dataframe(hv, hide_index=True, use_container_width=True)

# --- Tab 3: Forecast & FCFF --------------------------------------------------------------------
with tabs[2]:
    st.subheader("Forecast income statement")
    f = r["fcff"].frame
    inc = pd.DataFrame({
        "Year": [f"{int(y)}E" for y in f["year"]],
        "Revenue": f["revenue"].map(lambda v: f"{v:,.0f}"),
        "Growth": f["revenue_growth"].map(pct),
        "EBITDA": f["ebitda"].map(lambda v: f"{v:,.0f}"),
        "Margin": f["ebitda_margin"].map(pct),
        "EBIT": f["ebit"].map(lambda v: f"{v:,.0f}"),
    })
    st.dataframe(inc, hide_index=True, use_container_width=True)

    st.subheader("Free cash flow to the firm")
    st.caption("Tax is charged on unlevered EBIT and interest never appears: the financing "
              "effect belongs in WACC, and counting it here too would value the debt tax "
              "shield twice.")
    fcff_view = pd.DataFrame({
        "Year": [f"{int(y)}E" for y in f["year"]],
        "NOPAT": f["nopat"].map(lambda v: f"{v:,.0f}"),
        "D&A": f["dep_amort"].map(lambda v: f"{v:,.0f}"),
        "Capex": f["capex"].map(lambda v: f"-{v:,.0f}"),
        "Δ NWC": f["change_in_nwc"].map(lambda v: f"{v:,.0f}"),
        "FCFF": f["fcff"].map(lambda v: f"{v:,.0f}"),
    })
    st.dataframe(fcff_view, hide_index=True, use_container_width=True)
    for note in r["fcff"].warnings:
        st.caption(f"⚠️ {note}")

# --- Tab 4: WACC --------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Cost of capital")
    st.dataframe(r["wacc"].component_table(), hide_index=True, use_container_width=True)
    st.caption(f"Risk-free from FRED {r['risk_free_series']} as at {r['rf_date']}. "
              "Equity at market value, debt at book: book equity is an accounting residual "
              "that can be negative after buybacks, which would make the weights meaningless.")
    for n in r["wacc"].notes:
        st.markdown(f"- {n}")
    for w in r["wacc"].warnings:
        st.markdown(f"- ⚠️ {w}")

    st.subheader("Beta")
    b = r["beta"]
    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("Raw beta", f"{b.raw:.3f}")
    bc2.metric("Adjusted (Blume)", f"{b.adjusted:.3f}")
    bc3.metric("R-squared", f"{b.r_squared:.2f}")
    bc4.metric("Confidence", b.confidence)
    st.caption(f"Regression of monthly returns against {r['index_name']}, "
              f"{b.observations} observations.")
    for w in b.warnings:
        st.caption(f"⚠️ {w}")

# --- Tab 5: Scenarios --------------------------------------------------------------------------
with tabs[4]:
    if not r["scenarios"]:
        st.info("WACC failed validation, so no scenarios could be built.")
    else:
        s = r["scenarios"]
        for key, label in [("bear", "Bear"), ("base", "Base"), ("bull", "Bull")]:
            sc = s[key]
            st.markdown(f"#### {label}")
            st.caption(sc.adjustments.rationale)
            if sc.dcf is None:
                st.error(f"Not usable: {sc.error}")
                continue
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue growth (yr 1)", pct(sc.assumptions.revenue_growth_path[0]))
            c2.metric("EBITDA margin", pct(sc.assumptions.ebitda_margin_path[0]))
            c3.metric("WACC", pct(sc.wacc))
            c4.metric("Implied price", f"{cur} {sc.dcf.implied_share_price_floored:,.2f}",
                     f"{sc.dcf.upside:+.1%}")
            if sc.dcf.equity_value <= 0:
                st.caption("Equity value is not positive on these assumptions: net debt "
                          "exceeds enterprise value, floored at zero rather than shown "
                          "negative, since equity cannot trade below zero.")

# --- Tab 6: Monte Carlo ------------------------------------------------------------------------
with tabs[5]:
    mc = r["monte_carlo"]
    if mc is None:
        st.info("WACC failed validation, so no Monte Carlo could be run.")
    else:
        st.subheader("Driver distributions")
        st.caption(
            "Every width below is measured from the company's own record, except terminal "
            "growth, which describes a period that has not happened and is drawn uniformly "
            "between the inflation floor and the GDP ceiling rather than given an invented "
            "mean and variance. A Monte Carlo with invented variances is the modeller's guess "
            "with error bars drawn on it."
        )
        st.dataframe(mc.distributions.table(), width="stretch", hide_index=True)
        for note in mc.distributions.notes:
            st.caption(f"ℹ️ {note}")

        st.subheader("Distribution of fair values")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("5th percentile", f"{cur} {mc.percentile(5):,.2f}")
        m2.metric("Median", f"{cur} {mc.median:,.2f}")
        m3.metric("95th percentile", f"{cur} {mc.percentile(95):,.2f}")
        m4.metric("Above market", f"{mc.probability_above_market:.0%}")

        clipped = mc.prices[mc.prices <= mc.percentile(99)]
        counts, edges = np.histogram(clipped, bins=40)
        midpoints = (edges[:-1] + edges[1:]) / 2
        st.bar_chart(
            pd.DataFrame({"trials": counts}, index=pd.Index(midpoints.round(0), name=f"{cur} per share")),
            y="trials",
        )
        st.caption(
            f"Histogram truncated at the 99th percentile for readability. "
            f"Completed {mc.completed} of {mc.trials} trials; {mc.failure_rate:.1%} failed, "
            "where the drawn terminal growth met or exceeded the drawn WACC and the perpetuity "
            "has no finite value. Failed trials are counted rather than dropped, since "
            "discarding them would remove exactly the bad draws and bias the distribution upward."
        )
        st.caption(
            f"**{mc.probability_above_market:.0%} of trials value the company above its current "
            "price.** This is what a Monte Carlo is for: a point estimate says more or less, "
            "this says how often, across the plausible range of the company's own inputs."
        )
        if mc.completed and (mc.percentile(95) - mc.median) > (mc.median - mc.percentile(5)) * 1.5:
            st.caption(
                "⚠️ The upper tail is much longer than the lower one. That is the Gordon-growth "
                "denominator: as drawn terminal growth approaches the drawn WACC, the spread "
                "between them shrinks toward zero and the terminal value rises without bound. "
                "The mean is a poor summary here; the percentiles are the honest reading."
            )

# --- Tab 7: Methodology --------------------------------------------------------------------------
with tabs[6]:
    st.markdown("""
This app runs Stages 1 to 7 of the valuation engine live for the selected company:

1. **Historical read & assumptions**: every forecast input is derived from the company's
   own history by a stated rule, never hard-coded, with a confidence level attached.
2. **Forecast & FCFF**: unlevered free cash flow to the firm, taxed on EBIT with interest
   never appearing, so the debt tax shield is never counted twice against WACC.
3. **WACC**: CAPM cost of equity with beta from live regression, cost of debt from the
   statements, weights at market value of equity and book value of debt.
4. **DCF**: discounted cash flow with terminal value made internally consistent
   (reinvestment tied to terminal growth via ROIC), and a reverse-DCF cross-check.
5. **Comparables**: peer sector multiples (P/E, P/B, EV/EBITDA, EV/Revenue), with a minimum
   peer-group size and outlier flagging.
6. **Scenarios**: bear/base/bull, each an independently recomputed valuation, not a flat
   haircut on the base case.
7. **Monte Carlo & blended summary**: the same drivers as distributions rather than three
   chosen points, with every width measured from the company's own record, and a final
   range across independent methods that refuses to average them when they disagree.

**Full source, every test, and the complete methodology and known limitations are in the
repository:** [github.com/kaotekwarrushikesh-dotcom/valuation-engine](https://github.com/kaotekwarrushikesh-dotcom/valuation-engine)

**This is a live, working model, not a finished verdict.** The Calibration section of the
README documents what happened when the DCF was found to read systematically below market:
a country-risk double count in the cost of equity was found and fixed, and the gap that
remained after it was reported rather than tuned away. Reading the DCF and comparables
together, with the caveats above, is the intended use of this tool, not picking whichever
number looks more decisive.
""")
