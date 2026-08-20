"""Tests for Stage 2: the forecast and the FCFF build.

A valuation model that runs is not a valuation model that is right. These check the
identities independently of the code that produced them, and pin the conventions that are
easy to get wrong in a way that still looks plausible on the page.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from valuation_engine import assumptions as asmp
from valuation_engine import historical
from valuation_engine.fcff import build_fcff, statement_frame, validate_fcff
from valuation_engine.forecasting import build_forecast
from tests.test_stage1 import make_history


def build(years=8, growth=0.08, **kw):
    hist = make_history(years=years, growth=growth, **kw)
    a = asmp.derive(hist, historical.analyse(hist), "TEST")
    fc = build_forecast(hist, a, "TEST")
    return hist, a, fc, build_fcff(fc)


# --- Forecast --------------------------------------------------------------------------

def test_revenue_compounds_off_the_last_actual_year():
    hist, a, fc, _ = build()
    f = fc.frame
    assert f["revenue"].iloc[0] == pytest.approx(fc.base_revenue * (1 + a.revenue_growth_path[0]))
    for i in range(1, len(f)):
        assert f["revenue"].iloc[i] == pytest.approx(
            f["revenue"].iloc[i - 1] * (1 + a.revenue_growth_path[i]))


def test_ebit_is_derived_from_ebitda_less_da_not_forecast_separately():
    _, _, fc, _ = build()
    f = fc.frame
    assert ((f["ebitda"] - f["dep_amort"] - f["ebit"]).abs() < 1e-9).all()


def test_forecast_years_follow_the_base_year_consecutively():
    _, _, fc, _ = build()
    years = fc.frame["year"].tolist()
    assert years[0] == fc.base_year + 1
    assert years == list(range(years[0], years[0] + len(years)))


def test_horizon_is_configurable():
    hist = make_history(years=8)
    a = asmp.derive(hist, historical.analyse(hist), "TEST", horizon=7)
    assert len(build_forecast(hist, a).frame) == 7


# --- FCFF identities -------------------------------------------------------------------

def test_fcff_reconciles_to_its_components():
    _, _, _, r = build()
    f = r.frame
    rebuilt = f["nopat"] + f["dep_amort"] - f["capex"] - f["change_in_nwc"]
    assert ((rebuilt - f["fcff"]).abs() < 1e-9).all()


def test_nopat_is_ebit_less_tax_on_ebit():
    _, a, _, r = build()
    f = r.frame
    assert ((f["ebit"] * a.tax_rate - f["tax_on_ebit"]).abs() < 1e-9).all()
    assert ((f["ebit"] - f["tax_on_ebit"] - f["nopat"]).abs() < 1e-9).all()


def test_tax_is_computed_on_unlevered_ebit_not_after_interest():
    """The central FCFF trap. Taxing pretax income after interest would import the debt
    tax shield into the cash flow, while WACC also puts it in the discount rate, valuing
    the same shield twice and always flattering the answer."""
    hist, a, _, r = build()
    f = r.frame

    interest = float(hist["interest_expense"].iloc[-1])
    assert interest > 0, "fixture must carry interest for this test to mean anything"

    levered_tax = (f["ebit"] - interest) * a.tax_rate
    assert (f["tax_on_ebit"] > levered_tax).all()
    assert "interest_expense" not in f.columns


def test_da_is_added_back_because_it_is_not_cash():
    _, _, _, r = build()
    f = r.frame
    without = f["nopat"] - f["capex"] - f["change_in_nwc"]
    assert ((f["fcff"] - without - f["dep_amort"]).abs() < 1e-9).all()


def test_change_in_working_capital_uses_the_increment_not_the_level():
    """Only the movement consumes cash. Using the level would charge the company every
    year for a balance it merely continues to carry."""
    _, a, fc, r = build()
    f = r.frame
    expected = fc.frame["revenue_increment"] * a.nwc_pct_revenue
    assert ((f["change_in_nwc"] - expected).abs() < 1e-9).all()
    assert (f["change_in_nwc"].abs() < (f["revenue"] * abs(a.nwc_pct_revenue)).abs()).all()


def test_negative_working_capital_releases_cash_as_revenue_grows():
    hist = make_history(years=8)
    # Suppliers fund the business: operating liabilities exceed operating assets.
    hist["nwc"] = -0.10 * hist["revenue"]
    hist["nwc_pct_revenue"] = hist["nwc"] / hist["revenue"]

    a = asmp.derive(hist, historical.analyse(hist), "NEG")
    r = build_fcff(build_forecast(hist, a, "NEG"))

    assert a.nwc_pct_revenue < 0
    assert (r.frame["change_in_nwc"] < 0).all()
    # A cash release lifts FCFF above NOPAT plus D&A less capex.
    base = r.frame["nopat"] + r.frame["dep_amort"] - r.frame["capex"]
    assert (r.frame["fcff"] > base).all()


def test_first_forecast_year_has_no_phantom_working_capital_step():
    """Bridging from the last reported balance to a ratio-implied one would create a
    one-off cash flow reflecting the ratio choice rather than the business."""
    hist = make_history(years=10)
    hist.loc[hist.index[-3:], "nwc"] = -0.10 * hist.loc[hist.index[-3:], "revenue"]
    hist["nwc_pct_revenue"] = hist["nwc"] / hist["revenue"]

    a = asmp.derive(hist, historical.analyse(hist), "STEP")
    r = build_fcff(build_forecast(hist, a, "STEP"))

    first, second = r.frame["change_in_nwc"].iloc[0], r.frame["change_in_nwc"].iloc[1]
    assert abs(first) < abs(second) * 3, "first year is out of line with the rest"


# --- Validation and presentation -------------------------------------------------------

def test_validate_passes_on_a_clean_build():
    _, _, _, r = build()
    assert validate_fcff(r.frame) == []


def test_validate_catches_a_broken_identity():
    _, _, _, r = build()
    f = r.frame.copy()
    f.loc[0, "fcff"] = f.loc[0, "fcff"] * 1.5
    errors = validate_fcff(f)
    assert any("reconcile" in e for e in errors)


def test_validate_catches_non_consecutive_years():
    _, _, _, r = build()
    f = r.frame.copy()
    f.loc[2, "year"] = f.loc[2, "year"] + 5
    assert any("consecutive" in e for e in validate_fcff(f))


def test_statement_shows_outflows_as_negative():
    _, _, _, r = build()
    s = statement_frame(r)
    assert (s.loc["Tax on EBIT"] < 0).all()
    assert (s.loc["Capital expenditure"] < 0).all()
    assert (s.loc["EBIT"] > 0).all()
    # The statement column must still add up to the reported FCFF.
    parts = s.loc[["NOPAT", "Depreciation and amortisation", "Capital expenditure",
                   "Change in working capital"]].sum()
    assert ((parts - s.loc["FCFF"]).abs() < 1e-6).all()


def test_heavy_reinvestment_is_flagged_not_smoothed_away():
    hist = make_history(years=8)
    hist["capex"] = hist["revenue"] * 0.40
    hist["capex_pct_revenue"] = 0.40

    a = asmp.derive(hist, historical.analyse(hist), "HEAVY")
    r = build_fcff(build_forecast(hist, a, "HEAVY"))

    assert (r.frame["fcff"] < 0).any()
    assert r.warnings, "a cash-consuming forecast must say so"


# --- Currency resolution ---------------------------------------------------------------

class _FakeInfo(dict):
    pass


class _FakeTicker:
    """Stands in for a yfinance Ticker with a chosen declared currency and market cap."""

    def __init__(self, declared, market_cap):
        self.info = {"financialCurrency": declared}
        self.fast_info = _FakeInfo({"marketCap": market_cap})


def test_usd_reporter_is_converted_when_the_flag_is_right(monkeypatch):
    """Infosys genuinely reports in USD while trading in INR. Left unconverted, its cash
    flows would be paired with a rupee share price and the valuation divided by the rate."""
    from valuation_engine import nse_data

    monkeypatch.setattr(nse_data, "fx_to_inr", lambda c: 95.0)
    # Revenue 20bn USD against a market cap of ~4.7tn INR.
    currency, rate = nse_data.resolve_currency(_FakeTicker("USD", 4.7e12), "INFY.NS", 20e9)
    assert currency == "USD"
    assert rate == 95.0


def test_usd_flag_is_ignored_when_the_statements_are_already_in_rupees(monkeypatch):
    """HCL Technologies carries a USD flag while reporting in INR. Believing it would
    multiply the company by the exchange rate and overstate it a hundredfold."""
    from valuation_engine import nse_data

    monkeypatch.setattr(nse_data, "fx_to_inr", lambda c: 95.0)
    # Revenue 1.3tn INR against a market cap of ~3.7tn INR: already consistent.
    currency, rate = nse_data.resolve_currency(_FakeTicker("USD", 3.7e12), "HCLTECH.NS", 1.3e12)
    assert currency == "INR"
    assert rate == 1.0


def test_inr_reporter_is_never_converted():
    from valuation_engine import nse_data

    currency, rate = nse_data.resolve_currency(_FakeTicker("INR", 1.0e12), "TCS.NS", 2.6e12)
    assert (currency, rate) == ("INR", 1.0)


def test_conversion_is_skipped_when_market_cap_is_unavailable(monkeypatch):
    """Without the cross-check there is no evidence, and converting on a guess would be a
    silent hundredfold error while leaving it alone is at worst the raw filing."""
    from valuation_engine import nse_data

    monkeypatch.setattr(nse_data, "fx_to_inr", lambda c: 95.0)
    currency, rate = nse_data.resolve_currency(_FakeTicker("USD", 0), "X.NS", 20e9)
    assert (currency, rate) == ("INR", 1.0)


def test_currency_scaling_leaves_ratios_untouched():
    """A constant scaling cancels in any ratio, which is why margins and growth are
    unaffected by the conversion and only absolute cash flows move."""
    hist = make_history(years=8)
    a1 = asmp.derive(hist, historical.analyse(hist), "A")

    scaled = hist.copy()
    money = ["revenue", "ebitda", "ebit", "dep_amort", "capex", "nwc", "net_income",
             "tax_expense", "cash", "current_assets", "current_liabilities", "total_debt",
             "equity", "cfo", "short_term_debt", "long_term_debt", "total_assets"]
    for col in money:
        if col in scaled.columns:
            scaled[col] = scaled[col] * 95.0
    a2 = asmp.derive(scaled, historical.analyse(scaled), "B")

    assert a2.revenue_growth_path == pytest.approx(a1.revenue_growth_path)
    assert a2.ebitda_margin_path == pytest.approx(a1.ebitda_margin_path)
    assert a2.capex_pct_revenue == pytest.approx(a1.capex_pct_revenue)


# --- Capex fade toward terminal-consistent reinvestment --------------------------------

def test_capex_fades_toward_terminal_consistent_level():
    """Holding capex flat at its historical share of revenue while growth fades down
    underneath it charges the company for reinvestment it is never credited with in the
    explicit years, not only the terminal one."""
    hist = make_history(years=8, growth=0.25, ebitda_margin=0.30)
    a = asmp.derive(hist, historical.analyse(hist), "FADE")
    path = a.capex_pct_revenue_path
    assert len(path) == a.horizon
    assert path[0] == pytest.approx(a.detail["capex_pct_revenue"].value, abs=1e-4)
    # A growth-fading company reinvesting heavily should see capex intensity move, not
    # stay pinned at the historical level for all five years.
    assert path[-1] != pytest.approx(path[0], abs=1e-4)


def test_capex_path_feeds_the_fcff_build_not_a_flat_ratio():
    """fcff.py must use the per-year path, not the starting-year scalar, or the fade is
    computed but silently discarded."""
    hist = make_history(years=8, growth=0.20)
    a = asmp.derive(hist, historical.analyse(hist), "USEPATH")
    fc = build_forecast(hist, a, "USEPATH")
    r = build_fcff(fc)
    implied_ratios = (r.frame["capex"] / r.frame["revenue"]).tolist()
    assert implied_ratios == pytest.approx(a.capex_pct_revenue_path, abs=1e-6)
    # If the fade were being discarded, every year would equal the starting ratio.
    assert len(set(round(x, 6) for x in implied_ratios)) > 1


def test_capex_pct_revenue_property_returns_the_starting_value():
    """Callers that want a single number (a summary table) get the level history actually
    supports, not an average across the fade or the terminal figure."""
    hist = make_history(years=8)
    a = asmp.derive(hist, historical.analyse(hist), "SCALAR")
    assert a.capex_pct_revenue == pytest.approx(a.capex_pct_revenue_path[0])


def test_terminal_roic_is_a_single_source_used_everywhere():
    """WACC-independent, computed once inside derive() from history and tax rate alone, so
    Stage 4 and the capex fade cannot silently disagree about what ROIC is."""
    hist = make_history(years=8)
    a = asmp.derive(hist, historical.analyse(hist), "ONESOURCE")
    from valuation_engine.terminal_value import estimate_roic
    assert a.terminal_roic == pytest.approx(estimate_roic(hist, a.tax_rate))


def test_terminal_consistent_capex_ratio_matches_the_reinvestment_identity():
    from valuation_engine.terminal_value import terminal_consistent_capex_ratio
    ebitda_margin, da_pct, nwc_pct, tax_rate, growth, roic = 0.25, 0.05, 0.02, 0.25, 0.06, 0.15
    ratio, _ = terminal_consistent_capex_ratio(ebitda_margin, da_pct, nwc_pct, tax_rate, growth, roic)

    ebit_margin = ebitda_margin - da_pct
    nopat_margin = ebit_margin * (1 - tax_rate)
    reinvestment_rate = growth / roic
    expected = da_pct + reinvestment_rate * nopat_margin - nwc_pct * growth
    assert ratio == pytest.approx(expected, abs=1e-9)


def test_terminal_consistent_capex_ratio_never_goes_negative():
    from valuation_engine.terminal_value import terminal_consistent_capex_ratio
    # Deliberately extreme: very low D&A, very low growth, so the naive formula would go
    # negative. Negative capex is not a real thing a company can do.
    ratio, _ = terminal_consistent_capex_ratio(
        ebitda_margin=0.10, da_pct_revenue=0.01, nwc_pct_revenue=0.30,
        tax_rate=0.25, growth=0.01, roic=0.30,
    )
    assert ratio >= 0.0


def test_reinvestment_rate_for_growth_caps_when_roic_at_or_below_growth():
    from valuation_engine.terminal_value import reinvestment_rate_for_growth
    rate, effective_growth = reinvestment_rate_for_growth(growth=0.08, roic=0.06)
    assert effective_growth < 0.08
    assert effective_growth == pytest.approx(0.05)
    assert 0 <= rate <= 1
