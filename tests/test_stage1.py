"""Tests for Stage 1: the data bridge, the historical read, and the assumption rules.

These target the cases where a wrong answer looks entirely reasonable on screen, because
those are the ones that reach a valuation without anyone noticing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from valuation_engine import assumptions as asmp
from valuation_engine import historical
from valuation_engine.data_bridge import data_quality_report, derive_valuation_inputs


def make_history(years=10, revenue0=1000.0, growth=0.08, ebitda_margin=0.30, **overrides):
    """A synthetic but internally consistent company."""
    rows = []
    revenue = revenue0
    for i in range(years):
        ebitda = revenue * ebitda_margin
        da = revenue * 0.05
        ebit = ebitda - da
        tax = ebit * 0.25
        rows.append({
            "fiscal_year": 2016 + i,
            "revenue": revenue,
            "ebitda": ebitda,
            "ebit": ebit,
            "dep_amort": da,
            "interest_expense": 10.0,
            "tax_expense": tax,
            "net_income": ebit - tax - 10.0,
            "cash": revenue * 0.10,
            "current_assets": revenue * 0.40,
            "current_liabilities": revenue * 0.25,
            "short_term_debt": revenue * 0.05,
            "long_term_debt": revenue * 0.20,
            "total_debt": revenue * 0.25,
            "equity": revenue * 0.50,
            "cfo": ebitda * 0.80,
            "capex": revenue * 0.06,
        })
        revenue *= 1 + growth

    df = pd.DataFrame(rows)
    for k, v in overrides.items():
        df[k] = v
    return derive_valuation_inputs(df)


# --- Data bridge -----------------------------------------------------------------------

def test_working_capital_excludes_cash_and_short_term_debt():
    """Cash and debt are financing. Leaving them in makes working capital move with the
    cash balance, which is an output of the model rather than an input to it."""
    h = make_history(years=3)
    row = h.iloc[0]
    expected = (row["current_assets"] - row["cash"]) - (row["current_liabilities"] - row["short_term_debt"])
    assert row["nwc"] == pytest.approx(expected)


def test_change_in_nwc_is_the_movement_not_the_level():
    h = make_history(years=4)
    assert pd.isna(h["change_in_nwc"].iloc[0])
    assert h["change_in_nwc"].iloc[1] == pytest.approx(h["nwc"].iloc[1] - h["nwc"].iloc[0])


def test_da_falls_back_to_ebitda_less_ebit():
    h = make_history(years=3)
    h_no_da = h.drop(columns=["dep_amort"])
    rebuilt = derive_valuation_inputs(h_no_da)
    assert rebuilt["dep_amort"].iloc[0] == pytest.approx(h["ebitda"].iloc[0] - h["ebit"].iloc[0])


def test_effective_tax_rate_uses_rebuilt_pretax_income():
    h = make_history(years=3)
    row = h.iloc[0]
    assert row["effective_tax_rate"] == pytest.approx(
        row["tax_expense"] / (row["net_income"] + row["tax_expense"])
    )


def test_quality_report_blocks_when_a_driver_is_absent():
    h = make_history(years=6)
    h["capex"] = np.nan
    h = derive_valuation_inputs(h)
    report = data_quality_report(h, "TEST")
    assert not report["usable"]
    assert any("capex" in b for b in report["blocking"])


# --- Historical read -------------------------------------------------------------------

def test_growth_classification_detects_deceleration():
    rows = []
    revenue = 1000.0
    for i, g in enumerate([0.20, 0.20, 0.18, 0.16, 0.05, 0.04, 0.03, 0.02]):
        rows.append({"fiscal_year": 2018 + i, "revenue": revenue})
        revenue *= 1 + g
    df = pd.DataFrame(rows)
    df["revenue_growth"] = df["revenue"].pct_change()
    assert historical.classify_growth(df).classification == "decelerating"


def test_cagr_undefined_for_non_positive_endpoints():
    assert np.isnan(historical.cagr(-100, 200, 3))
    assert np.isnan(historical.cagr(100, -200, 3))
    assert np.isnan(historical.cagr(0, 100, 3))


def test_cagr_matches_manual_calculation():
    assert historical.cagr(100, 200, 5) == pytest.approx(2 ** 0.2 - 1)


# --- Assumptions -----------------------------------------------------------------------

def test_terminal_growth_never_exceeds_the_gdp_ceiling():
    """A company compounding at 30% cannot do so in perpetuity."""
    h = make_history(years=8, growth=0.30)
    a = asmp.derive(h, historical.analyse(h), "FAST", nominal_gdp_growth=0.04)
    assert a.terminal_growth <= 0.04


def test_terminal_growth_never_exceeds_current_growth():
    """A slow-growing company must not be assumed to re-accelerate into perpetuity, since
    that quietly moves value into the terminal period on no evidence."""
    h = make_history(years=8, growth=0.015)
    a = asmp.derive(h, historical.analyse(h), "SLOW", nominal_gdp_growth=0.04, inflation=0.0)
    assert a.terminal_growth <= 0.015 + 1e-3
    assert a.revenue_growth_path == [pytest.approx(a.terminal_growth, abs=1e-3)] * a.horizon


def test_terminal_growth_is_floored_at_inflation():
    """The cap at current growth reads a short window as a perpetual rate. A few weak years
    are evidence about the cycle, not about the next century, and letting them set terminal
    growth below inflation claims the business shrinks in real terms forever, which is a
    much stronger claim than the caution it looks like."""
    h = make_history(years=8, growth=0.01)
    a = asmp.derive(h, historical.analyse(h), "WEAK", nominal_gdp_growth=0.09, inflation=0.04)
    assert a.terminal_growth == pytest.approx(0.04)
    assert "inflation" in a.detail["terminal_growth"].rationale.lower()


def test_inflation_floor_never_raises_growth_above_the_gdp_ceiling():
    """The floor must not become a back door around the ceiling."""
    h = make_history(years=8, growth=0.002)
    a = asmp.derive(h, historical.analyse(h), "TINY", nominal_gdp_growth=0.03, inflation=0.06)
    assert a.terminal_growth <= 0.06
    assert a.terminal_growth >= 0.03


def test_growth_path_fades_downward_and_lands_on_terminal():
    h = make_history(years=8, growth=0.20)
    a = asmp.derive(h, historical.analyse(h), "FADE", horizon=5, nominal_gdp_growth=0.04)
    assert a.revenue_growth_path == sorted(a.revenue_growth_path, reverse=True)
    assert a.revenue_growth_path[-1] == pytest.approx(a.terminal_growth, abs=1e-4)
    assert a.revenue_growth_path[0] <= 0.20 + 1e-9


def test_working_capital_ratio_anchors_on_recent_level_not_median():
    """The forecast bridges from the last reported balance, so a ratio drawn from a period
    the business has left behind creates a phantom first-year cash flow."""
    h = make_history(years=10)
    # Structural shift: working capital swings negative in the last three years.
    h.loc[h.index[-3:], "nwc"] = -0.10 * h.loc[h.index[-3:], "revenue"]
    h["nwc_pct_revenue"] = h["nwc"] / h["revenue"]

    a = asmp.derive(h, historical.analyse(h), "SHIFT")
    recent = h["nwc_pct_revenue"].tail(3).mean()
    assert a.nwc_pct_revenue == pytest.approx(recent, abs=1e-3)
    assert a.nwc_pct_revenue != pytest.approx(h["nwc_pct_revenue"].median(), abs=1e-3)


def test_margin_is_held_flat_even_when_history_is_expanding():
    """Extending margin expansion is the most common way a DCF is steered upward."""
    h = make_history(years=10)
    h["ebitda_margin"] = np.linspace(0.20, 0.40, len(h))
    a = asmp.derive(h, historical.analyse(h), "EXPAND")
    assert len(set(a.ebitda_margin_path)) == 1
    assert a.ebitda_margin_path[0] == pytest.approx(h["ebitda_margin"].tail(3).mean(), abs=1e-4)


def test_tax_rate_is_bounded():
    h = make_history(years=6)
    h["effective_tax_rate"] = 0.01  # implausible in perpetuity
    a = asmp.derive(h, historical.analyse(h), "LOWTAX")
    assert a.tax_rate >= 0.10


def test_overrides_are_recorded_as_overrides():
    """A hand-set assumption must never be mistakable for a derived one."""
    h = make_history(years=8)
    a = asmp.derive(h, historical.analyse(h), "OVR", overrides={"terminal_growth": 0.035})
    assert a.terminal_growth == 0.035
    assert a.detail["terminal_growth"].source == "override"
    assert "Set by hand" in a.detail["terminal_growth"].rationale


def test_every_assumption_carries_a_rationale():
    h = make_history(years=8)
    a = asmp.derive(h, historical.analyse(h), "DOC")
    assert a.detail
    for name, item in a.detail.items():
        assert item.rationale.strip(), f"{name} has no rationale"
        assert item.confidence in {"high", "medium", "low"}


def test_gdp_ceiling_travels_with_the_market():
    """Rupee cash flows need a rupee GDP ceiling; a 4% cap would understate Indian
    terminal value for reasons of geography rather than economics."""
    h = make_history(years=8, growth=0.30)
    india = asmp.derive(h, historical.analyse(h), "IN", nominal_gdp_growth=0.09)
    us = asmp.derive(h, historical.analyse(h), "US", nominal_gdp_growth=0.04)
    assert india.terminal_growth > us.terminal_growth
