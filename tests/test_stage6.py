"""Tests for Stage 6: scenarios and sensitivity.

These target the places a scenario engine goes quietly wrong: bear coming out above bull,
overrides silently not taking effect, a grid that does not move in the direction its own
inputs imply, and the specific edge case of negative equity value, which a levered or
declining company can genuinely produce and which must never be presented as a literal
negative share price.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from valuation_engine import assumptions as asmp
from valuation_engine import historical
from valuation_engine.beta import BetaEstimate
from valuation_engine.dcf import run_dcf
from valuation_engine.fcff import build_fcff
from valuation_engine.forecasting import build_forecast
from valuation_engine.scenarios import BASE, BEAR, BULL, build_scenario, run_all_scenarios
from valuation_engine.sensitivity import driver_sensitivity, growth_margin_grid, wacc_terminal_growth_grid
from valuation_engine.wacc import build_wacc
from tests.test_stage1 import make_history

NOMINAL_GDP = 0.09
INFLATION = 0.04


def setup(years=8, growth=0.08, ebitda_margin=0.25):
    hist = make_history(years=years, growth=growth, ebitda_margin=ebitda_margin)
    analysis = historical.analyse(hist)
    base = asmp.derive(hist, analysis, "TEST", nominal_gdp_growth=NOMINAL_GDP, inflation=INFLATION)
    return hist, analysis, base


def run_scenario(hist, analysis, base, adj, wacc=0.13, net_debt=200.0, shares=10.0, price=100.0):
    return build_scenario(
        hist, analysis, "TEST", base, wacc, adj, net_debt, shares, price, "INR",
        NOMINAL_GDP, INFLATION, base.horizon,
    )


# --- Scenario ordering and independence --------------------------------------------------

def test_bear_base_bull_are_ordered_correctly():
    hist, analysis, base = setup()
    wacc = 0.13
    bear = run_scenario(hist, analysis, base, BEAR, wacc=wacc)
    basec = run_scenario(hist, analysis, base, BASE, wacc=wacc)
    bull = run_scenario(hist, analysis, base, BULL, wacc=wacc)

    assert bear.dcf is not None and basec.dcf is not None and bull.dcf is not None
    assert bear.dcf.implied_share_price < basec.dcf.implied_share_price < bull.dcf.implied_share_price


def test_base_scenario_reproduces_the_unmodified_assumptions():
    """BASE has zero deltas, so it must recompute the same forecast as calling the pipeline
    directly, not merely a similar one."""
    hist, analysis, base = setup()
    fc = build_forecast(hist, base, "TEST")
    fcff = build_fcff(fc)

    basec = run_scenario(hist, analysis, base, BASE, wacc=0.13)

    assert basec.fcff.frame["fcff"].tolist() == pytest.approx(fcff.frame["fcff"].tolist())
    assert basec.assumptions.terminal_growth == pytest.approx(base.terminal_growth)


def test_each_scenario_recomputes_the_whole_chain_not_a_flat_haircut():
    """A flat percentage applied to the base price would move every driver by the same
    implied amount; an independently rebuilt scenario should not produce that."""
    hist, analysis, base = setup()
    bear = run_scenario(hist, analysis, base, BEAR, wacc=0.13)
    basec = run_scenario(hist, analysis, base, BASE, wacc=0.13)

    assert bear.assumptions.revenue_growth_path[0] != basec.assumptions.revenue_growth_path[0]
    assert bear.assumptions.ebitda_margin_path[0] != basec.assumptions.ebitda_margin_path[0]
    assert bear.wacc != basec.wacc
    # The ratio of bear to base FCFF should not be uniform across years, since margin and
    # growth shocks compound differently year to year.
    ratios = [b / s for b, s in zip(bear.fcff.frame["fcff"], basec.fcff.frame["fcff"])]
    assert len(set(round(r, 6) for r in ratios)) > 1


def test_run_all_scenarios_returns_all_three():
    hist, analysis, base = setup()
    result = run_all_scenarios(hist, analysis, "TEST", base, 0.13, 200.0, 10.0, 100.0, "INR",
                               NOMINAL_GDP, INFLATION, base.horizon)
    assert set(result) == {"bear", "base", "bull"}
    assert all(r.dcf is not None for r in result.values())


# --- Negative equity handling ------------------------------------------------------------

def test_negative_equity_value_floors_the_displayed_price_at_zero():
    """A heavily levered or declining company can genuinely produce negative equity value;
    it must read as worthless, not as a literal negative price no equity can trade at."""
    hist, analysis, base = setup(growth=-0.05, ebitda_margin=0.08)
    result = run_scenario(hist, analysis, base, BASE, wacc=0.13, net_debt=50_000.0)

    assert result.dcf is not None
    assert result.dcf.equity_value < 0
    assert result.dcf.implied_share_price < 0  # raw arithmetic stays honest
    assert result.dcf.implied_share_price_floored == 0.0  # display floors at zero


def test_upside_is_computed_off_the_floored_price():
    """Downside is capped at -100%: an investor cannot lose more than the amount invested,
    so upside must not read past -100% just because the raw arithmetic went further negative."""
    hist, analysis, base = setup(growth=-0.05, ebitda_margin=0.08)
    result = run_scenario(hist, analysis, base, BASE, wacc=0.13, net_debt=50_000.0)
    assert result.dcf.upside == pytest.approx(-1.0)


def test_positive_equity_value_is_unaffected_by_flooring():
    hist, analysis, base = setup()
    result = run_scenario(hist, analysis, base, BASE, wacc=0.13)
    assert result.dcf.equity_value > 0
    assert result.dcf.implied_share_price_floored == pytest.approx(result.dcf.implied_share_price)


def test_bear_terminal_growth_cannot_go_negative():
    """The scenario override bypasses assumptions.derive's inflation floor on purpose (a
    stress test may need to go below it), but a terminal contraction to nothing is a
    different company, not a stress test, so zero is an absolute floor."""
    hist, analysis, base = setup(growth=0.001, ebitda_margin=0.15)  # already near the floor
    bear = run_scenario(hist, analysis, base, BEAR, wacc=0.13)
    assert bear.assumptions.terminal_growth >= 0.0


# --- Sensitivity grids -------------------------------------------------------------------

def test_wacc_terminal_growth_grid_is_monotonic():
    """Higher WACC must discount harder (lower price); higher terminal growth must lift the
    terminal value (higher price). A grid that violates either direction is broken."""
    hist, analysis, base = setup()
    fc = build_forecast(hist, base, "TEST")
    fcff = build_fcff(fc)

    grid = wacc_terminal_growth_grid(fcff, 0.13, base.terminal_growth, base.terminal_roic,
                                     200.0, 10.0, wacc_steps=3, growth_steps=3)

    # Down each column (increasing WACC), price should fall.
    for col in grid.columns:
        values = grid[col].dropna().tolist()
        assert values == sorted(values, reverse=True)

    # Across each row (increasing terminal growth), price should rise.
    for _, row in grid.iterrows():
        values = row.dropna().tolist()
        assert values == sorted(values)


def test_growth_margin_grid_is_monotonic():
    hist, analysis, base = setup()
    grid = growth_margin_grid(hist, analysis, "TEST", base, 0.13, 200.0, 10.0, "INR",
                              NOMINAL_GDP, INFLATION, base.horizon,
                              growth_steps=3, margin_steps=3)

    for _, row in grid.iterrows():
        values = row.dropna().tolist()
        assert values == sorted(values)  # higher margin -> higher price, left to right


def test_grid_holds_wacc_fixed_for_growth_margin_but_not_for_wacc_grid():
    """The two grids isolate different things: growth/margin holds WACC at the base case,
    while the WACC grid is defined by varying it."""
    hist, analysis, base = setup()
    grid = growth_margin_grid(hist, analysis, "TEST", base, 0.20, 200.0, 10.0, "INR",
                              NOMINAL_GDP, INFLATION, base.horizon,
                              growth_steps=2, margin_steps=2)
    # A punitively high fixed WACC should produce much lower prices than the base-WACC run.
    base_grid = growth_margin_grid(hist, analysis, "TEST", base, 0.10, 200.0, 10.0, "INR",
                                   NOMINAL_GDP, INFLATION, base.horizon,
                                   growth_steps=2, margin_steps=2)
    assert grid.to_numpy().mean() < base_grid.to_numpy().mean()


# --- Driver sensitivity --------------------------------------------------------------------

def test_driver_sensitivity_ranks_by_absolute_impact():
    hist, analysis, base = setup()
    fc = build_forecast(hist, base, "TEST")
    fcff = build_fcff(fc)
    dcf = run_dcf("TEST", fcff, 0.13, base.terminal_growth, 200.0, 10.0, 100.0, "INR",
                  roic=base.terminal_roic)

    driv = driver_sensitivity(hist, analysis, "TEST", base, 0.13, dcf.implied_share_price_floored,
                              200.0, 10.0, "INR", NOMINAL_GDP, INFLATION, base.horizon)

    impacts = driv["impact_pct"].abs().tolist()
    assert impacts == sorted(impacts, reverse=True)


def test_driver_sensitivity_reports_na_when_base_is_worthless():
    hist, analysis, base = setup(growth=-0.05, ebitda_margin=0.08)
    fc = build_forecast(hist, base, "TEST")
    fcff = build_fcff(fc)
    dcf = run_dcf("TEST", fcff, 0.13, base.terminal_growth, 50_000.0, 10.0, 100.0, "INR",
                  roic=base.terminal_roic)
    assert dcf.implied_share_price_floored == 0.0

    driv = driver_sensitivity(hist, analysis, "TEST", base, 0.13, dcf.implied_share_price_floored,
                              50_000.0, 10.0, "INR", NOMINAL_GDP, INFLATION, base.horizon)
    assert driv["impact_pct"].isna().all()
