"""Tests for Stage 7: Monte Carlo distributions and the blended summary.

The properties worth pinning here are the ones that would silently corrupt the output
rather than crash it: a Monte Carlo that drops its failed trials reads too optimistic, a
correlation that does not survive the Cholesky step erases the joint tail, and a blended
summary that averages disagreeing methods manufactures a fair value no method supports.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from valuation_engine.blended import WIDE_DISPERSION, build_blended
from valuation_engine.monte_carlo import (
    GROWTH_MARGIN_CORRELATION,
    MAX_GROWTH_SIGMA,
    MIN_GROWTH_SIGMA,
    MIN_MARGIN_SIGMA,
    _correlated_normals,
    build_distributions,
)


class FakeAssumptions:
    def __init__(self, growth=0.08, margin=0.25):
        self.revenue_growth_path = [growth]
        self.ebitda_margin_path = [margin]


def frame(growths, margins):
    return pd.DataFrame({"revenue_growth": growths, "ebitda_margin": margins})


# --- Driver distributions ------------------------------------------------------------

def test_dispersion_comes_from_the_companys_own_history():
    hist = frame([np.nan, 0.10, 0.04, 0.16], [0.20, 0.24, 0.22, 0.26])
    d = build_distributions(hist, FakeAssumptions(), 0.11, 0.15, 0.075, 0.9, 0.04, 0.09)
    assert d.growth_sigma == pytest.approx(float(pd.Series([0.10, 0.04, 0.16]).std()))
    assert d.margin_sigma == pytest.approx(float(pd.Series([0.20, 0.24, 0.22, 0.26]).std()))


def test_an_unusually_smooth_history_still_gets_a_floor():
    """A company with four near-identical years must not come out looking near-certain:
    that would be an artefact of a short window, not evidence of predictability."""
    hist = frame([0.0800, 0.0801, 0.0799, 0.0800], [0.2500, 0.2501, 0.2499, 0.2500])
    d = build_distributions(hist, FakeAssumptions(), 0.11, 0.15, 0.075, 0.9, 0.04, 0.09)
    assert d.growth_sigma == MIN_GROWTH_SIGMA
    assert d.margin_sigma == MIN_MARGIN_SIGMA


def test_one_distorted_year_cannot_set_an_unbounded_dispersion():
    hist = frame([0.05, 0.08, -0.60, 0.90], [0.20, 0.22, 0.05, 0.40])
    d = build_distributions(hist, FakeAssumptions(), 0.11, 0.15, 0.075, 0.9, 0.04, 0.09)
    assert d.growth_sigma == MAX_GROWTH_SIGMA
    assert any("clamped" in n for n in d.notes)


def test_a_clamp_too_small_to_see_is_not_announced():
    """A measured 0.99% clamped to the 1.0% floor displays as "1.0% clamped to 1.0%", which
    reads as a broken model rather than a rounding artefact."""
    # Four values alternating +/- d around a mean have standard deviation d * sqrt(4/3),
    # so this lands the measured dispersion just under the floor but at the same 0.1%.
    d = 0.0099 / np.sqrt(4 / 3)
    margins = [0.20 - d, 0.20 + d, 0.20 - d, 0.20 + d]
    hist = frame([0.05, 0.08, 0.06, 0.07], margins)
    measured = float(pd.Series(margins).std())
    assert measured < MIN_MARGIN_SIGMA  # the clamp does fire
    assert round(measured, 3) == round(MIN_MARGIN_SIGMA, 3)  # but invisibly
    d = build_distributions(hist, FakeAssumptions(), 0.11, 0.15, 0.075, 0.9, 0.04, 0.09)
    assert d.margin_sigma == MIN_MARGIN_SIGMA
    assert not any("margin volatility" in n for n in d.notes)


def test_missing_history_widens_rather_than_narrows_the_distribution():
    """Not knowing the dispersion is a reason for more uncertainty, not less."""
    hist = frame([np.nan], [np.nan])
    d = build_distributions(hist, FakeAssumptions(), 0.11, 0.15, 0.075, 0.9, 0.04, 0.09)
    assert d.growth_sigma > MIN_GROWTH_SIGMA
    assert any("wide default" in n for n in d.notes)


def test_wacc_uncertainty_scales_with_beta_standard_error_and_equity_weight():
    """A poorly estimated beta must produce a genuinely wider valuation, and a company
    financed mostly by debt must carry less of the equity-risk uncertainty."""
    hist = frame([0.05, 0.08, 0.06, 0.07], [0.20, 0.22, 0.21, 0.23])
    precise = build_distributions(hist, FakeAssumptions(), 0.11, 0.05, 0.075, 1.0, 0.04, 0.09)
    vague = build_distributions(hist, FakeAssumptions(), 0.11, 0.40, 0.075, 1.0, 0.04, 0.09)
    assert vague.wacc_sigma > precise.wacc_sigma
    assert vague.wacc_sigma == pytest.approx(0.40 * 0.075 * 1.0)

    levered = build_distributions(hist, FakeAssumptions(), 0.11, 0.40, 0.075, 0.3, 0.04, 0.09)
    assert levered.wacc_sigma < vague.wacc_sigma


def test_unavailable_beta_error_still_assumes_some_discount_rate_uncertainty():
    hist = frame([0.05, 0.08, 0.06, 0.07], [0.20, 0.22, 0.21, 0.23])
    d = build_distributions(hist, FakeAssumptions(), 0.11, float("nan"), 0.075, 0.9, 0.04, 0.09)
    assert d.wacc_sigma > 0
    assert any("standard error" in n for n in d.notes)


def test_terminal_growth_is_bounded_by_the_inflation_floor_and_gdp_ceiling():
    hist = frame([0.05, 0.08, 0.06, 0.07], [0.20, 0.22, 0.21, 0.23])
    d = build_distributions(hist, FakeAssumptions(), 0.11, 0.15, 0.075, 0.9, 0.04, 0.09)
    assert (d.terminal_growth_low, d.terminal_growth_high) == (0.04, 0.09)


# --- Correlated draws ----------------------------------------------------------------

def test_growth_and_margin_draws_carry_the_intended_correlation():
    """Independent draws would let good growth pair with bad margin as often as bad with
    bad, cancelling the joint downside the correlation exists to preserve."""
    rng = np.random.default_rng(7)
    a, b = _correlated_normals(rng, 200_000, GROWTH_MARGIN_CORRELATION)
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(GROWTH_MARGIN_CORRELATION, abs=0.01)
    # Both series must still be standard normal; a Cholesky slip would inflate the second.
    assert b.std() == pytest.approx(1.0, abs=0.01)


def test_zero_correlation_leaves_the_series_independent():
    rng = np.random.default_rng(7)
    a, b = _correlated_normals(rng, 100_000, 0.0)
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(0.0, abs=0.02)


# --- Blended summary -----------------------------------------------------------------

def test_agreeing_methods_produce_a_central_figure():
    b = build_blended("TCS", "INR", 100.0, dcf_share_price=98.0, comparables_share_price=104.0)
    assert not b.methods_disagree
    assert b.central == pytest.approx(101.0)
    assert "agree within" in b.verdict


def test_disagreeing_methods_refuse_to_average_into_a_fair_value():
    """A DCF at -60% and comparables at +3% have a midpoint no analysis supports."""
    b = build_blended("HINDUNILVR", "INR", 100.0,
                      dcf_share_price=40.0, comparables_share_price=103.0)
    assert b.dispersion > WIDE_DISPERSION
    assert b.methods_disagree
    assert "too wide to average" in b.verdict


def test_monte_carlo_is_context_not_a_third_method():
    """Counting it as a peer of the comparables would double-count the DCF's view and
    narrow the apparent disagreement between genuinely independent methods."""
    b = build_blended("TCS", "INR", 100.0, dcf_share_price=40.0,
                      comparables_share_price=103.0, monte_carlo_median=45.0)
    assert len(b.usable) == 2
    assert {e.name for e in b.usable} == {"Discounted cash flow", "Comparable companies"}
    assert any("context rather than as a third method" in n for n in b.notes)
    # The median of two methods must be unmoved by the Monte Carlo figure.
    assert b.central == pytest.approx(71.5)


def test_a_single_method_is_reported_as_having_no_cross_check():
    b = build_blended("ASML", "USD", 100.0, dcf_share_price=60.0)
    assert len(b.usable) == 1
    assert "no cross-check" in b.verdict
    assert any("no cross-check" in n for n in b.notes)


def test_no_usable_method_gives_no_view_rather_than_a_number():
    b = build_blended("X", "INR", 100.0, dcf_share_price=float("nan"))
    assert not b.usable
    assert np.isnan(b.central)
    assert "no view" in b.verdict


def test_dispersion_needs_two_methods_to_mean_anything():
    b = build_blended("X", "INR", 100.0, dcf_share_price=60.0)
    assert np.isnan(b.dispersion)
    assert not b.methods_disagree
