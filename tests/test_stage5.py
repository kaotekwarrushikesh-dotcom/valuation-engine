"""Tests for Stage 5: comparable company valuation.

These target the places a multiples engine quietly produces nonsense: a multiple computed
on a negative or near-zero denominator, a peer group too thin to mean anything, and an
outlier dragging the reported statistic without being flagged.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from valuation_engine.comparables import (
    MIN_PEERS,
    CompanyMultiples,
    compute_multiples,
    implied_valuation,
    peer_statistics,
    run_comparables,
)


def make_hist_row(revenue=1000.0, ebitda_margin=0.25, net_income=100.0, equity=500.0,
                  net_debt=200.0, fiscal_year=2025):
    return pd.DataFrame([{
        "fiscal_year": fiscal_year, "revenue": revenue, "ebitda": revenue * ebitda_margin,
        "net_income": net_income, "equity": equity, "net_debt": net_debt,
    }])


def make_company(ticker, sector="Test", price=100.0, shares=10.0, **kw) -> CompanyMultiples:
    hist = make_hist_row(**kw)
    market_cap = price * shares
    return compute_multiples(ticker, ticker, sector, hist, price, shares, market_cap)


# --- Multiple computation ---------------------------------------------------------------

def test_pe_and_pb_match_hand_calculation():
    c = make_company("A", net_income=100.0, equity=500.0, price=50.0, shares=10.0)
    # EPS = 100/10 = 10, P/E = 50/10 = 5
    assert c.pe == pytest.approx(5.0)
    # BVPS = 500/10 = 50, P/B = 50/50 = 1.0
    assert c.pb == pytest.approx(1.0)


def test_ev_multiples_include_net_debt():
    c = make_company("A", revenue=1000.0, ebitda_margin=0.20, net_debt=300.0,
                     price=100.0, shares=10.0)
    # EV = market cap (1000) + net debt (300) = 1300
    assert c.enterprise_value == pytest.approx(1300.0)
    assert c.ev_ebitda == pytest.approx(1300.0 / 200.0)
    assert c.ev_revenue == pytest.approx(1300.0 / 1000.0)


def test_negative_earnings_excludes_pe_not_inflates_it():
    """A loss-making year must not produce a large negative P/E that looks like a real
    number; it has to be excluded outright."""
    c = make_company("A", net_income=-50.0, price=100.0, shares=10.0)
    assert np.isnan(c.pe)
    assert "pe" in c.excluded


def test_negative_book_equity_excludes_pb():
    """Sustained buybacks push book equity negative; P/B on a negative denominator is not
    meaningful and must be dropped rather than reported as a negative multiple."""
    c = make_company("A", equity=-200.0, price=100.0, shares=10.0)
    assert np.isnan(c.pb)
    assert "pb" in c.excluded


def test_ebitda_near_zero_excludes_ev_ebitda():
    """EBITDA a hair above zero produces a nominal multiple in the thousands, which is not
    a large number, it is a meaningless one, so it must be excluded, not merely large."""
    c = make_company("A", revenue=1000.0, ebitda_margin=0.0001, price=100.0, shares=10.0)
    assert np.isnan(c.ev_ebitda)
    assert "ev_ebitda" in c.excluded


# --- Peer statistics ----------------------------------------------------------------------

def test_peer_statistics_require_the_minimum_peer_count():
    peers = [make_company("A", net_income=100.0), make_company("B", net_income=120.0)]
    assert peer_statistics(peers, "pe") is not None  # exactly MIN_PEERS
    assert MIN_PEERS == 2

    one_peer = [make_company("A", net_income=100.0)]
    assert peer_statistics(one_peer, "pe") is None


def test_undefined_multiples_are_excluded_before_counting_peers():
    """A peer with negative earnings must not count toward the minimum peer requirement for
    P/E, since it contributes nothing to that particular statistic."""
    peers = [
        make_company("A", net_income=100.0),
        make_company("B", net_income=-50.0),  # excluded from P/E
    ]
    assert peer_statistics(peers, "pe") is None  # only 1 usable, below MIN_PEERS


def test_outlier_is_flagged_but_does_not_get_removed_from_the_median():
    """The median is already robust; an outlier is reported for transparency, not silently
    dropped, since dropping data invisibly is its own kind of dishonesty."""
    peers = [
        make_company("A", net_income=100.0, price=100.0, shares=10.0),   # P/E 10
        make_company("B", net_income=100.0, price=105.0, shares=10.0),   # P/E 10.5
        make_company("C", net_income=100.0, price=95.0, shares=10.0),    # P/E 9.5
        make_company("D", net_income=10.0, price=500.0, shares=10.0),    # P/E 500, wild outlier
    ]
    stat = peer_statistics(peers, "pe")
    assert stat is not None
    assert stat.n == 4
    assert "D" in stat.outliers
    # Median stays anchored near the tight cluster despite the outlier.
    assert 9.0 < stat.median < 11.0


# --- Implied valuation -------------------------------------------------------------------

def test_ev_multiple_implied_valuation_bridges_through_net_debt():
    target = make_company("T", revenue=1000.0, ebitda_margin=0.25, net_debt=200.0,
                          price=100.0, shares=10.0)
    imp = implied_valuation("ev_ebitda", peer_median=10.0, target=target)
    # Implied EV = 10 * EBITDA(250) = 2500; equity = 2500 - net_debt(200) = 2300
    assert imp.implied_enterprise_value == pytest.approx(2500.0)
    assert imp.implied_equity_value == pytest.approx(2300.0)
    assert imp.implied_share_price == pytest.approx(230.0)
    assert not imp.is_equity_multiple


def test_equity_multiple_implied_valuation_has_no_enterprise_value_step():
    target = make_company("T", net_income=100.0, price=100.0, shares=10.0)
    imp = implied_valuation("pe", peer_median=12.0, target=target)
    # EPS = 10, implied price = 12 * 10 = 120
    assert imp.implied_enterprise_value is None
    assert imp.implied_share_price == pytest.approx(120.0)
    assert imp.is_equity_multiple


def test_net_cash_raises_implied_equity_above_enterprise_value():
    target = make_company("T", revenue=1000.0, ebitda_margin=0.25, net_debt=-300.0,
                          price=100.0, shares=10.0)
    imp = implied_valuation("ev_ebitda", peer_median=10.0, target=target)
    assert imp.implied_equity_value > imp.implied_enterprise_value


# --- Full run --------------------------------------------------------------------------

def test_run_comparables_blends_across_available_multiples():
    target = make_company("T", revenue=1000.0, ebitda_margin=0.25, net_income=100.0,
                          equity=500.0, net_debt=200.0, price=100.0, shares=10.0)
    peers = [
        make_company("A", revenue=1000.0, ebitda_margin=0.25, net_income=100.0,
                     equity=500.0, net_debt=200.0, price=110.0, shares=10.0),
        make_company("B", revenue=1000.0, ebitda_margin=0.25, net_income=100.0,
                     equity=500.0, net_debt=200.0, price=90.0, shares=10.0),
    ]
    result = run_comparables(target, peers)
    assert not np.isnan(result.blended_share_price)
    assert set(result.blended_multiples_used) <= {"pe", "pb", "ev_ebitda", "ev_revenue"}
    assert result.upside == pytest.approx(result.blended_share_price / 100.0 - 1.0)


def test_run_comparables_handles_a_peer_group_too_thin_for_any_multiple():
    target = make_company("T", price=100.0, shares=10.0)
    result = run_comparables(target, [make_company("A", price=100.0, shares=10.0)])
    assert result.stats == {}
    assert np.isnan(result.blended_share_price)
    assert np.isnan(result.upside)


def test_premium_discount_reads_the_targets_own_multiple_against_the_peer_median():
    target = make_company("T", net_income=100.0, price=150.0, shares=10.0)  # P/E 15
    peers = [
        make_company("A", net_income=100.0, price=100.0, shares=10.0),  # P/E 10
        make_company("B", net_income=100.0, price=100.0, shares=10.0),  # P/E 10
    ]
    result = run_comparables(target, peers)
    assert result.premium_discount("pe") == pytest.approx(0.5, abs=1e-6)  # 15 vs 10 = +50%
