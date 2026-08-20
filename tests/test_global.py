"""Tests for the Quick DCF path: global currency resolution and market-profile fallback.

Currency resolution is ported from Module 1's yahoo.py, already verified live against Shell,
Infosys and HCL Technologies; these tests pin the same logic so the port cannot silently
drift from the original's behaviour.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from valuation_engine.global_data import _fx_rate as fx_rate_impl
from valuation_engine.global_data import (
    _normalise_quote,
    _resolve_statement_currency,
)
from valuation_engine.universe import (
    GENERIC_MARKET,
    INDIA_MARKET,
    US_MARKET,
    resolve_market_for_currency,
)


# --- Quote normalisation -----------------------------------------------------------------

def test_pence_quotes_are_converted_to_pounds():
    price, cap, currency, notes = _normalise_quote(3320.0, 18_334_199_410_400.0, "GBp")
    assert price == pytest.approx(33.20)
    assert cap == pytest.approx(183_341_994_104.0)
    assert currency == "GBP"
    assert notes


def test_major_unit_quotes_pass_through_untouched():
    price, cap, currency, notes = _normalise_quote(180.14, 207_918_349_648.0, "EUR")
    assert (price, cap, currency) == (180.14, 207_918_349_648.0, "EUR")
    assert not notes


# --- Statement currency resolution ---------------------------------------------------------

def test_far_apart_currencies_are_decided_by_market_cap(monkeypatch):
    """Infosys reports in USD, quotes in INR, roughly 95 to one: the plausibility test
    discriminates here, and must, since the flag is correct in this case."""
    import valuation_engine.global_data as gd
    monkeypatch.setattr(gd, "_fx_rate", lambda a, b: 95.0)
    currency, rate, notes = _resolve_statement_currency("USD", "INR", 20e9, 4.7e12)
    assert currency == "USD"
    assert rate == 95.0
    assert notes


def test_far_apart_currencies_ignore_a_wrong_flag(monkeypatch):
    """HCL Technologies is tagged USD while reporting in rupees; believing the tag would
    overstate the company a hundredfold."""
    import valuation_engine.global_data as gd
    monkeypatch.setattr(gd, "_fx_rate", lambda a, b: 95.0)
    currency, rate, _ = _resolve_statement_currency("USD", "INR", 1.3e12, 3.7e12)
    assert (currency, rate) == ("INR", 1.0)


def test_similar_currencies_trust_the_declared_flag(monkeypatch):
    """Shell reports in USD and quotes in GBP at about 0.79; both readings look equally
    plausible on a price-to-sales test, so the source's tag is the better evidence."""
    import valuation_engine.global_data as gd
    monkeypatch.setattr(gd, "_fx_rate", lambda a, b: 0.79)
    currency, rate, notes = _resolve_statement_currency("USD", "GBP", 285e9, 183e9)
    assert currency == "USD"
    assert rate == pytest.approx(0.79)
    assert notes


def test_matching_currencies_need_no_conversion():
    currency, rate, notes = _resolve_statement_currency("EUR", "EUR", 36.8e9, 208e9)
    assert (currency, rate) == ("EUR", 1.0)
    assert not notes


# --- Market profile resolution --------------------------------------------------------------

def test_inr_resolves_to_the_calibrated_india_profile():
    profile, calibrated = resolve_market_for_currency("INR")
    assert profile is INDIA_MARKET
    assert calibrated is True


def test_usd_resolves_to_the_calibrated_us_profile():
    profile, calibrated = resolve_market_for_currency("USD")
    assert profile is US_MARKET
    assert calibrated is True


def test_uncalibrated_currency_gets_the_generic_fallback_and_is_flagged():
    profile, calibrated = resolve_market_for_currency("EUR")
    assert calibrated is False
    assert profile["currency"] == "EUR"
    assert profile["is_generic"] is True
    # The fallback must still be a complete, usable profile, not a partial stub.
    for key in ("index_ticker", "risk_free_series", "equity_risk_premium",
               "nominal_gdp_growth", "inflation"):
        assert key in profile


def test_generic_fallback_is_never_mutated_across_calls():
    """resolve_market_for_currency must return a fresh copy each time, or setting the
    currency for one company would corrupt the shared GENERIC_MARKET template for the next."""
    eur_profile, _ = resolve_market_for_currency("EUR")
    jpy_profile, _ = resolve_market_for_currency("JPY")
    assert eur_profile["currency"] == "EUR"
    assert jpy_profile["currency"] == "JPY"
    assert GENERIC_MARKET["currency"] is None  # the template itself stays unset
