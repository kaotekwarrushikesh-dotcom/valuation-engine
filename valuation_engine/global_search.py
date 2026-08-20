"""Resolve a typed company name to a ticker, for the Quick DCF path.

Ported from Module 1's `src/providers/resolve.py` for the same standalone-deployment reason
as `global_data.py`: the deployed app cannot see Module 1's files at runtime.
"""

import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class Match:
    ticker: str
    name: str
    exchange: str


def search(query: str, limit: int = 8) -> list[Match]:
    """Candidate listings for a name or ticker. Returns several: the same company often
    trades in more than one place, at different prices and in different currencies, and
    picking one silently would hide that choice from the user."""
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            quotes = yf.Search(query, max_results=limit).quotes
        except Exception:  # noqa: BLE001 - search is a best-effort convenience
            return []

    matches: list[Match] = []
    for q in quotes:
        symbol = q.get("symbol")
        if not symbol or q.get("quoteType") not in (None, "EQUITY"):
            continue
        matches.append(Match(
            ticker=symbol, name=q.get("longname") or q.get("shortname") or symbol,
            exchange=q.get("exchange", ""),
        ))
    return matches
