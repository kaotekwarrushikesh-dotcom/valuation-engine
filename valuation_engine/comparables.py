"""Stage 5: comparable company valuation.

Values a company by asking what the market currently pays for similar businesses, rather
than by forecasting its own cash flows. This is the natural check on a DCF: the two methods
share almost no machinery, so where they agree the answer is more trustworthy, and where
they disagree the disagreement is informative on its own. That independence matters more
than usual here, because Stage 4's Gordon-growth terminal value has a known failure mode for
high-quality, low-growth compounders (see the README's Calibration section): comparables do
not share that mechanism, since they price off what the market actually pays today rather
than off a perpetuity formula.

**Peer selection is sector membership**, defined once in `universe.py` so it cannot be
adjusted per company to produce a wanted answer. A peer group of one is not a peer group, so
a target with fewer than `MIN_PEERS` usable peers is reported as unusable rather than
silently valued off a thin sample.

**Four multiples, two families.**

  Equity multiples (price paid per unit of what equity holders receive)
    P/E        price / diluted EPS. The most quoted multiple, but earnings include the
               effect of leverage and one-off items, and a loss-making year makes it
               undefined rather than merely large.
    P/B        price / book equity per share. Useful for capital-intensive or asset-heavy
               businesses where book value approximates replacement cost; close to
               meaningless for asset-light businesses whose value is mostly intangible.

  Enterprise multiples (price paid for the whole business, independent of capital structure)
    EV/EBITDA  enterprise value / EBITDA. The standard cross-capital-structure multiple,
               since EBITDA sits above the interest line, so two companies financed
               differently are still comparable on it.
    EV/Revenue the multiple of last resort, used when EBITDA is negative or too small to
               be a stable base. Says nothing about profitability on its own.

A multiple is dropped from the peer statistics, not clamped or estimated, when its
denominator is at or near zero or negative: a P/E on negative earnings is not a large
number, it is not a number.

**Implied valuation** applies the peer median multiple to the target's own metric. EV
multiples imply an enterprise value, which is bridged to equity through net debt exactly as
the DCF is, so the two valuations differ only in how they arrive at enterprise value, not in
how they get from there to a share price.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_PEERS = 2
# A multiple more than this many median-absolute-deviations from the peer median is treated
# as an outlier and reported separately rather than pulling the median toward it. Financial
# multiples are heavy-tailed (a company near break-even earnings creates a P/E of 300), and
# the median is already robust to that, but a mean or a naive average would not be.
OUTLIER_MAD_MULTIPLE = 4.0

MULTIPLE_LABELS = {
    "pe": "P/E", "pb": "P/B", "ev_ebitda": "EV/EBITDA", "ev_revenue": "EV/Revenue",
}


@dataclass(frozen=True)
class CompanyMultiples:
    """One company's trading multiples, as at its latest reported year and current price."""

    ticker: str
    name: str
    sector: str
    fiscal_year: int
    revenue: float
    ebitda: float
    net_income: float
    book_equity: float
    net_debt: float
    market_cap: float
    enterprise_value: float
    share_price: float
    shares_outstanding: float
    pe: float
    pb: float
    ev_ebitda: float
    ev_revenue: float
    excluded: dict[str, str] = field(default_factory=dict)


def _safe_ratio(numerator: float, denominator: float, min_denominator: float = 0.0) -> float:
    """A ratio, or NaN where the denominator is at or below the floor.

    The floor is not always zero: EBITDA a hair above zero produces a multiple in the
    thousands that is not wrong so much as meaningless, so callers can pass a small positive
    floor rather than only excluding an exact zero or negative value.
    """
    if denominator <= min_denominator or np.isnan(denominator) or np.isnan(numerator):
        return float("nan")
    return numerator / denominator


def compute_multiples(
    ticker: str, name: str, sector: str, hist: pd.DataFrame,
    share_price: float, shares_outstanding: float, market_cap: float,
) -> CompanyMultiples:
    """One company's trading multiples from its latest reported year."""
    latest = hist.iloc[-1]
    revenue = float(latest["revenue"])
    ebitda = float(latest["ebitda"])
    net_income = float(latest["net_income"])
    book_equity = float(latest["equity"])
    net_debt = float(latest["net_debt"])

    enterprise_value = market_cap + net_debt
    eps = net_income / shares_outstanding if shares_outstanding > 0 else float("nan")
    book_value_per_share = book_equity / shares_outstanding if shares_outstanding > 0 else float("nan")

    excluded: dict[str, str] = {}

    pe = _safe_ratio(share_price, eps, min_denominator=0.0)
    if np.isnan(pe):
        excluded["pe"] = "negative or zero earnings, so price-to-earnings is undefined rather than large"

    pb = _safe_ratio(share_price, book_value_per_share, min_denominator=0.0)
    if np.isnan(pb):
        excluded["pb"] = "negative or zero book equity, most often from sustained buybacks"

    # EBITDA has to clear a small positive floor, not just zero, since a value near zero
    # produces a nominally finite multiple that is still not informative.
    ebitda_floor = max(revenue * 0.01, 0.0)
    ev_ebitda = _safe_ratio(enterprise_value, ebitda, min_denominator=ebitda_floor)
    if np.isnan(ev_ebitda):
        excluded["ev_ebitda"] = "EBITDA is negative or too close to zero to be a stable base"

    ev_revenue = _safe_ratio(enterprise_value, revenue, min_denominator=0.0)
    if np.isnan(ev_revenue):
        excluded["ev_revenue"] = "revenue is not positive"

    return CompanyMultiples(
        ticker=ticker, name=name, sector=sector, fiscal_year=int(latest["fiscal_year"]),
        revenue=revenue, ebitda=ebitda, net_income=net_income, book_equity=book_equity,
        net_debt=net_debt, market_cap=market_cap, enterprise_value=enterprise_value,
        share_price=share_price, shares_outstanding=shares_outstanding,
        pe=pe, pb=pb, ev_ebitda=ev_ebitda, ev_revenue=ev_revenue, excluded=excluded,
    )


@dataclass(frozen=True)
class MultipleStats:
    """Peer statistics for one multiple, and which peers actually fed it."""

    multiple: str
    n: int
    mean: float
    median: float
    min: float
    max: float
    outliers: list[str]


def _mad_outliers(values: dict[str, float]) -> list[str]:
    """Tickers whose value sits unusually far from the peer median.

    Median absolute deviation rather than standard deviation, since MAD is itself robust to
    the outlier it is being used to detect; a standard deviation computed including the
    outlier would be inflated by the very thing it is supposed to flag.
    """
    if len(values) < 3:
        return []
    arr = np.array(list(values.values()))
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad == 0:
        return []
    return [t for t, v in values.items() if abs(v - median) / mad > OUTLIER_MAD_MULTIPLE]


def peer_statistics(peers: list[CompanyMultiples], multiple: str) -> MultipleStats | None:
    """Mean, median, min, max for one multiple across a peer set, excluding undefined values."""
    values = {p.ticker: getattr(p, multiple) for p in peers if not np.isnan(getattr(p, multiple))}
    if len(values) < MIN_PEERS:
        return None

    arr = np.array(list(values.values()))
    return MultipleStats(
        multiple=multiple, n=len(values), mean=float(arr.mean()), median=float(np.median(arr)),
        min=float(arr.min()), max=float(arr.max()), outliers=_mad_outliers(values),
    )


@dataclass(frozen=True)
class ImpliedValuation:
    """What a peer multiple implies the target is worth, bridged to a share price."""

    multiple: str
    peer_median: float
    target_metric: float
    implied_enterprise_value: float | None
    implied_equity_value: float
    implied_share_price: float
    is_equity_multiple: bool


def implied_valuation(
    multiple: str, peer_median: float, target: CompanyMultiples,
) -> ImpliedValuation:
    """Apply a peer median multiple to the target's own metric."""
    is_equity = multiple in ("pe", "pb")

    if multiple == "pe":
        eps = target.net_income / target.shares_outstanding if target.shares_outstanding > 0 else float("nan")
        implied_price = peer_median * eps
        return ImpliedValuation(multiple, peer_median, eps, None,
                               implied_price * target.shares_outstanding, implied_price, True)

    if multiple == "pb":
        bvps = target.book_equity / target.shares_outstanding if target.shares_outstanding > 0 else float("nan")
        implied_price = peer_median * bvps
        return ImpliedValuation(multiple, peer_median, bvps, None,
                               implied_price * target.shares_outstanding, implied_price, True)

    if multiple == "ev_ebitda":
        implied_ev = peer_median * target.ebitda
    elif multiple == "ev_revenue":
        implied_ev = peer_median * target.revenue
    else:
        raise ValueError(f"unknown multiple: {multiple}")

    implied_equity = implied_ev - target.net_debt
    implied_price = implied_equity / target.shares_outstanding if target.shares_outstanding > 0 else float("nan")
    metric = target.ebitda if multiple == "ev_ebitda" else target.revenue
    return ImpliedValuation(multiple, peer_median, metric, implied_ev, implied_equity, implied_price, False)


@dataclass(frozen=True)
class ComparablesResult:
    """The full comparable-company read for one target."""

    target: CompanyMultiples
    peers: list[CompanyMultiples]
    stats: dict[str, MultipleStats]
    implied: dict[str, ImpliedValuation]
    blended_share_price: float
    blended_multiples_used: list[str]
    current_share_price: float

    @property
    def upside(self) -> float:
        if self.current_share_price <= 0 or np.isnan(self.blended_share_price):
            return float("nan")
        return self.blended_share_price / self.current_share_price - 1.0

    def premium_discount(self, multiple: str) -> float:
        """The target's own multiple versus the peer median, as a premium or discount."""
        target_value = getattr(self.target, multiple)
        stat = self.stats.get(multiple)
        if stat is None or np.isnan(target_value) or stat.median == 0:
            return float("nan")
        return target_value / stat.median - 1.0


def run_comparables(target: CompanyMultiples, peers: list[CompanyMultiples]) -> ComparablesResult:
    """Run the full comparable-company valuation for one target against its peers."""
    stats: dict[str, MultipleStats] = {}
    implied: dict[str, ImpliedValuation] = {}

    for m in MULTIPLE_LABELS:
        stat = peer_statistics(peers, m)
        if stat is None:
            continue
        stats[m] = stat
        implied[m] = implied_valuation(m, stat.median, target)

    # The blend weights enterprise multiples above equity ones, since EV/EBITDA and
    # EV/Revenue are unaffected by the target's own capital structure while P/E and P/B are
    # distorted by it (leverage inflates ROE and P/E in ways that have nothing to do with
    # operating quality). Equal weight across whichever multiples actually survived exclusion.
    weights = {"ev_ebitda": 0.35, "ev_revenue": 0.20, "pe": 0.30, "pb": 0.15}
    available = {m: w for m, w in weights.items() if m in implied}
    total_weight = sum(available.values())

    if total_weight > 0:
        blended = sum(implied[m].implied_share_price * w for m, w in available.items()) / total_weight
    else:
        blended = float("nan")

    return ComparablesResult(
        target=target, peers=peers, stats=stats, implied=implied,
        blended_share_price=blended, blended_multiples_used=list(available),
        current_share_price=target.share_price,
    )
