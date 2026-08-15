"""Beta: how much a company's returns move with the market.

Beta is the slope of a regression of the company's returns on the market's returns. A beta
of 1.3 says that when the index moves 10%, this stock has historically moved 13%. In CAPM
it is the only company-specific term, so everything the cost of equity says about risk
comes through this one number.

Choices that matter, and why:

**Monthly returns over five years.** Daily returns for a single stock against an index are
dominated by noise and by non-synchronous trading, which biases beta downward. Monthly over
five years gives roughly 60 observations, enough for a usable estimate and the standard
practice for a beta used in a valuation.

**The raw beta is adjusted toward 1.** Betas mean-revert: a company with a measured beta of
1.8 is far more likely to sit closer to 1.3 in future than to stay at 1.8. The Blume
adjustment, `0.33 + 0.67 x raw`, is the standard correction and is what Bloomberg reports as
"adjusted beta". The adjusted figure is used in the cost of equity, and the raw one is
reported alongside so the size of the adjustment is visible.

**R-squared is reported, not hidden.** It is the share of the company's return variation the
index explains. At 0.6 the beta is meaningful; at 0.1 the regression is mostly fitting
noise, and a cost of equity built on it deserves far less confidence than its two decimal
places suggest.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Blume adjustment weights. The raw estimate is pulled a third of the way to the market
# beta of 1, which is the standard correction for the mean reversion betas display.
BLUME_MARKET_WEIGHT = 0.33
BLUME_RAW_WEIGHT = 0.67

MIN_OBSERVATIONS = 24
WEAK_FIT_R_SQUARED = 0.20


@dataclass(frozen=True)
class BetaEstimate:
    """A beta, and everything needed to judge whether to trust it."""

    raw: float
    adjusted: float
    r_squared: float
    observations: int
    standard_error: float
    index_name: str
    warnings: tuple[str, ...]

    @property
    def confidence(self) -> str:
        if self.warnings:
            return "low"
        if self.r_squared >= 0.40 and self.observations >= 48:
            return "high"
        return "medium"


def to_returns(prices: pd.DataFrame, column: str = "close") -> pd.Series:
    """Period-over-period simple returns."""
    return prices[column].pct_change().dropna()


def align(company: pd.Series, market: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Line the two return series up on their common dates.

    Regressing unaligned series produces a number that looks like a beta and means nothing,
    so the join is inner and dates that exist in only one series are dropped.
    """
    joined = pd.concat([company.rename("company"), market.rename("market")], axis=1).dropna()
    return joined["company"].to_numpy(), joined["market"].to_numpy()


def estimate_beta(
    company_prices: pd.DataFrame,
    market_prices: pd.DataFrame,
    index_name: str = "index",
) -> BetaEstimate:
    """Regress company returns on market returns."""
    y, x = align(to_returns(company_prices), to_returns(market_prices))
    n = len(y)

    warnings: list[str] = []
    if n < MIN_OBSERVATIONS:
        warnings.append(
            f"only {n} overlapping monthly returns, below the {MIN_OBSERVATIONS} needed for a "
            "meaningful regression"
        )

    if n < 3 or np.std(x) == 0:
        return BetaEstimate(float("nan"), float("nan"), float("nan"), n, float("nan"),
                            index_name, tuple(warnings + ["regression not possible"]))

    # Ordinary least squares slope: covariance over the market's variance.
    x_centred = x - x.mean()
    y_centred = y - y.mean()
    raw = float((x_centred * y_centred).sum() / (x_centred**2).sum())

    intercept = float(y.mean() - raw * x.mean())
    fitted = intercept + raw * x
    residuals = y - fitted

    total_ss = float((y_centred**2).sum())
    residual_ss = float((residuals**2).sum())
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else float("nan")

    # Standard error of the slope, which says how tightly the beta is pinned down.
    dof = n - 2
    standard_error = (
        float(np.sqrt(residual_ss / dof / (x_centred**2).sum())) if dof > 0 else float("nan")
    )

    if r_squared < WEAK_FIT_R_SQUARED:
        warnings.append(
            f"the index explains only {r_squared:.0%} of this company's return variation, so "
            "the beta is largely fitting noise and the cost of equity built on it is soft"
        )
    if raw < 0:
        warnings.append("negative beta, which is rare and usually a sign of a data problem")

    adjusted = BLUME_MARKET_WEIGHT + BLUME_RAW_WEIGHT * raw

    return BetaEstimate(
        raw=raw,
        adjusted=adjusted,
        r_squared=r_squared,
        observations=n,
        standard_error=standard_error,
        index_name=index_name,
        warnings=tuple(warnings),
    )


def peer_median_beta(betas: dict[str, BetaEstimate]) -> float:
    """Median adjusted beta across a peer set.

    A single company's regression can be distorted by one-off events. The sector median is
    steadier, and a large gap between a company's own beta and its peers' is worth
    explaining rather than accepting.
    """
    values = [b.adjusted for b in betas.values() if not np.isnan(b.adjusted)]
    return float(np.median(values)) if values else float("nan")
