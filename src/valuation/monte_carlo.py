"""Stage 7a: Monte Carlo, the scenario drivers as distributions rather than three points.

Stage 6 answers "what if things go badly" with one bear case. That is a real answer, but it
is one path chosen by hand, and it says nothing about how *likely* any outcome is. A bear
case two points below the base tells you the same thing whether those two points are a
routine year or a once-in-a-decade shock. Monte Carlo replaces the three chosen points with
a distribution for each driver, runs the full valuation a few thousand times, and reports
the spread of fair values that comes out.

**Every distribution's width is measured, not chosen.** This is the whole discipline of the
module, because a Monte Carlo with invented variances produces a confident-looking
distribution that is really just the modeller's guess with error bars drawn on it:

  revenue growth    Standard deviation of the company's own historical revenue growth.
  EBITDA margin     Standard deviation of its own historical EBITDA margin.
  WACC              Propagated from the *regression standard error* of beta, which is a
                    genuine statistical uncertainty already computed in Stage 3, scaled by
                    the equity risk premium and the weight of equity in the capital
                    structure. A company whose beta is poorly estimated gets a genuinely
                    wider valuation, which is the honest consequence of not knowing its risk.
  terminal growth   Bounded uniform between the inflation floor and the GDP ceiling. This is
                    the one driver with no measurable history, since it describes a period
                    that has not happened, so a bounded "somewhere in this range, no view
                    within it" is more honest than a normal distribution whose mean and
                    variance would both be invented.

**Growth and margin are drawn correlated.** In a downturn revenue and margin fall together:
operating leverage means a volume shortfall hits margin too. Drawing them independently
would let good growth pair with bad margin as often as bad with bad, which cancels much of
the tail and produces a distribution that is too narrow. Narrowness that comes from a
modelling shortcut looks exactly like precision, which is the specific way this kind of
model misleads. The correlation is a stated constant rather than measured, because four
years of history cannot support a meaningful correlation estimate, and that limitation is
disclosed rather than hidden behind a number computed from too little data.

**Trials that fail are counted, not dropped.** A draw where terminal growth lands above WACC
has no finite value, and silently discarding those would bias the reported distribution
upward by removing exactly the bad draws. The share of failed trials is reported alongside
the percentiles, and a high share is itself the finding: it means the company's valuation is
not robust to its own plausible inputs.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.valuation import assumptions as asmp
from src.valuation.dcf import run_dcf
from src.valuation.fcff import build_fcff
from src.valuation.forecasting import build_forecast

DEFAULT_TRIALS = 2000
DEFAULT_SEED = 12345

# Operating leverage ties revenue and margin together in the same direction. Modest and
# positive rather than strong: a high correlation would manufacture fat tails as surely as
# zero correlation would erase them, and neither is measurable from four years of history.
GROWTH_MARGIN_CORRELATION = 0.35

# Floors on the measured dispersions. A company with an unusually smooth four-year run would
# otherwise get a near-zero variance and a Monte Carlo that reports near-certainty, which
# would be an artefact of a short history rather than evidence of a predictable business.
MIN_GROWTH_SIGMA = 0.02
MIN_MARGIN_SIGMA = 0.01
# And ceilings, so one distorted year (a COVID trough, a merger) cannot set a dispersion so
# wide that the distribution says nothing at all.
MAX_GROWTH_SIGMA = 0.20
MAX_MARGIN_SIGMA = 0.10


@dataclass
class DriverDistributions:
    """The sampled distribution for each driver, with the basis for its width."""

    growth_mean: float
    growth_sigma: float
    margin_mean: float
    margin_sigma: float
    wacc_mean: float
    wacc_sigma: float
    terminal_growth_low: float
    terminal_growth_high: float
    notes: list[str] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        rows = [
            ("Revenue growth (year 1)", f"{self.growth_mean:.1%}", f"+/- {self.growth_sigma:.1%}",
             "standard deviation of the company's own historical revenue growth"),
            ("EBITDA margin", f"{self.margin_mean:.1%}", f"+/- {self.margin_sigma:.1%}",
             "standard deviation of its own historical EBITDA margin"),
            ("WACC", f"{self.wacc_mean:.2%}", f"+/- {self.wacc_sigma:.2%}",
             "propagated from the regression standard error of beta"),
            ("Terminal growth",
             f"{self.terminal_growth_low:.1%} to {self.terminal_growth_high:.1%}", "uniform",
             "no measurable history; bounded by the inflation floor and the GDP ceiling"),
        ]
        return pd.DataFrame(rows, columns=["Driver", "Centre", "Dispersion", "Basis"])


@dataclass
class MonteCarloResult:
    """The distribution of fair values, and how much of it to believe."""

    ticker: str
    currency: str
    trials: int
    failed_trials: int
    prices: np.ndarray
    current_share_price: float
    distributions: DriverDistributions

    @property
    def completed(self) -> int:
        return len(self.prices)

    @property
    def failure_rate(self) -> float:
        return self.failed_trials / self.trials if self.trials else float("nan")

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.prices, p)) if self.completed else float("nan")

    @property
    def median(self) -> float:
        return self.percentile(50)

    @property
    def probability_above_market(self) -> float:
        """The share of trials valuing the company above its current price.

        This is the number a Monte Carlo is actually for. A point estimate says the company
        is worth more or less than the market; this says how often it is worth more across
        the plausible range of its own inputs, which is a claim about confidence rather than
        about a single answer.
        """
        if not self.completed or self.current_share_price <= 0:
            return float("nan")
        return float((self.prices > self.current_share_price).mean())

    def summary_table(self) -> pd.DataFrame:
        rows = [
            ("5th percentile", self.percentile(5)),
            ("25th percentile", self.percentile(25)),
            ("Median", self.median),
            ("75th percentile", self.percentile(75)),
            ("95th percentile", self.percentile(95)),
            ("Current price", self.current_share_price),
        ]
        return pd.DataFrame(rows, columns=["Outcome", f"{self.currency} per share"])


def build_distributions(
    hist: pd.DataFrame,
    base: asmp.ForecastAssumptions,
    base_wacc: float,
    beta_standard_error: float,
    equity_risk_premium: float,
    weight_equity: float,
    inflation: float,
    nominal_gdp_growth: float,
) -> DriverDistributions:
    """Derive each driver's distribution from measured dispersion where one exists."""
    notes: list[str] = []

    growth_hist = hist["revenue_growth"].dropna()
    margin_hist = hist["ebitda_margin"].dropna()

    growth_sigma = float(growth_hist.std()) if len(growth_hist) >= 2 else float("nan")
    margin_sigma = float(margin_hist.std()) if len(margin_hist) >= 2 else float("nan")

    if np.isnan(growth_sigma):
        growth_sigma = MIN_GROWTH_SIGMA * 2
        notes.append(
            "Too few years to measure growth volatility, so a wide default is used rather "
            "than a narrow one: not knowing the dispersion is a reason for more uncertainty, "
            "not less."
        )
    if np.isnan(margin_sigma):
        margin_sigma = MIN_MARGIN_SIGMA * 2
        notes.append("Too few years to measure margin volatility, so a wide default is used.")

    clamped_growth = float(np.clip(growth_sigma, MIN_GROWTH_SIGMA, MAX_GROWTH_SIGMA))
    clamped_margin = float(np.clip(margin_sigma, MIN_MARGIN_SIGMA, MAX_MARGIN_SIGMA))

    # Only worth saying when the clamp actually changed the number the reader will see.
    # A measured 0.995% clamped to 1.0% both display as "1.0%", and a note claiming one was
    # clamped to the other reads as a defect in the model rather than a rounding artefact.
    def _visible(before: float, after: float) -> bool:
        return round(before, 3) != round(after, 3)

    if _visible(growth_sigma, clamped_growth):
        notes.append(
            f"Measured growth volatility of {growth_sigma:.1%} was clamped to "
            f"{clamped_growth:.1%}; four years is too short a window to take an extreme "
            "dispersion at face value."
        )
    if _visible(margin_sigma, clamped_margin):
        notes.append(
            f"Measured margin volatility of {margin_sigma:.1%} was clamped to "
            f"{clamped_margin:.1%}."
        )

    # Beta's regression standard error is an uncertainty about the cost of equity, and only
    # the equity slice of the capital structure carries it through to WACC.
    if np.isnan(beta_standard_error):
        wacc_sigma = 0.01
        notes.append(
            "Beta's standard error is unavailable, so a 100 basis point discount-rate "
            "uncertainty is assumed rather than treating the rate as known exactly."
        )
    else:
        wacc_sigma = float(beta_standard_error * equity_risk_premium * weight_equity)

    return DriverDistributions(
        growth_mean=base.revenue_growth_path[0],
        growth_sigma=clamped_growth,
        margin_mean=base.ebitda_margin_path[0],
        margin_sigma=clamped_margin,
        wacc_mean=base_wacc,
        wacc_sigma=wacc_sigma,
        terminal_growth_low=inflation,
        terminal_growth_high=nominal_gdp_growth,
        notes=notes,
    )


def _correlated_normals(rng, n: int, correlation: float) -> tuple[np.ndarray, np.ndarray]:
    """Two standard normal series with the given correlation, via a Cholesky factor."""
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    return a, correlation * a + np.sqrt(1.0 - correlation**2) * b


def run_monte_carlo(
    hist: pd.DataFrame,
    analysis,
    ticker: str,
    base: asmp.ForecastAssumptions,
    base_wacc: float,
    beta_standard_error: float,
    equity_risk_premium: float,
    weight_equity: float,
    net_debt: float,
    shares_outstanding: float,
    current_share_price: float,
    currency: str,
    nominal_gdp_growth: float,
    inflation: float,
    horizon: int,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> MonteCarloResult:
    """Run the full valuation `trials` times over the driver distributions.

    Seeded by default: a valuation that changes every time it is opened cannot be discussed,
    and reproducibility matters more here than fresh randomness. Pass a different seed to
    confirm the percentiles are stable rather than an artefact of one draw.
    """
    dist = build_distributions(
        hist, base, base_wacc, beta_standard_error, equity_risk_premium,
        weight_equity, inflation, nominal_gdp_growth,
    )

    rng = np.random.default_rng(seed)
    z_growth, z_margin = _correlated_normals(rng, trials, GROWTH_MARGIN_CORRELATION)

    growths = dist.growth_mean + z_growth * dist.growth_sigma
    margins = dist.margin_mean + z_margin * dist.margin_sigma
    waccs = dist.wacc_mean + rng.standard_normal(trials) * dist.wacc_sigma
    terminals = rng.uniform(dist.terminal_growth_low, dist.terminal_growth_high, trials)

    # The same bounds the deterministic path enforces, applied to the draws rather than
    # rejecting them: a draw outside a bound is not an impossible world, it is a world where
    # the bound binds, which is what the base case would also do with that input.
    growths = np.clip(growths, -0.15, 0.60)
    margins = np.clip(margins, 0.01, None)

    prices: list[float] = []
    failed = 0

    for g, m, w, tg in zip(growths, margins, waccs, terminals):
        try:
            a = asmp.derive(
                hist, analysis, ticker, horizon=horizon,
                overrides={
                    "revenue_growth_start": float(g),
                    "ebitda_margin": float(m),
                    "terminal_growth": float(tg),
                },
                nominal_gdp_growth=nominal_gdp_growth, inflation=inflation,
            )
            fcff = build_fcff(build_forecast(hist, a, ticker))
            result = run_dcf(
                ticker=ticker, fcff=fcff, wacc=float(w), terminal_growth=a.terminal_growth,
                net_debt=net_debt, shares_outstanding=shares_outstanding,
                current_share_price=current_share_price, currency=currency,
                roic=a.terminal_roic,
            )
            prices.append(result.implied_share_price_floored)
        except (ValueError, ZeroDivisionError):
            failed += 1

    return MonteCarloResult(
        ticker=ticker, currency=currency, trials=trials, failed_trials=failed,
        prices=np.array(prices), current_share_price=current_share_price,
        distributions=dist,
    )
