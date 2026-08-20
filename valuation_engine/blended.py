"""Stage 7b: one summary from several methods, without pretending they agree.

The obvious way to finish a valuation is to average the methods into a single number. This
module deliberately does not, because averaging destroys the most useful thing the engine
has produced. When the DCF and the comparables disagree, that disagreement is a finding: it
is what proved, in this engine's own history, that a systematic gap came from the DCF's
terminal-value mechanism rather than from the underlying data, because two methods built on
the same statements but different machinery moved apart rather than together. An average
would have hidden exactly that signal behind a plausible-looking midpoint.

So the summary reports a **range**, the methods that produced it, and how far apart they
sit. The central figure is the median of the available methods rather than a weighted mean:
a median does not require inventing weights, and with two or three estimates it resists a
single extreme reading pulling the answer somewhere no method actually pointed.

**Wide disagreement is surfaced, not smoothed.** Where the methods span more than
`WIDE_DISPERSION`, the summary says so and declines to present the central figure as a fair
value. A range from -60% to +3% has a midpoint near -30%, and no analysis supports -30%;
what the evidence supports is "these methods disagree, and here is why", which is a more
useful thing to hand someone than a number with the disagreement averaged out of it.
"""

from dataclasses import dataclass, field

import numpy as np

# Beyond this spread between the highest and lowest method, as a share of the current price,
# the methods are telling different stories and a central estimate misrepresents them.
WIDE_DISPERSION = 0.40


@dataclass(frozen=True)
class MethodEstimate:
    """One method's fair value, and what it is worth trusting on."""

    name: str
    share_price: float
    basis: str


@dataclass
class BlendedValuation:
    """The range across methods, and whether they agree enough to have a central view."""

    ticker: str
    currency: str
    current_share_price: float
    estimates: list[MethodEstimate]
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> list[MethodEstimate]:
        return [e for e in self.estimates
                if e.share_price is not None and not np.isnan(e.share_price)]

    @property
    def low(self) -> float:
        return min(e.share_price for e in self.usable) if self.usable else float("nan")

    @property
    def high(self) -> float:
        return max(e.share_price for e in self.usable) if self.usable else float("nan")

    @property
    def central(self) -> float:
        """Median of the available methods. Not a weighted mean: the weights would be
        invented, and inventing them is how a summary acquires false authority."""
        if not self.usable:
            return float("nan")
        return float(np.median([e.share_price for e in self.usable]))

    @property
    def dispersion(self) -> float:
        """The spread between methods, as a share of the current price."""
        if len(self.usable) < 2 or self.current_share_price <= 0:
            return float("nan")
        return (self.high - self.low) / self.current_share_price

    @property
    def methods_disagree(self) -> bool:
        d = self.dispersion
        return bool(not np.isnan(d) and d > WIDE_DISPERSION)

    @property
    def upside(self) -> float:
        if self.current_share_price <= 0 or np.isnan(self.central):
            return float("nan")
        return self.central / self.current_share_price - 1.0

    @property
    def verdict(self) -> str:
        """What the engine is willing to say, given how far the methods sit apart."""
        if not self.usable:
            return ("No method produced a usable valuation, so the engine has no view. That "
                    "is the honest output, not a failure to be worked around.")
        if len(self.usable) == 1:
            only = self.usable[0]
            return (
                f"Only the {only.name.lower()} produced a usable valuation, so there is no "
                "cross-check here and the figure carries the whole of that method's known "
                "biases with nothing to test them against."
            )
        if self.methods_disagree:
            return (
                f"The methods span {self.dispersion:.0%} of the current price, which is too "
                "wide to average into a fair value. The disagreement is the result: it says "
                "the methods are reading different things about this company, and which one "
                "to believe is a judgement about the business rather than an output of the "
                "model. Both readings are shown rather than one being chosen."
            )
        return (
            f"The methods agree within {self.dispersion:.0%} of the current price, so the "
            f"central figure of {self.currency} {self.central:,.2f} is supported by more than "
            "one route to it rather than resting on a single model's assumptions."
        )


def build_blended(
    ticker: str,
    currency: str,
    current_share_price: float,
    dcf_share_price: float | None = None,
    comparables_share_price: float | None = None,
    monte_carlo_median: float | None = None,
) -> BlendedValuation:
    """Assemble whatever methods produced a usable answer into one range.

    Monte Carlo's median is deliberately *not* treated as a third independent method: it is
    the same DCF run over a distribution of its own inputs, so it shares every assumption
    and every structural bias the point-estimate DCF has. Including it as a peer of the
    comparables would double-count the DCF's view and narrow the apparent disagreement
    between genuinely independent methods, which is the one signal this summary exists to
    preserve. It is carried as context alongside the range instead.
    """
    estimates: list[MethodEstimate] = []
    notes: list[str] = []

    if dcf_share_price is not None and not np.isnan(dcf_share_price):
        estimates.append(MethodEstimate(
            "Discounted cash flow", float(dcf_share_price),
            "the company's own forecast cash flows, discounted at its own cost of capital",
        ))
    if comparables_share_price is not None and not np.isnan(comparables_share_price):
        estimates.append(MethodEstimate(
            "Comparable companies", float(comparables_share_price),
            "what the market pays for its sector peers, independent of any cash-flow forecast",
        ))

    if monte_carlo_median is not None and not np.isnan(monte_carlo_median):
        notes.append(
            f"Monte Carlo median is {currency} {monte_carlo_median:,.2f}. It is shown as "
            "context rather than as a third method, because it is the same DCF sampled over "
            "its own input distributions and therefore carries the same structural "
            "assumptions rather than testing them."
        )

    if len(estimates) < 2:
        notes.append(
            "Fewer than two independent methods are available, so there is no cross-check. "
            "The engine's own calibration work only became possible because two methods "
            "built from the same data disagreed; a single method cannot produce that."
        )

    return BlendedValuation(
        ticker=ticker, currency=currency, current_share_price=current_share_price,
        estimates=estimates, notes=notes,
    )
