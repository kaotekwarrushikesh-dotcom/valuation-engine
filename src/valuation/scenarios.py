"""Stage 6a: bull, base and bear scenarios.

Each scenario is a complete, independently computed valuation, not a percentage haircut
applied to the base case afterward. Applying a flat -20% to a base-case fair value would
imply every driver moves by the same amount in the same direction, which is not what a
downside case actually looks like: margins compress, growth slows, capital gets more
expensive and the terminal outlook dims, and those effects compound through the model
rather than adding linearly. So each scenario re-runs the forecast, FCFF and DCF from a
shifted assumption set, using the override mechanism `assumptions.derive` already has, and
the whole chain (revenue, EBITDA, EBIT, FCFF, WACC, terminal value, enterprise value, equity
value, implied share price) is recomputed rather than only the final number moving.

**Bear**: growth and margin degrade, the discount rate rises (capital gets more expensive
and demanded returns rise in a downside), terminal growth is pulled toward the inflation
floor, and capex intensity rises (worse working-capital and capital efficiency under
pressure). **Bull** is the mirror image. **Base** is Stage 1 to 4 unchanged; it exists here
as a scenario so all three can be produced, compared, and validated by the same code path.

The shifts are fixed, stated constants, not fit to produce a target spread. They are
deliberately modest: a scenario spread that looks dramatic is usually a sign the shifts were
chosen for effect rather than derived from a plausible downside or upside case.
"""

from dataclasses import dataclass

from src.valuation import assumptions as asmp
from src.valuation.dcf import DCFResult, run_dcf
from src.valuation.fcff import FCFFResult, build_fcff
from src.valuation.forecasting import Forecast, build_forecast


@dataclass(frozen=True)
class ScenarioAdjustments:
    """The shifts applied to the base case to build one scenario.

    Deltas are additive (percentage points), multipliers are relative to the base case's own
    starting value, so a scenario is defined by how far it moves the company from where its
    own history already put it, not by an absolute target level.
    """

    name: str
    revenue_growth_delta: float
    ebitda_margin_delta: float
    capex_multiplier: float
    terminal_growth_delta: float
    wacc_delta: float
    rationale: str


BEAR = ScenarioAdjustments(
    name="Bear",
    revenue_growth_delta=-0.04,
    ebitda_margin_delta=-0.02,
    capex_multiplier=1.15,
    terminal_growth_delta=-0.01,
    wacc_delta=0.015,
    rationale=(
        "Revenue growth 4 points lower, EBITDA margin 2 points lower, capex 15% heavier "
        "(worse capital efficiency under pressure), terminal growth pulled 1 point toward "
        "the inflation floor, and WACC 150 basis points higher (capital is more expensive "
        "and required returns rise in a downturn)."
    ),
)

BASE = ScenarioAdjustments(
    name="Base", revenue_growth_delta=0.0, ebitda_margin_delta=0.0, capex_multiplier=1.0,
    terminal_growth_delta=0.0, wacc_delta=0.0,
    rationale="Stages 1 to 4 unchanged: the assumptions derived from the company's own history.",
)

BULL = ScenarioAdjustments(
    name="Bull",
    revenue_growth_delta=0.04,
    ebitda_margin_delta=0.02,
    capex_multiplier=0.90,
    terminal_growth_delta=0.01,
    wacc_delta=-0.010,
    rationale=(
        "Revenue growth 4 points higher, EBITDA margin 2 points higher, capex 10% lighter "
        "(better capital efficiency), terminal growth 1 point higher (capped by the same "
        "GDP ceiling as the base case), and WACC 100 basis points lower."
    ),
)

SCENARIOS = {"bear": BEAR, "base": BASE, "bull": BULL}


@dataclass
class ScenarioResult:
    name: str
    adjustments: ScenarioAdjustments
    assumptions: asmp.ForecastAssumptions
    forecast: Forecast
    fcff: FCFFResult
    wacc: float
    dcf: DCFResult | None
    error: str | None = None


def build_scenario(
    hist, analysis, ticker: str, base: asmp.ForecastAssumptions, base_wacc: float,
    adj: ScenarioAdjustments, net_debt: float, shares_outstanding: float,
    current_share_price: float, currency: str, nominal_gdp_growth: float, inflation: float,
    horizon: int, exit_ev_ebitda: float | None = None,
) -> ScenarioResult:
    """Build one complete scenario by overriding the base case's starting assumptions."""
    base_growth = base.revenue_growth_path[0]
    base_margin = base.ebitda_margin_path[0]
    base_capex = base.capex_pct_revenue_path[0]

    # Terminal growth is clamped even though a scenario override otherwise bypasses the
    # inflation floor that assumptions.derive applies internally. A bear case is allowed to
    # push below that floor on purpose, since stress-testing a permanent real decline is the
    # point of a downside scenario, but not below zero: a terminal contraction to nothing is
    # not a stress test, it is a different company. The GDP ceiling still binds on the way up.
    terminal_growth = max(
        0.0, min(base.terminal_growth + adj.terminal_growth_delta, nominal_gdp_growth)
    )

    overrides = {
        "revenue_growth_start": max(min(base_growth + adj.revenue_growth_delta, 0.60), -0.15),
        "ebitda_margin": max(base_margin + adj.ebitda_margin_delta, 0.01),
        "capex_pct_revenue": max(base_capex * adj.capex_multiplier, 0.0),
        "terminal_growth": terminal_growth,
    }

    a = asmp.derive(
        hist, analysis, ticker, horizon=horizon, overrides=overrides,
        nominal_gdp_growth=nominal_gdp_growth, inflation=inflation,
    )
    fc = build_forecast(hist, a, ticker)
    fcff = build_fcff(fc)
    wacc = base_wacc + adj.wacc_delta

    try:
        dcf = run_dcf(
            ticker=ticker, fcff=fcff, wacc=wacc, terminal_growth=a.terminal_growth,
            net_debt=net_debt, shares_outstanding=shares_outstanding,
            current_share_price=current_share_price, currency=currency,
            exit_ev_ebitda=exit_ev_ebitda, roic=a.terminal_roic,
        )
        error = None
    except ValueError as exc:
        dcf = None
        error = str(exc)

    return ScenarioResult(adj.name, adj, a, fc, fcff, wacc, dcf, error)


def run_all_scenarios(
    hist, analysis, ticker: str, base: asmp.ForecastAssumptions, base_wacc: float,
    net_debt: float, shares_outstanding: float, current_share_price: float, currency: str,
    nominal_gdp_growth: float, inflation: float, horizon: int,
    exit_ev_ebitda: float | None = None,
) -> dict[str, ScenarioResult]:
    """Bear, base and bull, each a fully independent valuation."""
    return {
        key: build_scenario(
            hist, analysis, ticker, base, base_wacc, adj, net_debt, shares_outstanding,
            current_share_price, currency, nominal_gdp_growth, inflation, horizon,
            exit_ev_ebitda,
        )
        for key, adj in SCENARIOS.items()
    }
