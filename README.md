# Module 2: Institutional-Style Valuation Engine

> Estimating intrinsic value using DCF, market multiples and scenario-based valuation analysis.

Takes cleaned historical financials, builds defensible forecasts, and works through to a
fair value per share. Built for **Nifty (NSE-listed) companies**, in INR.

**Status: Stages 1 to 6 built.** Assumptions, forecast, FCFF, WACC, the DCF, comparable
company valuation, and bull/base/bear scenarios with sensitivity tables all run and are
tested. The DCF's fair value is systematically below the market for every company in the
universe; comparables are not, and the gap between the two methods is itself the diagnosis,
not a mystery, so every scenario built on the DCF inherits it. **The DCF valuations are not
published or quoted on their own** until this is resolved with a richer terminal-value
treatment. What is happening and the evidence for it are written up in
[Calibration](#calibration-the-model-reads-low-and-why) below, because a model that is wrong
in a known direction, with the evidence for why, is more useful than one quietly tuned until
it agrees.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download the universe (33 Nifty companies):

```bash
.venv/bin/python fetch_nifty.py
```

Run the interactive app (Stages 1 to 6, live, any company in the universe):

```bash
.venv/bin/streamlit run app.py
```

The app deliberately never shows the DCF's implied price without comparables next to it, per
the Calibration section: across this universe the DCF's terminal-value method reads
systematically below market for high-quality, low-growth compounders, and comparables do not
share that failure mode, so pairing them is what stops one model's opinion from being read as
a verdict.

Run the stages for one company from the terminal instead:

```bash
.venv/bin/python stage1_historical.py --ticker RELIANCE
```

```bash
.venv/bin/python stage2_forecast.py --ticker RELIANCE
```

```bash
.venv/bin/python stage3_wacc.py --ticker RELIANCE
```

```bash
.venv/bin/python stage4_dcf.py --ticker RELIANCE
```

```bash
.venv/bin/python stage5_comparables.py --ticker RELIANCE
```

```bash
.venv/bin/python stage6_scenarios.py --ticker RELIANCE
```

Build the report (PDF plus a CSV of every driver):

```bash
.venv/bin/python build_report.py
```

Tests:

```bash
.venv/bin/python -m pytest tests/ -q
```

## Output

`outputs/valuation_report.pdf` is the deliverable: the assumption rules, every company's
derived assumptions with reasoning and confidence, its forecast and full FCFF build,
sector medians, cash flow across the universe, and the exclusions. It is explicitly **not**
a valuation, because nothing has been discounted yet. It is the document you would
circulate before running a DCF, so the inputs get argued with while they are still cheap
to change.

## Stage 2: forecast and FCFF

```text
EBIT
- tax on EBIT          computed on unlevered EBIT, not on income after interest
= NOPAT
+ D&A                  subtracted to reach EBIT but no cash left the business
- capex                the actual cash spent on assets
- change in NWC        the movement consumes cash, not the level
= FCFF
```

Three choices carry the finance:

**Tax is charged on unlevered EBIT and interest never appears.** FCFF is the cash available
to every provider of capital before any financing decision. The reported tax charge is
struck after interest, so it already contains the debt tax shield; using it here while WACC
also carries the shield in the discount rate values the same benefit twice. This is the
single most common way a DCF flatters itself, and it always moves the answer up.

**Margins are forecast, EBIT is derived.** Projecting an EBITDA figure hides the operating
assumption inside one number. Forecasting the margin states the claim. And deriving EBIT as
EBITDA less D&A means the implied depreciation is always the one that was actually assumed,
rather than two independent forecasts drifting apart.

**Working capital moves with the revenue increment.** A company can carry a large
receivables balance forever and use no cash, provided it is not growing. Applying the
intensity to the revenue *added* is what "growth consumes cash in proportion to the revenue
it adds" means arithmetically, and it avoids a phantom first-year step.

A negative FCFF is not an error. Grasim consumes cash across the whole horizon, and the
model says so rather than smoothing it away. Where terminal FCFF is not positive, a
perpetuity growth terminal value is meaningless and the report says the exit multiple is
the only defensible route.

Every identity is re-derived independently in `validate_fcff` and the runner refuses to
print results that fail. All 32 companies validate.

## Stage 5: comparable company valuation

Values a company off what the market currently pays for similar businesses, rather than off
a forecast of its own cash flows. This shares almost no machinery with the DCF, which is why
it works as a check on it: where the two agree the answer is more trustworthy, and where
they disagree (see Calibration below) the disagreement is itself informative.

**Peer selection is sector membership, fixed in `universe.py` independently of any single
run**, so it cannot be adjusted per company to reach a wanted answer. A company with fewer
than two usable peers is reported as unusable rather than valued off a group too thin to
mean anything; 8 of the 32 companies fall into this category, mostly single-company sectors
like Energy or Telecom in this universe.

Four multiples, in two families that fail differently:

| | Multiple | What it needs | Excluded when |
|---|---|---|---|
| Equity | P/E | positive earnings | earnings are negative: not a large P/E, an undefined one |
| Equity | P/B | positive book equity | buybacks have pushed book equity negative |
| Enterprise | EV/EBITDA | EBITDA above a small floor | EBITDA is negative or near zero, giving a nominal multiple that is not meaningful |
| Enterprise | EV/Revenue | positive revenue | used as the multiple of last resort |

The peer median (not the mean) is used, since financial multiples are heavy-tailed and a
single company near break-even earnings can produce a P/E in the hundreds. Outliers, found
by median absolute deviation (itself robust to the outlier it is meant to detect), are
flagged for visibility rather than silently dropped from the statistic. The blended
valuation weights enterprise multiples above equity ones, since EV/EBITDA and EV/Revenue are
unaffected by the target's own leverage while P/E and P/B are not.

## Stage 6: scenarios and sensitivity

**Bull, base and bear are each a complete, independently computed valuation**, not a
percentage haircut applied to the base case afterward. A flat -20% implies every driver
moves by the same amount in the same direction, which is not what a downside case looks
like: margins compress, growth slows, capital gets more expensive and the terminal outlook
dims, and those effects compound through the model rather than adding linearly. Each
scenario re-runs the forecast, FCFF and DCF from a shifted assumption set, using the
override mechanism `assumptions.derive` already has, so bear, base and bull can never
silently agree on a driver they were supposed to disagree on.

The shifts are fixed, stated constants (bear: growth -4pts, margin -2pts, capex +15%,
terminal growth -1pt, WACC +150bps; bull the mirror image), not fit to produce a target
spread. A scenario spread that looks dramatic is usually a sign the shifts were chosen for
effect rather than derived from a plausible case.

**Two sensitivity grids, built differently because they test different things.** WACC and
terminal growth sit together in the terminal-value formula and only affect the terminal
value and discounting, not the explicit-period cash flows, so one forecast is built once and
reused across the whole grid. Revenue growth and EBITDA margin drive the explicit-period cash
flows themselves, so every cell in that grid rebuilds the full forecast from scratch; it is
the more expensive table and is kept smaller as a result.

**Driver sensitivity** bumps each assumption by one point in its favourable direction,
holding everything else fixed, and ranks the resulting price change. This answers "what is
the valuation most sensitive to" directly rather than by inspection: for TCS, a one-point
move in WACC or revenue growth changes the valuation roughly three times more than the same
move in capex intensity.

**Negative equity value is floored at zero for display, never reported as a negative
price.** A heavily levered or structurally declining company can genuinely produce a DCF
where net debt exceeds enterprise value, and that is a real, reportable finding: the model
is saying the business does not cover its debt on these assumptions. But equity cannot trade
below zero under limited liability, so the raw negative arithmetic is kept internally
(`implied_share_price`) for anyone measuring how far underwater the business is, while the
displayed price and computed upside use the floored value
(`implied_share_price_floored`), capping downside at -100% rather than reporting something
past it that no investor could actually experience. The sensitivity grids are the one
exception: they show the raw, unfloored numbers on purpose, since a grid exists to reveal
the shape and magnitude of a sensitivity, and flattening a whole stressed region to zero
would hide exactly what the table is for.

## How this connects to Module 1

Module 1 (Financial Statement Intelligence) sources US filers from SEC EDGAR. Indian
companies do not file with the SEC, so that path cannot reach them.

What Module 2 actually depends on is Module 1's **schema**, not its source: one cleaned
table, one row per fiscal year, the same column names. `src/valuation/nse_data.py` is a
second loader that satisfies the same contract for NSE-listed companies, and every stage
downstream runs unchanged on either. Point `--data-dir` at Module 1's folder and a US
ticker works through the identical code path.

```text
SEC EDGAR ──► Module 1 loader ──┐
                                ├──►  cleaned schema  ──►  Module 2 engine
NSE / yfinance ──► NSE loader ──┘
```

Two differences are real and are not smoothed over:

| | US path (Module 1) | India path |
|---|---|---|
| History | 10 years | 4 to 5 years |
| Risk-free rate | US 10Y Treasury | India 10Y G-Sec |
| Equity risk premium | ~4.5% | ~7.5% (includes country risk) |
| Terminal growth ceiling | 4% nominal GDP | 9% nominal GDP |

Four years is at the floor of what a trend-based forecast can stand on. The engine warns
about it rather than treating a four-year read as equivalent to a ten-year one.

## What Stage 1 does

**1. Bridges Module 1's statements into valuation inputs** (`data_bridge.py`)

Derives what a valuation needs and history does not directly report:

- **D&A** from the reported line, falling back to EBITDA minus EBIT
- **Non-cash working capital** as `(current assets - cash) - (current liabilities - short-term debt)`.
  Cash and debt are financing, not operating. Leaving them in makes working capital move
  with the cash balance, which is an output of the model rather than an input to it.
- **Change in working capital**, which is what consumes cash. A company can carry a large
  balance and use no cash at all, as long as the balance is not growing.
- **Net debt** for the enterprise-to-equity bridge
- **Effective tax rate** as tax expense over rebuilt pretax income, because that is the
  rate actually paid rather than the statutory one

**2. Refuses to proceed on inadequate data.** The quality gate checks the specific series
the forecast reads, not the file as a whole, and blocks when a driver has under three
usable years. Nestlé India is blocked today: it moved from a December to a March year-end,
leaving a gap the engine will not bridge.

**3. Reads the history** (`historical.py`). Each driver is classified as accelerating,
stable, decelerating, expanding, compressing, improving or deteriorating, using the first
half of the history against the second half rather than first year against last. Endpoint
comparisons are hostage to a single unusual year at either end.

**4. Derives every forecast assumption from that history** (`assumptions.py`), each one
carrying its rationale and a confidence level.

## Assumption rules, and why

Nothing here is a hard-coded growth rate. Each rule exists to stop a specific way a DCF
gets quietly steered toward a wanted answer.

| Assumption | Rule | Why |
|---|---|---|
| Revenue growth start | Recent 3-year CAGR, with half the excess above 25% removed | The latest single year is noise. Exceptional growth rarely survives a full horizon. |
| Growth path | Geometric fade from start to terminal | Holding growth flat then dropping to the terminal rate implies a cliff no business experiences. |
| Terminal growth | `min(nominal GDP, current growth)` | A company cannot outgrow the economy forever. And a business growing at 2% should not be assumed to re-accelerate to 4% in perpetuity, which would push value into the terminal period on no evidence. |
| EBITDA margin | Recent 3-year average, held flat | Extending margin expansion is the most common way a DCF is talked upward. Expansion must be argued for, not defaulted to. |
| D&A / revenue | Full-period median | Keeps D&A tied to the asset base revenue implies. The median stops one impairment setting the run rate. |
| Capex / revenue | Full-period median | Capex is the assumption most often set too low, because it is the easiest way to manufacture free cash flow. |
| Working capital / revenue | Recent 3-year average | The forecast bridges from the last reported balance. A median spanning a structural shift creates a phantom first-year cash flow. |
| Tax rate | Median effective rate, bounded to 10-35% | The effective rate is what the company pays. The floor stops temporary credits being projected into perpetuity. |

**The working capital rule is not a detail.** Apple's non-cash working capital moved from
+5% of revenue to -10% over a decade. The 10-year median of -2.5% describes no state the
business was ever in, and forecasting it would have produced a fake ~$30bn cash outflow in
year one that was an artefact of averaging, not economics.

**Overrides are allowed and are recorded as overrides.** A hand-set assumption is tagged
`[OVERRIDE]` and reports what the derived value would have been, so it can never be
mistaken for something the evidence produced.

## Universe

33 Nifty companies grouped so every one has a real peer set, since a peer group of one is
not a peer group: IT Services (5), FMCG (5), Automobiles (5), Pharmaceuticals (4),
Metals (3), Cement (3), Energy (2), Utilities (2), Consumer Discretionary (2),
Infrastructure (1), Telecom (1).

Stage 5's comparable valuation needs at least two *other* companies in the sector to form a
peer group, so a sector with only two members gives each company exactly one peer, below the
minimum. 8 of the 33 companies (both Energy and Utilities names, both Consumer Discretionary
names, and the single-member Infrastructure and Telecom sectors) are reported as unusable for
comparables today rather than valued off a peer group too thin to mean anything.

**Banks and NBFCs are deliberately excluded.** FCFF and enterprise value assume debt is
financing. For a lender, borrowing is raw material, so EV/EBITDA and an unlevered DCF are
not merely imprecise, they are the wrong instrument. Valuing them needs an excess-return or
FCFE model, which is a separate build rather than a looser version of this one.

## Structure

```text
valuation_engine/
├── data/nifty/                 one cleaned CSV per company, Module 1 schema
├── src/valuation/
│   ├── nse_data.py             NSE loader into Module 1's schema
│   ├── universe.py             companies, sectors, India market parameters
│   ├── data_bridge.py          statements to valuation inputs, quality gate
│   ├── historical.py           trend classification and interpretation
│   ├── assumptions.py          forecast assumptions with rationale
│   ├── forecasting.py          revenue, EBITDA, D&A, EBIT projection
│   ├── fcff.py                 FCFF build, review and validation
│   ├── beta.py                 regression beta, Blume adjustment
│   ├── wacc.py                 CAPM cost of equity, cost of debt, capital weights
│   ├── terminal_value.py       perpetual growth, exit multiple, reinvestment consistency
│   ├── dcf.py                  discounting, EV-to-equity bridge, reverse DCF
│   ├── comparables.py          peer multiples, statistics, implied valuation
│   ├── scenarios.py            bull/base/bear, each independently computed
│   ├── sensitivity.py          WACC/growth grids, growth/margin grid, driver ranking
│   ├── reporting.py            the PDF report
│   └── market_data.py          prices, share counts, risk-free rate
├── tests/
├── outputs/
├── fetch_nifty.py
├── stage1_historical.py
├── stage2_forecast.py
├── stage3_wacc.py
├── stage4_dcf.py
├── stage5_comparables.py
└── stage6_scenarios.py
```

## Data sources

| Input | Source | Note |
|---|---|---|
| Financial statements | yfinance (NSE) | 4-5 years; EDGAR path available for US names |
| Share price, share count | yfinance | Cached, so a valuation does not change because a quote moved |
| Risk-free rate | FRED `INDIRLTLT01STM` | India 10Y government bond |
| Index for beta | `^NSEI` | Nifty 50 |
| Equity risk premium | Assumption, 7.5% | Named and overridable, not buried in the WACC |

## Calibration: the model reads low, and why

Running the DCF across the universe gives a fair value below the market price for **every
one of the 32 companies**. When a model disagrees with the market on every name in the same
direction, that is worth pinning down rather than publishing anyway, so the valuations are
not surfaced on the portfolio dashboard until this section is resolved or the finding is
confirmed and stated as a finding rather than a defect.

The reverse DCF is the instrument for pinning it down: it solves for the discount rate at
which the model would agree with today's price, which turns a verdict into a question.

| | Modelled WACC | Market-implied WACC | Gap | Terminal growth | Capex intensity |
|---|---|---|---|---|---|
| TCS | 14.24% | 11.78% | 2.46% | 5.8% | 1.5% to 3.2%, low throughout |
| Infosys | 12.90% | 10.88% | 2.02% | 3.4% | low throughout |
| Reliance | 13.01% | 8.47% | 4.54% | 6.4% | 15.3% fading to 12.6% |
| Hindustan Unilever | 13.48% | 6.00% | 7.48% | 4.0% (inflation floor) | minimal |
| **Median across 32** | **~13.4%** | **~9.3%** | **~4.0%** | | |

**First correction, real and confirmed.** Terminal reinvestment was made consistent with
terminal growth (`reinvestment rate = g / ROIC`), but that correction was applied only to
the terminal year. The explicit forecast years still held capex at its historical share of
revenue while growth faded down underneath it, which is the same incoherence in miniature,
repeated for five years. Capex intensity is now faded toward the terminal-consistent level
across the explicit horizon too, the same way revenue growth already fades toward terminal
growth. This is a real fix: it moved Reliance's implied share price from an unusable 22
rupees to 323, and the reinvestment identity behind it is tested directly.

**That fix did not close the gap, and the evidence says why.** After the correction, the
median WACC gap moved from 3.99% to 4.00%, essentially unchanged. TCS and Infosys, whose
capex is close to nothing in either version of the model, still show a two-point gap.
Hindustan Unilever, a fully domestic, low-capex, high-margin compounder, shows the *largest*
gap in the universe at 7.48 points, an implied discount rate of 6.00% against a 7.02%
risk-free rate, meaning the market prices it almost like a bond. Capex was never the driver
for either of these; both were already asset-light before the fix.

The pattern across the table is the actual explanation: **the gap is largest for the
highest-quality, lowest-growth, most richly-rated compounders, and smallest for the more
cyclical, higher-growth or higher-capex names.** That is the textbook, well-documented
failure mode of a Gordon-growth terminal value fed a conservative, GDP-and-inflation-capped
growth rate: the market pays a premium for stability, quality, and optionality beyond the
forecast horizon that a mechanical perpetuity formula does not capture, and it pays that
premium precisely for the businesses that most resemble annuities. This is not a coding
defect, and treating it as one by lowering the equity risk premium until the numbers agree
would be curve-fitting the assumption to a wanted answer, which is the one thing this engine
is built not to do.

**What was checked and deliberately left alone.** The equity risk premium was checked
before being ruled out as the culprit: working the cost of equity through dollars instead
(4.63% Treasury + 4.5% mature ERP + 3.0% country premium, plus the inflation differential)
lands at 14.5%, matching the rupee build, so 7.5% is defensible on its own terms. Lowering
it anyway to shrink the gap would treat the symptom rather than the cause, and the cause is
the terminal-value mechanism, not the discount rate.

**Confirmation from an independent method.** Stage 5 (comparable company valuation, below)
prices each company off what the market currently pays for its sector peers rather than off
a forecast of its own cash flows, so it shares almost none of the DCF's machinery and none
of its terminal-value mechanism. Across the 25 companies with enough sector peers to value:

| | DCF median upside | Comparables median upside |
|---|---|---|
| **Universe-wide** | **-72.6%** | **+2.9%** |
| TCS | -27.8% | -9.8% |

Comparables land close to fair value on the same underlying financial statements and market
data the DCF uses. That rules out a data or arithmetic bug as the explanation, since a bug
in the shared inputs would drag both methods off in the same direction, and confirms the
terminal-value mechanism specifically as the source: comparables do not have one, and
comparables do not show the gap.

**What is still open.** The fix is a richer terminal-value treatment, most likely a
multi-stage fade with an explicit high-growth, transition and mature phase rather than a
single perpetuity growth rate, or leaning on the exit-multiple terminal value (already built
in Stage 4) using sector multiples from Stage 5 rather than the perpetuity-growth method for
the businesses where the gap is largest. Until one of those lands, the DCF and comparables
outputs are best read side by side rather than the DCF being quoted alone.

## Roadmap

Stages 1 to 6 run end to end. The DCF (Stage 4) and comparables (Stage 5) disagree by
design at this point, per the Calibration section, and that disagreement is the current
state of the module rather than a bug hidden before Stage 6 landed; every scenario and
sensitivity table in Stage 6 inherits it, since they are all built on the same terminal-value
mechanism as the base-case DCF.

- **Stage 7** Monte Carlo (the scenario drivers as distributions rather than fixed points)
  and a final summary that blends DCF and comparables rather than picking one

## Known limitations

1. **Four years of history is thin.** Trend classification splits the history in half, so
   with four years each half is two observations. The engine warns; it does not pretend.
2. **yfinance is an unofficial interface.** It can change or return nothing without notice.
   Every call is treated as capable of failing.
3. **The equity risk premium is an estimate, not a measurement.** It is stated as a named
   assumption because there is no public feed for it the way there is for a bond yield.
4. **Assumption rules are deliberately conservative.** Flat margins and capped terminal
   growth will understate a genuinely improving business. The bias is toward not being
   talked into a valuation, and it is a bias.
5. **Cash includes short-term investments.** Indian companies park surplus in liquid funds
   rather than bank balances, so the narrow cash line understates liquidity and would turn
   net-cash companies into net-debt ones. Bajaj Auto reports 2,990 cr of cash against
   11,094 cr including short-term investments.
6. **Consolidated accounts include captive finance arms.** Bajaj Auto shows net debt
   because its lending subsidiary consolidates into the parent. That is what the statements
   say, but it is financing-company debt inside an industrial valuation and should be
   separated before the enterprise-to-equity bridge is trusted.
7. **Reporting currency is inferred, not trusted.** Infosys reports in USD while trading in
   INR, and HCL Technologies carries the same USD flag while reporting in rupees. The flag
   is checked against market capitalisation and only believed when the implied
   price-to-sales ratio is plausible. Where the evidence is ambiguous the statements are
   left alone, because converting on a guess is a silent hundredfold error.
8. **No accounting-quality checks.** One-off items, restatements and segment changes are
   not detected. The engine trusts the filed numbers.
