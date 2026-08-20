# Module 2: Institutional-Style Valuation Engine

> Estimating intrinsic value using DCF, market multiples and scenario-based valuation analysis.

Takes cleaned historical financials, builds defensible forecasts, and works through to a
fair value per share. Built for **Nifty (NSE-listed) companies**, in INR, with a second
**Quick DCF** mode for any listed company worldwide (see
[Quick DCF: any company](#quick-dcf-any-company)).

**Status: Stages 1 to 7 built, plus Quick DCF.** Assumptions, forecast, FCFF, WACC, the DCF,
comparable company valuation, bull/base/bear scenarios with sensitivity tables, and Monte
Carlo with a blended cross-method summary all run and are tested for the curated universe.

The DCF still reads below market across this universe, and that is reported rather than
tuned away. A real error was found in the process: the cost of equity was charging India's
country risk twice, once inside the rupee government yield and again inside a total equity
risk premium built to sit on a default-free rate. Fixing it moved the median WACC from
13.34% to 11.17% and closed about 15 points of the gap. The gap that remains, and the two
groups of companies it concentrates in, are set out in
[Calibration](#calibration-the-model-read-low-what-was-wrong-and-what-still-is) below, along
with the first diagnosis published here that turned out to be wrong and how it was
disproved. **The DCF is never quoted on its own**: the app pairs it with comparables, or
says plainly when no comparables exist.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download the universe (33 Nifty companies):

```bash
.venv/bin/python fetch_nifty.py
```

Run the interactive app:

```bash
.venv/bin/streamlit run app.py
```

The app has two modes. **Curated Nifty universe** runs the full workflow (Stages 1 to 7) for
the 33 hand-picked companies, DCF paired with comparables. **Quick DCF** takes any listed
company by name or ticker, fetched live via Yahoo Finance, anywhere in the world, and runs
DCF, scenarios and Monte Carlo without comparables, since real sector peers only exist for
the curated universe (see [Quick DCF: any company](#quick-dcf-any-company) below).

The app deliberately never shows the DCF's implied price without comparables next to it in
curated mode, per the Calibration section: the DCF reads below market across this universe,
comparables do not, and pairing them is what stops one model's opinion from being read as a
verdict. Where the two disagree by more than 40% of the current price, the summary reports
the range and refuses to average them into a central figure, because the midpoint of two
methods that disagree is a number no analysis supports. Quick DCF mode has no comparables to
pair with, so it says so explicitly instead of silently omitting the caveat.

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

```bash
.venv/bin/python stage7_monte_carlo.py --ticker RELIANCE
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

## Stage 7: Monte Carlo and the blended summary

Stage 6's bear case is one path chosen by hand. It says nothing about how *likely* any
outcome is, and a bear case two points below the base reads the same whether those two points
are a routine year or a once-in-a-decade shock. Stage 7 replaces the three chosen points with
a distribution for each driver and runs the full valuation a few thousand times.

**Every distribution's width is measured, not chosen.** This is the whole discipline of the
stage: a Monte Carlo with invented variances is the modeller's guess with error bars drawn on
it, and it looks far more authoritative than the guess did on its own.

| Driver | Width comes from |
|---|---|
| Revenue growth | Standard deviation of the company's own historical revenue growth |
| EBITDA margin | Standard deviation of its own historical EBITDA margin |
| WACC | The **regression standard error of beta** from Stage 3, scaled by the equity risk premium and the equity weight, so a poorly estimated beta produces a genuinely wider valuation |
| Terminal growth | Bounded uniform between the inflation floor and the GDP ceiling |

Terminal growth is the one driver with no measurable history, because it describes a period
that has not happened. A bounded "somewhere in this range, no view within it" is more honest
there than a normal distribution whose mean and variance would both be invented.

**Growth and margin are drawn correlated** (0.35, a stated constant). Operating leverage ties
them together in a downturn; drawing them independently would let good growth pair with bad
margin as often as bad with bad, cancelling much of the tail and producing a distribution too
narrow. Narrowness that comes from a modelling shortcut is indistinguishable from precision,
which is the specific way this kind of model misleads. Four years cannot support a measured
correlation, so the constant is disclosed rather than dressed up as an estimate.

**Failed trials are counted, not dropped.** A draw where terminal growth lands above WACC has
no finite value. Silently discarding those would remove exactly the bad draws and bias the
reported distribution upward, so the failure rate is reported alongside the percentiles, and a
high one is itself the finding.

The headline output is not a number but a probability: **the share of trials valuing the
company above its current price.** A point estimate says more or less; this says how often,
across the plausible range of the company's own inputs. The percentiles rather than the mean
are reported, because the Gordon-growth denominator produces a long right tail as drawn
terminal growth approaches drawn WACC, and the mean sits somewhere no trial clusters.

**The blended summary reports a range and refuses to average methods that disagree.** Where
the DCF and comparables span more than 40% of the current price, no central figure is
presented: a DCF at -60% and comparables at +3% have a midpoint near -30%, and no analysis
supports -30%. The disagreement is the result, and it is the same signal that made this
engine's own calibration work possible. Monte Carlo's median is carried as context rather
than as a third method, because it is the same DCF sampled over its own inputs and would
double-count the DCF's view against the one genuinely independent method available.

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

## Quick DCF: any company

The curated universe exists because comparable-company valuation needs real sector peers, and
a hand-picked list is what made Stage 5 trustworthy enough to diagnose the Calibration finding
in the first place (see above). Auto-discovering peers for an arbitrary company would be lower
quality without saying so, which defeats the point of comparables as a cross-check, so Quick
DCF does not attempt it: a company entered here gets a DCF and scenarios, not comparables.

Everything else runs live instead of from `data/nifty/`: statements, price and share count are
fetched from Yahoo Finance for whatever ticker is resolved from the typed name, through
`src/valuation/global_data.py` and `global_search.py`. These are self-contained ports of
Module 1's currency-handling logic (minor-unit quotes such as GBp, statement-vs-quote currency
mismatches such as Infosys reporting in USD while quoting in INR) rather than a shared import,
because the deployed app is its own standalone repository and cannot reach into a sibling
repo's files at runtime.

Market assumptions (risk-free rate, equity risk premium, terminal-growth ceiling) are only
calibrated for USD and INR, the two markets this engine has actually tuned. Any other resolved
currency gets a disclosed generic developed-market fallback rather than a silently wrong
number, shown as a warning in the app rather than a footnote, since a fallback the user cannot
see is worse than no fallback at all.

## Structure

```text
valuation_engine/
├── data/nifty/                 one cleaned CSV per company, Module 1 schema
├── src/valuation/
│   ├── nse_data.py             NSE loader into Module 1's schema
│   ├── global_data.py          live fetch + currency handling for any company (Quick DCF)
│   ├── global_search.py        name/ticker resolution for Quick DCF
│   ├── universe.py             companies, sectors, market parameters (India, US, generic)
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
│   ├── monte_carlo.py          driver distributions, measured widths, trial distribution
│   ├── blended.py              cross-method range; refuses to average disagreeing methods
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
├── stage6_scenarios.py
└── stage7_monte_carlo.py
```

## Data sources

| Input | Source | Note |
|---|---|---|
| Financial statements | yfinance (NSE) | 4-5 years; EDGAR path available for US names |
| Share price, share count | yfinance | Cached, so a valuation does not change because a quote moved |
| Risk-free rate | FRED `INDIRLTLT01STM` | India 10Y government bond, as quoted |
| Sovereign default spread | Assumption, 2.2% | Stripped from the yield before CAPM; see Calibration |
| Index for beta | `^NSEI` | Nifty 50 |
| Equity risk premium | Assumption, 7.5% | Total (mature + country); named and overridable |

## Calibration: the model read low, what was wrong, and what still is

Running the DCF across the universe originally gave a fair value below the market price for
**every one of the 32 companies**, a median of **-74.7%**, with only 2 companies above
market. When a model disagrees with the market on every name in the same direction, that is
worth pinning down rather than publishing anyway.

This section is kept as a record of the investigation rather than trimmed to its conclusion,
because the first published diagnosis here was wrong, and the way it was wrong is the useful
part: it was a plausible story that fitted a handful of examples, and it survived until it
was tested against the whole universe rather than the examples that suggested it.

### The first diagnosis, and why it did not survive

The original explanation was that a conservative, GDP-and-inflation-capped terminal growth
rate was starving the terminal value, and that the gap should therefore be largest for
high-quality, low-growth compounders whose growth was capped hardest. Hindustan Unilever,
pinned at the 4% inflation floor with the widest gap in the universe, fitted that story
neatly.

It does not survive contact with the rest of the universe. The companies pinned at the
*highest* terminal growth the model allows, the full 9% GDP ceiling, were among the **worst**
readings, not the best:

| At the 9% GDP ceiling | Upside (before any fix) |
|---|---|
| UltraTech Cement | -97.6% |
| Titan | -92.7% |
| Divi's Laboratories | -84.9% |
| Nestlé India | -82.2% |

If a capped terminal growth rate were the mechanism, these companies, capped at nothing,
should have read closest to market. They read furthest from it.

Testing each candidate lever one at a time across the universe made that conclusive. Each
row re-runs the full valuation for all 32 companies changing only the named input:

| Lever | Median upside | Above market |
|---|---|---|
| Baseline | -74.7% | 2 / 33 |
| Terminal growth raised to the GDP ceiling for every company | -66.8% | 7 / 33 |
| ROIC faded to WACC in perpetuity (competitive equilibrium) | -71.8% | 1 / 33 |
| ROIC faded halfway to WACC | -70.4% | 1 / 33 |
| **WACC reduced 200bp** | **-60.1%** | **7 / 32** |

Terminal growth moved the median 8 points. The standard stable-growth fix, letting excess
returns compete away, made it *worse*. The discount rate was the only lever that mattered,
which pointed the investigation at WACC rather than at the terminal value.

### The real error: country risk counted twice

The cost of equity was built as India's rupee 10-year government yield (6.89%) plus a 7.5%
equity risk premium. Both numbers are individually defensible. Together they are not.

A 7.5% India ERP is a *total* premium: a mature-market premium of roughly 4.6% plus an India
country risk premium of roughly 2.9%. A total premium of that construction is designed to sit
on top of a **default-free** rate. The rupee 10-year is not one. It yields more than a
Treasury partly because expected rupee inflation is higher and partly because lenders price
the chance the sovereign does not pay. That second component is country risk, and the country
premium inside the 7.5% charges for it a second time.

The size of the double count is visible directly in the bond market. India's 10-year yields
269bp over the US 10-year, and that spread is the market's combined price of the inflation
differential *and* sovereign risk. The dollar-route build added a 2.9% country premium and
then inflation-scaled the result, implying roughly 476bp of India-specific premium against
the 269bp the bond market actually charges.

**The previous version of this section defended the 7.5% ERP with a check that repeated the
error.** It reported that building the cost of equity through dollars (Treasury + mature ERP
+ country premium, inflation-adjusted) landed at 14.5%, matching the rupee build, and
concluded the premium was therefore sound. Both routes added the country premium, so agreeing
with each other confirmed nothing except that the same mistake had been made twice. A check
that cannot fail is not a check.

The fix strips the sovereign default spread from the risk-free rate before CAPM sees it, and
leaves the cost of *debt* on the full quoted yield, because a company really does borrow at a
spread over the actual government bond its lenders could buy instead. The two routes now
agree within about 70bp rather than disagreeing by 250bp, which is a real cross-check.

| | Before | After |
|---|---|---|
| Median WACC | 13.34% | **11.17%** |
| Median DCF upside | -74.7% | **-59.9%** |
| Companies above market | 2 / 33 | **8 / 32** |
| Within +/-30% of market | 4 / 33 | **9 / 32** |

### An earlier correction, kept for the record

Before the WACC finding, terminal reinvestment was made consistent with terminal growth
(`reinvestment rate = g / ROIC`), but only in the terminal year: the explicit forecast years
still held capex at its historical share of revenue while growth faded down underneath them,
the same incoherence repeated five times. Capex now fades toward its terminal-consistent
level across the whole horizon. That was a genuine fix, moving Reliance from an unusable 22
rupees a share to 323, and the reinvestment identity behind it is tested directly. It barely
moved the universe-wide median, which is what first suggested the problem lay somewhere other
than the cash flows.

### The independent cross-check, and what it still says

Stage 5 prices each company off what the market pays for its sector peers rather than off a
forecast of its own cash flows, so it shares almost none of the DCF's machinery and no
terminal-value mechanism at all. It is unaffected by the WACC fix, since multiples carry no
discount rate, which makes it a fixed benchmark across the whole investigation.

| | DCF (before) | DCF (after) | Comparables |
|---|---|---|---|
| Median upside | -74.7% | -59.9% | **+2.9%** |
| TCS | -26.8% | -2.5% | -9.8% |

That two methods built from the same statements moved apart rather than together is what
ruled out a data or arithmetic bug from the start: a bad input would have dragged both in the
same direction. After the fix, TCS's two methods agree within 7% of the current price, where
before they were 17 points apart.

### What is still wrong, and is being reported rather than fixed

A -59.9% median is still a long way from fair value, and the remaining gap is **not** claimed
to be resolved. It concentrates in two groups:

**Capital-intensive companies earning less on capital than their cost of capital.** UltraTech
earns about 10% on invested capital against an 11-13% WACC. The reinvestment identity then
charges it roughly 88% of NOPAT to fund GDP-rate growth, leaving almost no free cash and a
near-zero value. The arithmetic is internally consistent and the economics are defensible
(growth funded below the cost of capital does destroy value), but a DCF is arguably the wrong
instrument for a cyclical valued on mid-cycle or replacement-cost logic. Tata Steel, Hindalco,
JSW Steel and NTPC all fail the same way. The engine flags ROIC below WACC on every affected
company rather than printing the number quietly.

**Richly-rated consumer franchises.** Hindustan Unilever, Asian Paints, Britannia and Titan
still read 70-85% below market. Here the original terminal-value intuition may well be part
of the answer, even though it was wrong as a universe-wide explanation: the market pays for
brand durability and optionality beyond a five-year horizon that a perpetuity formula fed a
capped growth rate does not capture.

There is also a broader possibility this engine cannot settle on its own evidence: that a
CAPM cost of equity built on Indian inputs is simply higher than the discount rate Indian
equity investors actually apply, and that the Nifty trades at multiples no defensible DCF
reproduces. That is a claim about the market, not about the model, and it is not one to make
from 32 companies and four years of history.

What has deliberately **not** happened is tuning an assumption until the numbers agree. The
equity risk premium was not lowered to close the gap; a specific, identifiable double count
was found and corrected, and the gap that remained afterward is reported as it is.

## Roadmap

Stages 1 to 7 run end to end. The residual DCF-versus-comparables gap documented in
Calibration is the current state of the module rather than a defect hidden before Stage 7
landed: every scenario, sensitivity table and Monte Carlo trial is built on the same DCF and
inherits whatever remains of it, which is exactly why the Stage 7 summary reports a range
across methods instead of a single number.

What would move this further, in rough order of how much it would settle:

- **A multi-stage terminal value**: an explicit high-growth, transition and mature phase
  rather than a single perpetuity rate, or an exit-multiple terminal value (already built in
  Stage 4) fed by Stage 5 sector multiples for the companies where the gap is widest. This
  targets the richly-rated consumer franchises specifically.
- **An FCFE or excess-return model** for the capital-intensive companies earning below their
  cost of capital, where an unlevered DCF on mid-cycle economics is arguably the wrong
  instrument rather than a mis-calibrated one.
- **A longer history than four years**, which is the single constraint behind the most
  limitations listed below.

## Known limitations

1. **Four years of history is thin.** Trend classification splits the history in half, so
   with four years each half is two observations. The engine warns; it does not pretend.
2. **yfinance is an unofficial interface.** It can change or return nothing without notice.
   Every call is treated as capable of failing.
3. **The equity risk premium and the sovereign default spread are estimates, not
   measurements.** Both are stated as named assumptions because there is no public feed for
   either the way there is for a bond yield. They now also interact: the premium is a total
   one and the spread is what makes the risk-free rate it sits on default-free, so changing
   one without the other reintroduces the double count described in Calibration.
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
9. **Quick DCF has no comparables and, outside USD/INR, no calibrated market assumptions.**
   Both are disclosed in the app rather than hidden, but a Quick DCF result for a company in
   an uncalibrated currency is indicative only, more so than the limitations above that apply
   everywhere in this engine.
10. **The DCF still reads below market, and the remaining gap is unexplained.** A median of
    -59.9% after the country-risk fix is not a resolved model. Calibration sets out where it
    concentrates and what would likely move it; nothing here should be read as a fair value
    on the strength of the DCF alone.
11. **Monte Carlo samples the model's inputs, not the model's structure.** It answers "how
    much does the answer move across plausible inputs", never "is the model right". Every
    trial inherits the same terminal-value mechanism, so a distribution can be tight and
    wrong together, and a tight distribution is not evidence of a good valuation.
12. **The growth/margin correlation is assumed, not measured.** Four years cannot support a
    correlation estimate. It is set at 0.35 and stated; a materially different true value
    would change the width of the tails without changing the median much.
