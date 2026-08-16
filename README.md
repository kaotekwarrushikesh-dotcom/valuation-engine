# Module 2: Institutional-Style Valuation Engine

> Estimating intrinsic value using DCF, market multiples and scenario-based valuation analysis.

Takes cleaned historical financials, builds defensible forecasts, and works through to a
fair value per share. Built for **Nifty (NSE-listed) companies**, in INR.

**Status: Stages 1 to 4 built, and not yet calibrated.** Assumptions, forecast, FCFF, WACC
and the DCF all run and are tested. They produce a fair value per share, but that number is
systematically below the market for every company in the universe, so **the valuations are
not published and should not be quoted**. What went wrong and what fixes it are written up
in [Calibration](#calibration-the-model-reads-low-and-why) below, because a model that is
wrong in a known direction is more useful than one that is quietly tuned until it agrees.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download the universe (33 Nifty companies):

```bash
.venv/bin/python fetch_nifty.py
```

Run the stages for one company:

```bash
.venv/bin/python stage1_historical.py --ticker RELIANCE
```

```bash
.venv/bin/python stage2_forecast.py --ticker RELIANCE
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
│   ├── reporting.py            the PDF report
│   └── market_data.py          prices, share counts, risk-free rate
├── tests/
├── outputs/
├── fetch_nifty.py
└── stage1_historical.py
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

**What is still open.** Whether the fix is a richer terminal-value treatment (a multi-stage
fade with an explicit high-growth, transition and mature phase rather than one perpetuity
growth rate), sector-aware terminal multiples via comparables, or simply reporting the DCF
fair value alongside a comparables-based one and letting the two disagree on the record,
is the next real decision, not the closing of this section.

## Roadmap

Stages 1 to 4 run end to end and produce a fair value per share, with the calibration
question above still open.

- **Stage 5** Comparable company analysis against the sector peer set, which is independent
  of the terminal-growth mechanism and gives a second read that does not share the DCF's
  failure mode
- **Stage 6** Bull/base/bear scenarios and sensitivity tables
- **Stage 7** Monte Carlo, driver attribution, and the final valuation summary

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
