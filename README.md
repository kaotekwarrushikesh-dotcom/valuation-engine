# Module 2: Institutional-Style Valuation Engine

> Estimating intrinsic value using DCF, market multiples and scenario-based valuation analysis.

Takes cleaned historical financials, builds defensible forecasts, and works through to a
fair value per share. Built for **Nifty (NSE-listed) companies**, in INR.

**Status: Stages 1 and 2 complete.** Assumptions, the forecast and FCFF are built and
tested. Nothing is discounted yet, so there is no fair value: WACC is the next stage. This
README says so rather than implying otherwise. See [Roadmap](#roadmap).

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

## Roadmap

Stages 1 and 2 are done. The remaining stages build on these cash flows:

- **Stage 3** WACC: CAPM cost of equity, beta by regression against the Nifty 50, after-tax
  cost of debt, market-value capital structure weights
- **Stage 4** DCF: discounting, terminal value by both perpetual growth and exit multiple,
  the enterprise-to-equity bridge, and a flag when terminal value dominates enterprise value
- **Stage 5** Comparable company analysis against the sector peer set
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
