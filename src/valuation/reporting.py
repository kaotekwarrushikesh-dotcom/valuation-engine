"""Build the Stage 1 report: the assumption set behind every company, and its reasoning.

This is deliberately not a valuation report, because there is no valuation yet. It is the
document an analyst would circulate *before* running a DCF, so that the assumptions get
argued with while they are still cheap to change. A share price printed on top of
unexamined assumptions is a false comfort, and reviewing the inputs after the answer exists
is how a model gets talked into its conclusion.
"""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.valuation import assumptions as asmp
from src.valuation import historical
from src.valuation.data_bridge import data_quality_report, load_history
from src.valuation.fcff import FCFF_LINES, build_fcff, validate_fcff
from src.valuation.forecasting import build_forecast, forecast_vs_history
from src.valuation.universe import INDIA_MARKET, NIFTY_UNIVERSE

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#6b6b6b")
RULE = colors.HexColor("#d4d4d4")
BAND = colors.HexColor("#f4f4f4")
ACCENT = colors.HexColor("#0f4c3a")

CONFIDENCE_COLOUR = {
    "high": colors.HexColor("#1e7a4b"),
    "medium": colors.HexColor("#b8860b"),
    "low": colors.HexColor("#c1621f"),
}


def styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=21, textColor=INK, spaceAfter=4),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=10.5, textColor=MUTED, spaceAfter=14),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14.5, textColor=ACCENT, spaceBefore=10, spaceAfter=7),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11, textColor=INK, spaceBefore=9, spaceAfter=4),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5, leading=13.5, textColor=INK, alignment=TA_LEFT, spaceAfter=6),
        "small": ParagraphStyle("s", parent=base["Normal"], fontSize=7.8, leading=10.5, textColor=MUTED, spaceAfter=3),
        "rationale": ParagraphStyle("r", parent=base["Normal"], fontSize=8.2, leading=11.5, textColor=INK, leftIndent=10, spaceAfter=5),
    }


def _table(data: list[list], widths: list[float], right_from: int = 1, font: float = 7.6) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle([
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
            ("FONT", (0, 1), (-1, -1), "Helvetica", font),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("ALIGN", (right_from, 0), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
            ("GRID", (0, 0), (-1, -1), 0.25, RULE),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    return t


def pct(v, dp: int = 1) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.{dp}%}"


def esc(text: str) -> str:
    """Escape text for reportlab's mini-HTML parser.

    Rationale strings contain "D&A", which the parser reads as the start of an entity and
    renders as "D&A;". Escaping the ampersand is the fix; the tags this module adds itself
    are written after escaping.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def overall_confidence(a: asmp.ForecastAssumptions) -> str:
    """A company's assumption set is only as strong as its weakest driver."""
    levels = [x.confidence for x in a.detail.values()]
    if levels.count("low") >= 3:
        return "low"
    if "low" in levels or levels.count("medium") >= 4:
        return "medium"
    return "high"


def analyse_universe(data_dir: Path) -> tuple[dict, pd.DataFrame, list]:
    """Run Stage 1 across the universe."""
    results, rows, blocked = {}, [], []

    for full_ticker, (name, sector) in NIFTY_UNIVERSE.items():
        ticker = full_ticker.replace(".NS", "")
        try:
            hist = load_history(ticker, data_dir)
            quality = data_quality_report(hist, ticker)
            if not quality["usable"]:
                blocked.append((ticker, name, quality["blocking"][0]))
                continue

            analysis = historical.analyse(hist)
            a = asmp.derive(hist, analysis, ticker,
                            nominal_gdp_growth=INDIA_MARKET["nominal_gdp_growth"])
            fc = build_forecast(hist, a, ticker)
            fcff = build_fcff(fc)
            errors = validate_fcff(fcff.frame)
            if errors:
                blocked.append((ticker, name, f"FCFF failed validation: {errors[0]}"))
                continue
        except Exception as exc:  # noqa: BLE001 - report and continue
            blocked.append((ticker, name, str(exc)))
            continue

        results[ticker] = {"name": name, "sector": sector, "hist": hist,
                           "analysis": analysis, "assumptions": a, "quality": quality,
                           "forecast": fc, "fcff": fcff,
                           "notes": forecast_vs_history(hist, fc) + fcff.warnings}
        rows.append({
            "ticker": ticker, "company": name, "sector": sector,
            "years": quality["years"],
            "growth_start": a.revenue_growth_path[0],
            "terminal_growth": a.terminal_growth,
            "ebitda_margin": a.ebitda_margin_path[0],
            "capex_pct": a.capex_pct_revenue,
            "nwc_pct": a.nwc_pct_revenue,
            "tax_rate": a.tax_rate,
            "confidence": overall_confidence(a),
            "growth_read": analysis["trends"]["revenue_growth"].classification,
            "margin_read": analysis["trends"]["ebitda_margin"].classification,
            "fcff_y1": float(fcff.frame["fcff"].iloc[0]),
            "fcff_terminal": float(fcff.frame["fcff"].iloc[-1]),
            "fcff_margin": float(fcff.frame["fcff_margin"].mean()),
            "reinvestment": float(((fcff.frame["capex"] + fcff.frame["change_in_nwc"])
                                   / fcff.frame["nopat"]).mean()),
        })

    summary = pd.DataFrame(rows).sort_values(["sector", "ticker"]).reset_index(drop=True)
    return results, summary, blocked


def summary_table(summary: pd.DataFrame, width: float) -> Table:
    header = ["Company", "Sector", "Yrs", "Growth", "Term g", "EBITDA", "Capex", "NWC", "Tax", "Conf."]
    rows = [header]
    for _, r in summary.iterrows():
        rows.append([
            f"{r['ticker']}", r["sector"], str(r["years"]),
            pct(r["growth_start"]), pct(r["terminal_growth"]), pct(r["ebitda_margin"]),
            pct(r["capex_pct"]), pct(r["nwc_pct"]), pct(r["tax_rate"]), r["confidence"],
        ])

    w = [width * x for x in [0.135, 0.175, 0.05, 0.095, 0.09, 0.09, 0.085, 0.09, 0.08, 0.11]]
    t = _table(rows, w, right_from=2)

    # Colour the confidence column so a weak assumption set is visible at a glance.
    for i, (_, r) in enumerate(summary.iterrows(), start=1):
        t.setStyle(TableStyle([("TEXTCOLOR", (9, i), (9, i), CONFIDENCE_COLOUR[r["confidence"]])]))
    return t


def company_section(ticker: str, entry: dict, st: dict, width: float) -> list:
    hist, a, analysis = entry["hist"], entry["assumptions"], entry["analysis"]
    flow = []

    first, last = int(hist["fiscal_year"].iloc[0]), int(hist["fiscal_year"].iloc[-1])
    flow.append(Paragraph(f"{entry['name']} ({ticker})", st["h1"]))
    flow.append(Paragraph(
        f"{entry['sector']}  |  FY{first} to FY{last}  |  INR millions  |  "
        f"assumption confidence: {overall_confidence(a)}", st["small"]))
    flow.append(Spacer(1, 4))

    # History
    cols = [("fiscal_year", "FY", lambda v: str(int(v))),
            ("revenue", "Revenue", lambda v: f"{v:,.0f}"),
            ("ebitda", "EBITDA", lambda v: f"{v:,.0f}"),
            ("ebit", "EBIT", lambda v: f"{v:,.0f}"),
            ("dep_amort", "D&A", lambda v: f"{v:,.0f}"),
            ("capex", "Capex", lambda v: f"{v:,.0f}"),
            ("nwc", "NWC", lambda v: f"{v:,.0f}"),
            ("change_in_nwc", "dNWC", lambda v: "n/a" if pd.isna(v) else f"{v:,.0f}"),
            ("net_debt", "Net debt", lambda v: f"{v:,.0f}"),
            ("revenue_growth", "Growth", pct),
            ("ebitda_margin", "EBITDA %", pct),
            ("capex_pct_revenue", "Capex %", pct)]
    data = [[label for _, label, _ in cols]]
    for _, row in hist.iterrows():
        data.append([f(row[k]) for k, _, f in cols])
    flow.append(_table(data, [width / len(cols)] * len(cols)))
    flow.append(Spacer(1, 7))

    # The read
    flow.append(Paragraph("What the history says", st["h2"]))
    for line in historical.summary_lines(analysis):
        flow.append(Paragraph(f"&bull; {esc(line)}", st["body"]))

    # Assumptions with reasoning
    flow.append(Paragraph("Forecast assumptions and why", st["h2"]))
    order = ["revenue_growth_start", "revenue_growth_path", "ebitda_margin", "da_pct_revenue",
             "capex_pct_revenue", "nwc_pct_revenue", "tax_rate", "terminal_growth"]
    for name in order:
        item = a.detail.get(name)
        if not item:
            continue
        tag = " [OVERRIDE]" if item.source == "override" else ""
        flow.append(Paragraph(
            f"<b>{esc(name.replace('_', ' '))}</b>: {item.value:.4g} "
            f"<font color='#{CONFIDENCE_COLOUR[item.confidence].hexval()[2:]}'>({item.confidence})</font>{tag}",
            st["body"]))
        flow.append(Paragraph(esc(item.rationale), st["rationale"]))

    # Resulting driver path
    flow.append(Paragraph("Resulting forecast drivers", st["h2"]))
    frame = a.as_frame(last)
    drows = [["Year", "Rev growth", "EBITDA margin", "D&A / rev", "Capex / rev", "NWC / rev", "Tax"]]
    for _, r in frame.iterrows():
        drows.append([f"{int(r['year'])}E", pct(r["revenue_growth"]), pct(r["ebitda_margin"]),
                      pct(r["da_pct_revenue"]), pct(r["capex_pct_revenue"]),
                      pct(r["nwc_pct_revenue"]), pct(r["tax_rate"])])
    flow.append(_table(drows, [width * x for x in [0.12, 0.16, 0.17, 0.14, 0.15, 0.14, 0.12]]))

    # Stage 2: the FCFF build, every component visible
    fcff = entry["fcff"]
    f = fcff.frame
    flow.append(Paragraph("Free cash flow to the firm", st["h2"]))
    frows = [[""] + [f"{int(y)}E" for y in f["year"]]]
    for key, label in FCFF_LINES:
        sign = -1.0 if key in ("tax_on_ebit", "capex", "change_in_nwc") else 1.0
        frows.append([label] + [f"{v * sign:,.0f}" for v in f[key]])

    ft = _table(frows, [width * 0.26] + [width * 0.148] * len(f))
    ft.setStyle(TableStyle([
        ("FONT", (0, len(frows) - 1), (-1, -1), "Helvetica-Bold", 7.6),
        ("FONT", (0, 3), (-1, 3), "Helvetica-Bold", 7.6),
        ("LINEABOVE", (0, len(frows) - 1), (-1, len(frows) - 1), 0.8, ACCENT),
    ]))
    flow.append(ft)
    flow.append(Paragraph(
        "Every figure is a cash impact, so a negative number is an outflow. Tax is charged on "
        "unlevered EBIT and interest is absent: the financing effect belongs in the discount "
        "rate, and counting it here as well would value the debt tax shield twice.", st["small"]))

    if entry["notes"]:
        flow.append(Spacer(1, 4))
        flow.append(Paragraph("What to argue with", st["h2"]))
        for n in entry["notes"]:
            flow.append(Paragraph(f"&bull; {esc(n)}", st["body"]))

    if entry["quality"]["warnings"]:
        flow.append(Spacer(1, 4))
        flow.append(Paragraph("Data quality: " + "; ".join(entry["quality"]["warnings"]) + ".", st["small"]))

    flow.append(PageBreak())
    return flow


def build_pdf(results: dict, summary: pd.DataFrame, blocked: list, out: Path) -> None:
    st = styles()
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=13 * mm, rightMargin=13 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title="Valuation Engine, Stage 1", author="Module 2")
    width = doc.width
    flow = []

    flow.append(Paragraph("Institutional-Style Valuation Engine", st["title"]))
    flow.append(Paragraph(
        f"Stages 1 and 2: assumptions, forecast and FCFF &bull; {len(results)} Nifty companies "
        "&bull; INR &bull; sources: NSE filings, FRED, Nifty 50", st["subtitle"]))

    flow.append(Paragraph("What this document is, and what it is not", st["h1"]))
    flow.append(Paragraph(
        "This is not a valuation. There is no fair value in it, because nothing has been "
        "discounted: WACC is the next stage. What it contains is everything a discounted cash "
        "flow stands on, published on its own so that it can be argued with while it is still "
        "cheap to change. Reviewing inputs after a share price exists is how a model gets talked "
        "into its conclusion.", st["body"]))
    flow.append(Paragraph(
        "Stage 1 derives the assumptions. Stage 2 turns them into a forecast income statement and "
        "then into free cash flow to the firm, with every component of the build shown as its own "
        "line. Tax is charged on unlevered EBIT and interest never appears, because the financing "
        "effect belongs in the discount rate; putting it in the cash flow as well would value the "
        "debt tax shield twice, which is the single most common way a DCF flatters itself.",
        st["body"]))
    flow.append(Paragraph(
        "Every assumption here is produced by a stated rule applied to the company's own "
        "reported history, and each one carries the reasoning behind it and a confidence level. "
        "Nothing is a hard-coded growth rate. Where a value is set by hand it is tagged as an "
        "override and reports what the derived value would have been, so a judgement call can "
        "never be mistaken for something the evidence produced.", st["body"]))

    flow.append(Paragraph("The assumption rules", st["h1"]))
    rules = [
        ["Assumption", "Rule", "What it is defending against"],
        ["Revenue growth (start)", "Recent 3-year CAGR, half the excess above 25% removed",
         "One noisy year setting the path; exceptional growth rarely survives a full horizon"],
        ["Growth path", "Geometric fade from start to terminal",
         "A flat path then a drop to terminal implies a cliff no business experiences"],
        ["Terminal growth", "Lower of nominal GDP and current growth",
         "Assuming a slow-growing business re-accelerates forever, which loads the terminal value"],
        ["EBITDA margin", "Recent 3-year average, held flat",
         "Extending margin expansion, the most common way a DCF is talked upward"],
        ["D&A / revenue", "Full-period median",
         "One impairment setting the run rate"],
        ["Capex / revenue", "Full-period median",
         "Understating capex, the easiest way to manufacture free cash flow"],
        ["Working capital / revenue", "Recent 3-year average",
         "A median spanning a structural shift, creating a phantom first-year cash flow"],
        ["Tax rate", "Median effective rate, bounded 10-35%",
         "Projecting temporary credits into perpetuity"],
    ]
    rule_rows = [rules[0]] + [[r[0], Paragraph(r[1], st["small"]), Paragraph(r[2], st["small"])]
                              for r in rules[1:]]
    flow.append(_table(rule_rows, [width * 0.21, width * 0.35, width * 0.44], right_from=99))

    flow.append(PageBreak())

    # Universe summary
    flow.append(Paragraph("Derived assumptions across the universe", st["h1"]))
    flow.append(summary_table(summary, width))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(
        "Growth is the first forecast year. Term g is the perpetuity rate. Confidence is the "
        "weakest link in each company's assumption set, not an average, because a forecast is "
        "only as strong as its softest driver.", st["small"]))

    flow.append(PageBreak())
    flow.append(Paragraph("Forecast free cash flow across the universe", st["h1"]))
    crows = [["Company", "Sector", "FCFF yr 1", "FCFF terminal", "FCFF margin", "Reinvestment"]]
    for _, r in summary.sort_values("fcff_margin", ascending=False).iterrows():
        crows.append([r["ticker"], r["sector"], f"{r['fcff_y1']:,.0f}",
                      f"{r['fcff_terminal']:,.0f}", pct(r["fcff_margin"]), pct(r["reinvestment"], 0)])
    flow.append(_table(crows, [width * x for x in [0.15, 0.21, 0.17, 0.18, 0.15, 0.14]]))
    flow.append(Paragraph(
        "Reinvestment is capex plus the working capital movement as a share of NOPAT. Above 100% "
        "the company is investing more than it earns after tax and is funding growth externally, "
        "which is normal for cement, metals and utilities and would be a warning for IT services. "
        "All figures are INR millions and none of them has been discounted yet.", st["small"]))

    if blocked:
        flow.append(Spacer(1, 8))
        flow.append(Paragraph("Excluded, and why", st["h2"]))
        for ticker, name, reason in blocked:
            flow.append(Paragraph(f"&bull; <b>{ticker}</b> ({name}): {reason}.", st["body"]))
        flow.append(Paragraph(
            "These are excluded rather than filled in. A forecast built on two observations "
            "would look identical on the page to one built on ten.", st["small"]))

    # Sector view
    sector = (summary.groupby("sector")
              .agg(n=("ticker", "count"), growth=("growth_start", "median"),
                   margin=("ebitda_margin", "median"), capex=("capex_pct", "median"),
                   nwc=("nwc_pct", "median"))
              .sort_values("margin", ascending=False).reset_index())
    srows = [["Sector", "Companies", "Median growth", "Median EBITDA margin", "Median capex/rev", "Median NWC/rev"]]
    for _, r in sector.iterrows():
        srows.append([r["sector"], str(int(r["n"])), pct(r["growth"]), pct(r["margin"]),
                      pct(r["capex"]), pct(r["nwc"])])
    flow.append(KeepTogether([
        Paragraph("Sector medians", st["h1"]),
        _table(srows, [width * x for x in [0.24, 0.13, 0.16, 0.20, 0.15, 0.15]]),
        Paragraph(
            "The spread is the case for peer-relative comparison in the comparables stage: an 11% "
            "capex ratio is ordinary for cement and would be alarming for IT services.", st["small"]),
    ]))
    flow.append(PageBreak())

    for ticker in summary["ticker"]:
        flow.extend(company_section(ticker, results[ticker], st, width))

    # Limitations
    flow.append(Paragraph("Limitations", st["h1"]))
    for text in [
        "<b>Four to five years of history.</b> Indian companies are not on SEC EDGAR, and the "
        "available source is shorter than the ten years the US path gives. Trend classification "
        "splits the history in half, so each half here is two observations. Every company in this "
        "report carries that warning.",
        "<b>The rules are deliberately conservative.</b> Flat margins and a capped terminal growth "
        "rate will understate a genuinely improving business. The bias is toward not being talked "
        "into a valuation, and it is a bias, not neutrality.",
        "<b>The equity risk premium is an estimate.</b> Unlike a bond yield there is no feed for it. "
        f"{INDIA_MARKET['equity_risk_premium']:.1%} is used for India, including country risk, and it "
        "is stated as an assumption rather than buried in a discount rate.",
        "<b>Cash includes short-term investments.</b> Indian companies hold surplus in liquid funds "
        "rather than bank balances, so the narrow cash line understates liquidity and would turn "
        "net-cash companies into net-debt ones. The broader measure is used for both net debt and "
        "working capital.",
        "<b>Consolidated accounts include captive finance arms.</b> Bajaj Auto shows net debt because "
        "its lending subsidiary's borrowings consolidate into the parent. That is what the statements "
        "say, but it is financing-company debt sitting inside an industrial valuation, and it should "
        "be separated before the enterprise-to-equity bridge is trusted.",
        "<b>No accounting-quality checks.</b> One-off items, restatements and segment changes are not "
        "detected. The engine trusts the filed numbers.",
        "<b>Banks and NBFCs are absent by design.</b> FCFF and enterprise value assume debt is "
        "financing. For a lender it is raw material, so an unlevered DCF is the wrong instrument "
        "rather than an imprecise one.",
        "<b>No valuation is implied.</b> Nothing here has been discounted. A company with attractive "
        "assumptions is not thereby cheap; price has not entered the analysis.",
    ]:
        flow.append(Paragraph(f"&bull; {text}", st["body"]))

    doc.build(flow)


def main(project_root: Path) -> None:
    data_dir = project_root / "data" / "nifty"
    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)

    results, summary, blocked = analyse_universe(data_dir)
    summary.to_csv(out_dir / "stage1_2_drivers_all.csv", index=False)

    pdf = out_dir / "valuation_report.pdf"
    build_pdf(results, summary, blocked, pdf)

    print(f"Stage 1 across {len(results)} companies ({len(blocked)} excluded)\n")
    print(summary[["ticker", "sector", "years", "growth_start", "terminal_growth",
                   "ebitda_margin", "confidence"]].to_string(index=False))
    for ticker, name, reason in blocked:
        print(f"\nexcluded: {ticker} ({name}): {reason}")
    print(f"\nWrote:\n  {pdf}\n  {out_dir/'stage1_2_drivers_all.csv'}")
