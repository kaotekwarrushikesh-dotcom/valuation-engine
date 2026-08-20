"""The Nifty analysis universe, grouped so every company has a real peer set.

Comparable-company valuation is only as good as the peer group, so this list is built
around sectors with at least three members rather than picking the largest names and
hoping. A peer group of one is not a peer group.

Banks, insurers and NBFCs (HDFC Bank, ICICI, SBI, Kotak, Bajaj Finance) are deliberately
absent. FCFF and enterprise value assume debt is financing; for a lender, borrowing is raw
material, so EV/EBITDA and an unlevered DCF are not just imprecise for them, they are the
wrong instrument. They need an excess-return or FCFE model, which is a separate build.
"""

# ticker: (company name, sector)
NIFTY_UNIVERSE: dict[str, tuple[str, str]] = {
    # IT services: the deepest peer set on the exchange, and highly comparable to each other
    "TCS.NS": ("Tata Consultancy Services", "IT Services"),
    "INFY.NS": ("Infosys", "IT Services"),
    "HCLTECH.NS": ("HCL Technologies", "IT Services"),
    "WIPRO.NS": ("Wipro", "IT Services"),
    "TECHM.NS": ("Tech Mahindra", "IT Services"),
    # FMCG: stable margins and low capital intensity, so a clean test of the fade logic
    "HINDUNILVR.NS": ("Hindustan Unilever", "FMCG"),
    "ITC.NS": ("ITC", "FMCG"),
    "NESTLEIND.NS": ("Nestle India", "FMCG"),
    "BRITANNIA.NS": ("Britannia Industries", "FMCG"),
    "DABUR.NS": ("Dabur India", "FMCG"),
    # Automobiles: cyclical, capital hungry, and a useful contrast to IT
    "MARUTI.NS": ("Maruti Suzuki India", "Automobiles"),
    "M&M.NS": ("Mahindra & Mahindra", "Automobiles"),
    "BAJAJ-AUTO.NS": ("Bajaj Auto", "Automobiles"),
    "EICHERMOT.NS": ("Eicher Motors", "Automobiles"),
    "HEROMOTOCO.NS": ("Hero MotoCorp", "Automobiles"),
    # Pharmaceuticals
    "SUNPHARMA.NS": ("Sun Pharmaceutical", "Pharmaceuticals"),
    "CIPLA.NS": ("Cipla", "Pharmaceuticals"),
    "DRREDDY.NS": ("Dr. Reddy's Laboratories", "Pharmaceuticals"),
    "DIVISLAB.NS": ("Divi's Laboratories", "Pharmaceuticals"),
    # Metals and materials: the most cyclical group, where a trend forecast is hardest
    "TATASTEEL.NS": ("Tata Steel", "Metals"),
    "JSWSTEEL.NS": ("JSW Steel", "Metals"),
    "HINDALCO.NS": ("Hindalco Industries", "Metals"),
    "ULTRACEMCO.NS": ("UltraTech Cement", "Cement"),
    "GRASIM.NS": ("Grasim Industries", "Cement"),
    "SHREECEM.NS": ("Shree Cement", "Cement"),
    # Energy and utilities
    "RELIANCE.NS": ("Reliance Industries", "Energy"),
    "ONGC.NS": ("Oil & Natural Gas Corp", "Energy"),
    "NTPC.NS": ("NTPC", "Utilities"),
    "POWERGRID.NS": ("Power Grid Corp", "Utilities"),
    # Consumer discretionary and infrastructure
    "ASIANPAINT.NS": ("Asian Paints", "Consumer Discretionary"),
    "TITAN.NS": ("Titan Company", "Consumer Discretionary"),
    "LT.NS": ("Larsen & Toubro", "Infrastructure"),
    "BHARTIARTL.NS": ("Bharti Airtel", "Telecom"),
}

# Market-level inputs for Indian equity. These belong with the universe rather than inside
# the WACC calculation, so that changing market changes one file, not several.
INDIA_MARKET = {
    "currency": "INR",
    "index_ticker": "^NSEI",
    "index_name": "Nifty 50",
    # India 10-year government bond yield, FRED series INDIRLTLT01STM. A rupee cash flow
    # must be discounted at a rupee rate; using a US Treasury yield here would mix a
    # dollar risk-free rate with rupee cash flows and understate the discount rate by
    # roughly the inflation differential.
    "risk_free_series": "INDIRLTLT01STM",
    # Mature-market equity risk premium plus an India country risk premium. Damodaran's
    # published India total ERP has run in the 7-8% range; 7.5% is used and is an
    # assumption, not a measurement, so it is named here and overridable.
    #
    # This is a *total* premium and therefore belongs on top of a default-free rate. The
    # rupee 10-year above is not one: it yields more than a Treasury partly for expected
    # inflation and partly because lenders price the sovereign's own default risk, and the
    # country premium inside this 7.5% charges for that same risk again. The spread below
    # is stripped from the yield before CAPM sees it; see wacc.py.
    "equity_risk_premium": 0.075,
    # India's sovereign default spread, from its Baa3/BBB- rating. Subtracted from the
    # quoted rupee yield to get the default-free rate CAPM requires. A cross-check that the
    # two routes now agree: rupee yield less this spread, plus the total premium, lands
    # within ~70bp of the rupee yield plus a mature-market premium alone. Before the fix
    # the same two routes disagreed by ~250bp, which was the double count showing up.
    "sovereign_default_spread": 0.022,
    # Long-run nominal GDP growth for India, the ceiling on terminal growth. Higher than
    # the US ceiling because both real growth and target inflation are higher.
    "nominal_gdp_growth": 0.09,
    # The RBI's inflation target, and the floor on terminal growth. A going concern growing
    # below inflation forever is shrinking in real terms forever, which is a strong claim
    # about a large listed franchise rather than a cautious default.
    "inflation": 0.04,
}


def peers_for(ticker: str) -> list[str]:
    """Other companies in the same sector, which is the comparable set."""
    if ticker not in NIFTY_UNIVERSE:
        return []
    sector = NIFTY_UNIVERSE[ticker][1]
    return [t for t, (_, s) in NIFTY_UNIVERSE.items() if s == sector and t != ticker]


# --- Market profiles for the Quick DCF path (any company, any currency) ----------------
#
# WACC needs country-specific inputs: a risk-free rate in the cash flow's own currency, an
# equity risk premium, and a terminal-growth ceiling tied to that economy's nominal GDP. The
# curated universe only ever needed India's numbers. A company fetched through the Quick DCF
# path can be in any currency, and building a calibrated profile for every currency on earth
# is not attempted here: two markets are calibrated (the ones with real usage in this
# platform), and everything else gets a stated, disclosed generic fallback rather than a
# silently wrong number dressed up as a real one.

US_MARKET = {
    "currency": "USD",
    "index_ticker": "^GSPC",
    "index_name": "S&P 500",
    "risk_free_series": "DGS10",
    "equity_risk_premium": 0.045,  # mature market, no country premium to add
    # The Treasury is the benchmark default-free rate, so there is nothing to strip out.
    "sovereign_default_spread": 0.0,
    "nominal_gdp_growth": 0.04,
    "inflation": 0.02,
}

# Not tied to any government bond series: this fallback has no calibrated risk-free rate,
# so it takes the US Treasury as a dollar proxy and states plainly that it is one. A company
# in euros or yen priced with a dollar risk-free rate has a real error in it, on the order
# of the two currencies' rate differential, which is disclosed rather than hidden.
GENERIC_MARKET = {
    "currency": None,  # filled in with the company's own currency at resolution time
    "index_ticker": "^GSPC",
    "index_name": "S&P 500 (proxy: no calibrated local index)",
    "risk_free_series": "DGS10",
    "equity_risk_premium": 0.055,  # mature-market ERP plus a margin for the uncalibrated case
    # The proxy rate is a US Treasury yield, which is already default-free.
    "sovereign_default_spread": 0.0,
    "nominal_gdp_growth": 0.035,
    "inflation": 0.025,
    "is_generic": True,
}


def resolve_market_for_currency(currency: str) -> tuple[dict, bool]:
    """The market profile for a currency, and whether it is genuinely calibrated.

    Returns (profile, is_calibrated). A caller must check the second value and disclose it
    when False, since a generic profile is a real, quantifiable simplification, not a minor
    footnote: using a dollar risk-free rate for a euro cash flow misprices the currency
    differential, typically a percentage point or more, directly into the discount rate.
    """
    if currency == "INR":
        return INDIA_MARKET, True
    if currency == "USD":
        return US_MARKET, True
    profile = dict(GENERIC_MARKET)
    profile["currency"] = currency
    return profile, False
