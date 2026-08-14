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
    "equity_risk_premium": 0.075,
    # Long-run nominal GDP growth for India, the ceiling on terminal growth. Higher than
    # the US ceiling because both real growth and target inflation are higher.
    "nominal_gdp_growth": 0.09,
}


def peers_for(ticker: str) -> list[str]:
    """Other companies in the same sector, which is the comparable set."""
    if ticker not in NIFTY_UNIVERSE:
        return []
    sector = NIFTY_UNIVERSE[ticker][1]
    return [t for t, (_, s) in NIFTY_UNIVERSE.items() if s == sector and t != ticker]
