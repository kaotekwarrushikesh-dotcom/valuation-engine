"""Download the Nifty universe into Module 1's cleaned-statement schema."""

from pathlib import Path

from valuation_engine.nse_data import fetch_universe
from valuation_engine.universe import NIFTY_UNIVERSE

DATA_DIR = Path(__file__).parent / "data" / "nifty"


def main() -> None:
    ok, failed = fetch_universe(NIFTY_UNIVERSE, DATA_DIR)

    for ticker, name, years, span in ok:
        print(f"  {ticker:<16}{name:<28}{years:>2} yrs  {span}")
    for ticker, reason in failed:
        print(f"  {ticker:<16}FAILED: {reason}")

    print(f"\n{len(ok)}/{len(NIFTY_UNIVERSE)} companies fetched into {DATA_DIR}")


if __name__ == "__main__":
    main()
