"""Build the Stage 1 assumptions report across the Nifty universe."""

from pathlib import Path

from valuation_engine.reporting import main

if __name__ == "__main__":
    main(Path(__file__).parent)
