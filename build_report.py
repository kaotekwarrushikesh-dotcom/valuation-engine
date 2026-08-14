"""Build the Stage 1 assumptions report across the Nifty universe."""

from pathlib import Path

from src.valuation.reporting import main

if __name__ == "__main__":
    main(Path(__file__).parent)
