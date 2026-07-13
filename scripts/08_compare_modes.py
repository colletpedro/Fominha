#!/usr/bin/env python
"""CLI do comparativo Modo 1 vs Modo 2 (SPEC-MODE2.md contrato 6.5).

python scripts/08_compare_modes.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fominha.eval.protocol_mode2 import run_comparison

QUERIES_PATH = "eval_queries/nl_queries.json"
REPORT_PATH = "reports/comparison_mode1_vs_mode2.md"


def main():
    markdown = run_comparison(QUERIES_PATH, k=5)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(markdown)
        if not markdown.endswith("\n"):
            f.write("\n")

    print(f"Comparativo gravado em: {REPORT_PATH}")


if __name__ == "__main__":
    main()
