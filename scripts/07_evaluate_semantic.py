#!/usr/bin/env python
"""CLI de avaliacao do Modo 2 - Protocolo C (SPEC-MODE2.md contrato 6.5).

python scripts/07_evaluate_semantic.py --k 1 5 10 --n-eval 1000 --mask-frac 0.3 --seed 42

Grava reports/eval_mode2.json (contrato 6.4) e imprime a tabela comparativa
Modo 1 vs Modo 2.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fominha.eval.protocol_mode2 import evaluate_mode2

REPORT_PATH = "reports/eval_mode2.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--mask-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = evaluate_mode2(
        ks=tuple(args.k),
        n_eval=args.n_eval,
        mask_frac=args.mask_frac,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Relatorio gravado em: {REPORT_PATH}")
    print(f"n_eval_queries: {result['n_eval_queries']}")
    print()
    print(f"{'k':<6}{'mode1 hit_rate@k':<20}{'mode2 hit_rate@k':<20}")
    for k in args.k:
        m1 = result["mode1_reference"].get(f"hit_rate@{k}", "n/a")
        m2 = result["metrics"].get(f"mode2_hit_rate@{k}", "n/a")
        print(f"{k:<6}{m1:<20}{m2:<20}")


if __name__ == "__main__":
    main()
