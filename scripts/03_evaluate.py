#!/usr/bin/env python
"""CLI de avaliacao do Modo 1 (SPEC.md secao 6.4 / secao 8).

python scripts/03_evaluate.py --k 1 5 10 --n-eval 1000 --mask-frac 0.3 --seed 42

Grava reports/eval_mode1.json (contrato 6.5) e imprime as metricas legiveis.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fominha.eval.protocol import evaluate
from fominha.mode1.index import IndexNotBuiltError

REPORT_PATH = "reports/eval_mode1.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10],
                        help="valores de k para hit_rate@k (Protocolo A)")
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--mask-frac", type=float, default=0.3)
    parser.add_argument("--jaccard-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # precision@k fica travado em {5, 10} pela secao 8.2 do SPEC.
    try:
        result = evaluate(
            ks_hit=tuple(args.k),
            ks_prec=(5, 10),
            n_eval=args.n_eval,
            mask_frac=args.mask_frac,
            jaccard_threshold=args.jaccard_threshold,
            seed=args.seed,
        )
    except IndexNotBuiltError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Relatorio gravado em: {REPORT_PATH}")
    print(f"n_eval_queries: {result['n_eval_queries']}")
    print()
    print("Protocolo A (hit_rate@k):")
    for key, val in result["metrics"].items():
        if key.startswith("hit_rate"):
            print(f"  {key}: {val}")
    print("Protocolo B (precision@k):")
    for key, val in result["metrics"].items():
        if key.startswith("precision"):
            print(f"  {key}: {val}")
    print("Protocolo B - nº medio de relevantes por query:")
    for key, val in result["avg_relevant_per_query"].items():
        print(f"  relevantes{key}: {val}")


if __name__ == "__main__":
    main()
