#!/usr/bin/env python
"""CLI de ingestao (SPEC.md secao 6.4).

python scripts/01_ingest.py --raw data/raw/full_dataset.csv --n-recipes 100000 --seed 42
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fominha.ingest import ingest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--n-recipes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/processed/recipes.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        df = ingest(args.raw, args.n_recipes, seed=args.seed)
    except FileNotFoundError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, engine="pyarrow", index=False)
    print(f"Dataset tratado gravado em: {args.out} ({len(df)} receitas)")


if __name__ == "__main__":
    main()
