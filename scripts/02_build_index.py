#!/usr/bin/env python
"""CLI de build do indice TF-IDF (SPEC.md secao 6.4).

python scripts/02_build_index.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fominha.mode1.index import META_PATH, build_index


def main():
    start = time.perf_counter()
    build_index("data/processed/recipes.parquet")
    elapsed = time.perf_counter() - start

    with open(META_PATH) as f:
        meta = json.load(f)

    print(f"n_recipes: {meta['n_recipes']}")
    print(f"vocab_size: {meta['vocab_size']}")
    print(f"tempo de build: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
