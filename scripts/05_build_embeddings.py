#!/usr/bin/env python
"""CLI de build do indice semantico (SPEC-MODE2.md contrato 6.5).

python scripts/05_build_embeddings.py
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fominha.mode2.index import META_PATH, DEFAULT_PARQUET_PATH, build_semantic_index


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        build_semantic_index(DEFAULT_PARQUET_PATH)
    except FileNotFoundError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        sys.exit(1)

    with open(META_PATH) as f:
        meta = json.load(f)

    print(f"n_recipes: {meta['n_recipes']}")
    print(f"dims: {meta['dims']}")
    print(f"build_seconds: {meta['build_seconds']:.2f}")


if __name__ == "__main__":
    main()
