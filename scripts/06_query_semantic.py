#!/usr/bin/env python
"""CLI de query semantica manual (SPEC-MODE2.md contrato 6.5).

python scripts/06_query_semantic.py --query "something light with chicken" --k 10
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fominha.mode1.recommend import EmptyQueryError
from fominha.mode2 import SemanticIndexNotBuiltError
from fominha.mode2.search import search


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    try:
        results = search(args.query, k=args.k)
    except (EmptyQueryError, SemanticIndexNotBuiltError, ValueError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        sys.exit(1)

    for rank, res in enumerate(results, start=1):
        print(f"{rank}. {res.title} (recipe_id={res.recipe_id}, score={res.score:.4f})")
        print(f"   link: {res.link}")
        print()


if __name__ == "__main__":
    main()
