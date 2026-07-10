#!/usr/bin/env python
"""CLI de query manual do Modo 1 (SPEC.md secao 6.4).

python scripts/04_query.py --ingredients "chicken, rice, garlic" --k 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fominha.mode1.index import IndexNotBuiltError
from fominha.mode1.recommend import EmptyQueryError, recommend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingredients", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    ingredients = [item.strip() for item in args.ingredients.split(",") if item.strip()]

    try:
        results = recommend(ingredients, k=args.k)
    except (EmptyQueryError, IndexNotBuiltError, ValueError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("Nenhuma receita encontrada.")
        return

    for rank, rec in enumerate(results, start=1):
        print(f"{rank}. {rec.title} (recipe_id={rec.recipe_id}, score={rec.score:.4f})")
        print(f"   matched: {', '.join(rec.matched_ingredients)}")
        print(f"   missing: {', '.join(rec.missing_ingredients)}")
        print(f"   link: {rec.link}")
        print()


if __name__ == "__main__":
    main()
