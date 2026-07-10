"""Funcao publica recommend() do Modo 1 (SPEC.md contrato 6.3, fluxo 7.2)."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from fominha.mode1.index import load_index
from fominha.normalize import normalize_ingredients

_index_cache = None


class EmptyQueryError(Exception):
    """Query normaliza para zero tokens canonicos (edge case E-04)."""


@dataclass(frozen=True)
class Recommendation:
    recipe_id: int
    title: str
    score: float
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    link: str


def _get_index():
    global _index_cache
    if _index_cache is None:
        _index_cache = load_index()
    return _index_cache


def _reset_index_cache() -> None:
    """Limpa o cache do modulo. Uso interno para testes."""
    global _index_cache
    _index_cache = None


def recommend(ingredients: list[str], k: int = 10) -> list[Recommendation]:
    if isinstance(k, bool) or not isinstance(k, int) or not (1 <= k <= 100):
        raise ValueError(f"k deve ser um inteiro entre 1 e 100, recebido: {k!r}")

    canonical = normalize_ingredients(ingredients)
    if not canonical:
        raise EmptyQueryError(
            "Query vazia apos normalizacao. Exemplos de input valido: "
            "['chicken breast', 'rice', 'garlic'] ou ['2 cups flour', 'sugar']."
        )

    vectorizer, matrix, df = _get_index()

    query_doc = " ".join(canonical)
    query_vec = vectorizer.transform([query_doc])
    scores = cosine_similarity(query_vec, matrix).ravel()

    order = np.argsort(-scores)
    query_set = set(canonical)

    results = []
    for idx in order:
        if len(results) >= k:
            break
        score = float(scores[idx])
        if score == 0.0:
            break

        row = df.iloc[idx]
        recipe_canonical = list(row["ingredients_canonical"])
        recipe_set = set(recipe_canonical)

        matched = [token for token in canonical if token in recipe_set]
        missing = [token for token in recipe_canonical if token not in query_set]

        results.append(Recommendation(
            recipe_id=int(row["recipe_id"]),
            title=str(row["title"]),
            score=score,
            matched_ingredients=matched,
            missing_ingredients=missing,
            link=str(row["link"]),
        ))

    return results
