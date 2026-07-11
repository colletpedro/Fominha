"""Funcao publica search() do Modo 2 (SPEC-MODE2.md contrato 6.2, fluxo 7.2)."""

from dataclasses import dataclass

from fominha.mode1.recommend import EmptyQueryError
from fominha.mode2.index import load_semantic_index

_index_cache = None


@dataclass(frozen=True)
class SemanticResult:
    recipe_id: int
    title: str
    score: float
    link: str


def _get_index():
    global _index_cache
    if _index_cache is None:
        _index_cache = load_semantic_index()
    return _index_cache


def _reset_index_cache() -> None:
    """Limpa o cache do modulo. Uso interno para testes."""
    global _index_cache
    _index_cache = None


def search(query: str, k: int = 10) -> list[SemanticResult]:
    if isinstance(k, bool) or not isinstance(k, int) or not (1 <= k <= 100):
        raise ValueError(f"k deve ser um inteiro entre 1 e 100, recebido: {k!r}")

    stripped = query.strip()
    if not stripped:
        raise EmptyQueryError(
            "Query vazia. Exemplo de input valido: "
            "'something light and quick with chicken'."
        )

    model, index, df = _get_index()

    vec = model.encode([stripped], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, ids = index.search(vec, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        row = df.iloc[int(idx)]
        results.append(SemanticResult(
            recipe_id=int(row["recipe_id"]),
            title=str(row["title"]),
            score=float(score),
            link=str(row["link"]),
        ))

    return results
