"""Testa a MECANICA do protocolo de avaliacao (SPEC.md secao 8), nao os numeros
absolutos (que dependem do dataset real). Usa fixtures pequenas para rodar rapido.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fominha.eval import protocol as protocol_module
from fominha.eval.protocol import build_masked_queries, _jaccard, evaluate
from fominha.mode1 import index as index_module
from fominha.mode1 import recommend as recommend_module


def _make_df(recipes):
    """DataFrame minimo com recipe_id + ingredients_canonical (para o masking)."""
    return pd.DataFrame(
        [{"recipe_id": i, "ingredients_canonical": r} for i, r in enumerate(recipes)]
    )


def _install_index(tmp_path, monkeypatch, recipes):
    """Constroi um indice de fixture em tmp_path e redireciona os paths do modulo.

    protocol.py importou META_PATH por valor, entao patcheamos tambem a copia
    ligada no namespace de protocol (para o built_at deterministico).
    """
    rows = [
        {
            "recipe_id": i,
            "title": f"Recipe {i}",
            "ingredients_raw": c,
            "ingredients_canonical": c,
            "directions": "",
            "link": f"http://example.com/{i}",
        }
        for i, c in enumerate(recipes)
    ]
    df = pd.DataFrame(rows)
    parquet_path = tmp_path / "recipes.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    artifacts = tmp_path / "artifacts"
    meta = str(artifacts / "index_meta.json")
    monkeypatch.setattr(index_module, "ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setattr(index_module, "VECTORIZER_PATH", str(artifacts / "tfidf_vectorizer.joblib"))
    monkeypatch.setattr(index_module, "MATRIX_PATH", str(artifacts / "tfidf_matrix.npz"))
    monkeypatch.setattr(index_module, "META_PATH", meta)
    monkeypatch.setattr(index_module, "DEFAULT_PARQUET_PATH", str(parquet_path))
    monkeypatch.setattr(protocol_module, "META_PATH", meta)

    index_module.build_index(str(parquet_path))
    recommend_module._reset_index_cache()


# Fixture com vocabulario compartilhado (tokens repetem >= 5x para sobreviver ao
# min_df=5). Serve para exercitar evaluate() ponta a ponta.
FIXTURE_RECIPES = [
    ["chicken", "rice", "garlic"],
    ["chicken", "rice", "onion"],
    ["chicken", "garlic", "onion"],
    ["chicken", "rice", "garlic", "onion"],
    ["chicken", "rice"],
    ["rice", "garlic", "onion"],
    ["rice", "garlic"],
    ["garlic", "onion"],
    ["beef", "potato"],
    ["beef", "onion"],
    ["beef", "rice"],
    ["beef", "garlic"],
    ["beef", "potato", "onion"],
    ["potato", "onion"],
    ["chicken", "onion"],
]


@pytest.fixture
def fixture_index(tmp_path, monkeypatch):
    _install_index(tmp_path, monkeypatch, FIXTURE_RECIPES)
    yield
    recommend_module._reset_index_cache()


@pytest.fixture
def single_eligible_index(tmp_path, monkeypatch):
    """Exatamente 1 receita elegivel (>= 3 tokens); o resto sao fillers de 2
    tokens (com um token junk unico) so para dar df >= 5 aos tokens da origem.

    Nenhum filler tem o mesmo conjunto de tokens que a query mascarada, entao
    nenhum filler alcanca jaccard alto -- isola a regra "origem relevante por
    definicao" do SPEC 8.2.
    """
    recipes = [["chicken", "rice", "garlic"]]  # id 0, unico elegivel
    for n in range(5):
        recipes.append(["chicken", f"jc{n}"])
    for n in range(5):
        recipes.append(["rice", f"jr{n}"])
    for n in range(5):
        recipes.append(["garlic", f"jg{n}"])
    _install_index(tmp_path, monkeypatch, recipes)
    yield
    recommend_module._reset_index_cache()


# --------------------------------------------------------------------------- #
# 1. Masking (Protocolo A, secao 8.1)
# --------------------------------------------------------------------------- #

def test_masking_keeps_at_least_two_masks_at_least_one_and_boundary():
    recipes = [
        ["a", "b", "c"],                      # N=3 (limite)
        ["a", "b", "c", "d"],                 # N=4
        ["a", "b", "c", "d", "e", "f", "g"],  # N=7
    ]
    df = _make_df(recipes)
    rng = np.random.default_rng(42)
    queries = build_masked_queries(df, n_eval=100, mask_frac=0.3, rng=rng)

    assert {rid for rid, _ in queries} == {0, 1, 2}
    for rid, kept in queries:
        original = recipes[rid]
        assert set(kept) <= set(original)          # mantidos sao subconjunto
        assert len(kept) >= 2                        # >= 2 mantidos
        assert len(kept) <= len(original) - 1        # >= 1 mascarado

    # N=3 -> exatamente 2 mantidos, 1 mascarado
    kept_n3 = dict(queries)[0]
    assert len(kept_n3) == 2


def test_recipes_below_three_tokens_are_excluded():
    recipes = [
        ["a", "b", "c"],        # 0 elegivel
        ["a", "b"],             # 1 excluida (2 tokens)
        ["a"],                  # 2 excluida (1 token)
        ["a", "b", "c", "d"],   # 3 elegivel
    ]
    df = _make_df(recipes)
    rng = np.random.default_rng(0)
    queries = build_masked_queries(df, n_eval=100, mask_frac=0.3, rng=rng)

    ids = {rid for rid, _ in queries}
    assert ids == {0, 3}
    assert 1 not in ids
    assert 2 not in ids


# --------------------------------------------------------------------------- #
# 2. Jaccard (Protocolo B, secao 8.2)
# --------------------------------------------------------------------------- #

def test_jaccard_identical_sets_is_one():
    assert _jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_partial_overlap_is_two_over_four():
    assert _jaccard({"a", "b"}, {"a", "b", "c", "d"}) == 0.5


def test_jaccard_threshold_relevance_boundary():
    threshold = 0.5
    # 2/4 = 0.5 -> conta como relevante (>= threshold)
    assert _jaccard({"a", "b"}, {"a", "b", "c", "d"}) >= threshold
    # 1/5 = 0.2 -> abaixo do threshold, nao conta
    assert _jaccard({"a", "b"}, {"a", "c", "d", "e"}) < threshold


def test_origin_relevant_by_definition_even_below_threshold(single_eligible_index):
    # jaccard(query, origem) = 2/3 ~ 0.667 < 0.99, e nenhum filler alcanca 0.99,
    # entao a origem so pode contar como relevante pela regra "por definicao".
    res = evaluate(
        ks_hit=(1, 5, 10), ks_prec=(5, 10),
        n_eval=50, mask_frac=0.3, jaccard_threshold=0.99, seed=42,
    )
    assert res["n_eval_queries"] == 1
    assert res["metrics"]["hit_rate@5"] == 1.0            # origem foi recuperada
    assert res["avg_relevant_per_query"]["@5"] == 1.0     # e contada, apesar de <0.99
    # precision@5 = 1 relevante / k=5 (denominador = k, secao 8.2)
    assert res["metrics"]["precision@5"] == pytest.approx(1 / 5)


def test_lower_threshold_yields_more_relevants(fixture_index):
    low = evaluate(ks_hit=(5,), ks_prec=(5,), n_eval=50, jaccard_threshold=0.0, seed=42)
    high = evaluate(ks_hit=(5,), ks_prec=(5,), n_eval=50, jaccard_threshold=0.99, seed=42)
    # threshold menor -> mais candidatos passam o corte de relevancia
    assert low["metrics"]["precision@5"] >= high["metrics"]["precision@5"]
    assert low["metrics"]["precision@5"] > 0


# --------------------------------------------------------------------------- #
# 3. Determinismo (RNF-01)
# --------------------------------------------------------------------------- #

def test_evaluate_is_deterministic_for_same_seed(fixture_index):
    r1 = evaluate(ks_hit=(1, 5, 10), ks_prec=(5, 10), n_eval=50, seed=42)
    r2 = evaluate(ks_hit=(1, 5, 10), ks_prec=(5, 10), n_eval=50, seed=42)
    assert r1 == r2
