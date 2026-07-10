import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fominha.mode1 import index as index_module
from fominha.mode1 import recommend as recommend_module
from fominha.mode1.recommend import EmptyQueryError, recommend

# Ingredientes de cada receita da fixture. Tokens de interesse (chicken, rice,
# garlic, onion, beef) aparecem em >= 5 receitas cada, para sobreviver ao
# min_df=5 do TfidfVectorizer (contrato 6.2 / secao 5.2).
RECIPES = [
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


@pytest.fixture(autouse=True)
def fixture_index(tmp_path, monkeypatch):
    rows = [
        {
            "recipe_id": i,
            "title": f"Recipe {i}",
            "ingredients_raw": canonical,
            "ingredients_canonical": canonical,
            "directions": "",
            "link": f"http://example.com/{i}",
        }
        for i, canonical in enumerate(RECIPES)
    ]
    df = pd.DataFrame(rows)
    parquet_path = tmp_path / "recipes.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(index_module, "ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setattr(index_module, "VECTORIZER_PATH", str(artifacts_dir / "tfidf_vectorizer.joblib"))
    monkeypatch.setattr(index_module, "MATRIX_PATH", str(artifacts_dir / "tfidf_matrix.npz"))
    monkeypatch.setattr(index_module, "META_PATH", str(artifacts_dir / "index_meta.json"))
    monkeypatch.setattr(index_module, "DEFAULT_PARQUET_PATH", str(parquet_path))

    index_module.build_index(str(parquet_path))
    recommend_module._reset_index_cache()

    yield

    recommend_module._reset_index_cache()


def test_results_sorted_by_score_desc():
    results = recommend(["chicken", "rice", "garlic"], k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_raises_empty_query_error():
    with pytest.raises(EmptyQueryError):
        recommend(["1/2 cup", "2 tbsp"], k=5)


def test_k_zero_raises_value_error():
    with pytest.raises(ValueError):
        recommend(["chicken"], k=0)


def test_k_over_limit_raises_value_error():
    with pytest.raises(ValueError):
        recommend(["chicken"], k=101)


def test_k_non_integer_raises_value_error():
    with pytest.raises(ValueError):
        recommend(["chicken"], k=1.5)


def test_all_oov_query_returns_empty_list():
    results = recommend(["zzzznotinvocabulary"], k=10)
    assert results == []


def test_result_length_never_exceeds_k():
    results = recommend(["chicken", "rice", "garlic", "onion"], k=3)
    assert len(results) <= 3
