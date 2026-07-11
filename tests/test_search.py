import os
import sys

import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fominha.mode1.recommend import EmptyQueryError
from fominha.mode2 import index as index_module
from fominha.mode2 import search as search_module
from fominha.mode2.corpus import build_embedding_text
from fominha.mode2.search import search

# Textos distintos o suficiente para o modelo semantico nao confundir top-1.
FIXTURE_RECIPES = [
    {
        "title": "Grilled Chicken Salad",
        "ingredients_canonical": ["chicken", "lettuce", "tomato"],
        "directions": "Grill chicken and toss with lettuce and tomato.",
        "link": "http://example.com/0",
    },
    {
        "title": "Chocolate Lava Cake",
        "ingredients_canonical": ["chocolate", "flour", "butter", "egg"],
        "directions": "Bake chocolate cake until molten center.",
        "link": "http://example.com/1",
    },
    {
        "title": "Spicy Vegetable Curry",
        "ingredients_canonical": ["curry powder", "vegetable", "coconut milk"],
        "directions": "Simmer vegetables in spicy curry sauce.",
        "link": "http://example.com/2",
    },
    {
        "title": "Classic Beef Stew",
        "ingredients_canonical": ["beef", "carrot", "potato"],
        "directions": "Slow cook beef with carrots and potatoes.",
        "link": "http://example.com/3",
    },
    {
        "title": "Fresh Fruit Smoothie",
        "ingredients_canonical": ["banana", "strawberry", "yogurt"],
        "directions": "Blend fruit with yogurt until smooth.",
        "link": "http://example.com/4",
    },
    {
        "title": "Garlic Butter Shrimp",
        "ingredients_canonical": ["shrimp", "garlic", "butter"],
        "directions": "Saute shrimp in garlic butter sauce.",
        "link": "http://example.com/5",
    },
    {
        "title": "Vegetarian Lentil Soup",
        "ingredients_canonical": ["lentil", "carrot", "onion"],
        "directions": "Simmer lentils with carrot and onion until tender.",
        "link": "http://example.com/6",
    },
]


@pytest.fixture(scope="module")
def fixture_index(tmp_path_factory):
    """Constroi um indice semantico pequeno uma unica vez para todo o modulo
    (evita recarregar o SentenceTransformer a cada teste)."""
    mp = MonkeyPatch()
    tmp_path = tmp_path_factory.mktemp("mode2_search")

    rows = [
        {
            "recipe_id": i,
            "title": r["title"],
            "ingredients_raw": r["ingredients_canonical"],
            "ingredients_canonical": r["ingredients_canonical"],
            "directions": r["directions"],
            "link": r["link"],
        }
        for i, r in enumerate(FIXTURE_RECIPES)
    ]
    df = pd.DataFrame(rows)
    parquet_path = tmp_path / "recipes.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    artifacts = tmp_path / "artifacts"
    mp.setattr(index_module, "ARTIFACTS_DIR", str(artifacts))
    mp.setattr(index_module, "FAISS_INDEX_PATH", str(artifacts / "embeddings.faiss"))
    mp.setattr(index_module, "META_PATH", str(artifacts / "embeddings_meta.json"))
    mp.setattr(index_module, "DEFAULT_PARQUET_PATH", str(parquet_path))

    index_module.build_semantic_index(str(parquet_path))
    search_module._reset_index_cache()

    yield

    search_module._reset_index_cache()
    mp.undo()


def test_results_sorted_by_score_desc(fixture_index):
    results = search("a warm hearty meal with meat and vegetables", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_raises_empty_query_error(fixture_index):
    with pytest.raises(EmptyQueryError):
        search("", k=5)


def test_whitespace_only_query_raises_empty_query_error(fixture_index):
    with pytest.raises(EmptyQueryError):
        search("   ", k=5)


def test_k_zero_raises_value_error(fixture_index):
    with pytest.raises(ValueError):
        search("chicken salad", k=0)


def test_k_over_limit_raises_value_error(fixture_index):
    with pytest.raises(ValueError):
        search("chicken salad", k=101)


def test_k_non_integer_raises_value_error(fixture_index):
    with pytest.raises(ValueError):
        search("chicken salad", k=1.5)


def test_result_length_equals_k_when_enough_recipes(fixture_index):
    results = search("something with chicken", k=5)
    assert len(results) == 5


def test_alignment_exact_text_returns_same_recipe_top1(fixture_index):
    origin = FIXTURE_RECIPES[3]  # "Classic Beef Stew"
    exact_text = build_embedding_text(
        origin["title"], origin["ingredients_canonical"], origin["directions"]
    )
    results = search(exact_text, k=1)
    assert len(results) == 1
    assert results[0].recipe_id == 3
    assert results[0].score == pytest.approx(1.0, abs=1e-4)
