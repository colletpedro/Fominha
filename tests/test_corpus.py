import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fominha.mode2.corpus import build_embedding_text


def test_exact_format_with_all_fields_filled():
    result = build_embedding_text(
        title="Chicken And Rice",
        ingredients_canonical=["chicken", "rice", "garlic"],
        directions="Cook the chicken and rice together with garlic.",
    )
    assert result == (
        "Chicken And Rice. Ingredients: chicken, rice, garlic. "
        "Cook the chicken and rice together with garlic."
    )


def test_directions_truncated_to_exactly_400_chars():
    directions = "x" * 401
    result = build_embedding_text("Title", ["a", "b"], directions)
    expected_directions = "x" * 400
    assert result == f"Title. Ingredients: a, b. {expected_directions}"
    # confirma o corte exato: nenhum caractere a mais sobrevive
    assert result.endswith("x" * 400)
    assert not result.endswith("x" * 401)


def test_directions_empty_omits_final_segment_no_trailing_space():
    result = build_embedding_text("Title", ["a", "b"], "")
    assert result == "Title. Ingredients: a, b."
    assert not result.endswith(" ")


def test_directions_exactly_400_chars_untouched():
    directions = "y" * 400
    result = build_embedding_text("Title", ["a"], directions)
    assert result == f"Title. Ingredients: a. {directions}"
    assert result.count("y") == 400
