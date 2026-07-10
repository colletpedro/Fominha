import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fominha.normalize import normalize_ingredients


def test_parenthetical_with_quantity_and_unit():
    assert normalize_ingredients(
        ["1 (8 ounce) package cream cheese, softened"]
    ) == ["cream cheese"]


def test_unicode_fraction_half():
    assert normalize_ingredients(["½ cup sugar"]) == ["sugar"]


def test_unicode_fraction_three_quarters():
    assert normalize_ingredients(["¾ cup broken nuts (pecans)"]) == ["broken nut"]


def test_mixed_number_and_unicode_fraction():
    assert normalize_ingredients(["1 ½ cups flour"]) == ["flour"]


def test_ascii_fraction_slash():
    assert normalize_ingredients(["1/2 cup evaporated milk"]) == ["evaporated milk"]


def test_plural_simple_tomatoes():
    assert normalize_ingredients(["3 tomatoes"]) == ["tomato"]


def test_plural_whitelist_molasses():
    assert normalize_ingredients(["2 tablespoons molasses"]) == ["molasses"]


def test_plural_whitelist_couscous():
    assert normalize_ingredients(["1 cup couscous"]) == ["couscous"]


def test_plural_whitelist_hummus():
    assert normalize_ingredients(["2 tbsp hummus"]) == ["hummus"]


def test_compound_unit_abbreviation():
    assert normalize_ingredients(["2 Tbsp. butter"]) == ["butter"]


def test_large_small_medium_removed():
    assert normalize_ingredients(["1 large egg"]) == ["egg"]


def test_fresh_and_frozen_removed():
    assert normalize_ingredients(["2 cups frozen peas"]) == ["pea"]
    assert normalize_ingredients(["1 cup fresh basil"]) == ["basil"]


def test_optional_removed():
    assert normalize_ingredients(["1 tsp salt, optional"]) == ["salt"]


def test_can_jar_package_removed():
    assert normalize_ingredients(["1 can black beans"]) == ["black bean"]
    assert normalize_ingredients(["1 jar salsa"]) == ["salsa"]


def test_short_token_discarded():
    assert normalize_ingredients(["1 ea onion"]) == ["onion"]


def test_empty_result_when_only_quantity_and_unit():
    assert normalize_ingredients(["1/2 cup", "2 tbsp"]) == []


def test_deduplication_preserves_order():
    assert normalize_ingredients(["1 cup sugar", "2 cups sugar", "1 cup flour"]) == [
        "sugar",
        "flour",
    ]


def test_ner_style_input_already_clean():
    assert normalize_ingredients(["cream cheese", "sugar", "chicken breast"]) == [
        "cream cheese",
        "sugar",
        "chicken breast",
    ]


def test_sliced_chopped_diced_minced_removed():
    assert normalize_ingredients(["2 cloves garlic, minced"]) == ["clove garlic"]
    assert normalize_ingredients(["1 onion, chopped"]) == ["onion"]
    assert normalize_ingredients(["1 cucumber, sliced"]) == ["cucumber"]


def test_measure_word_handful_removed():
    assert normalize_ingredients(["handful grape tomatoes"]) == ["grape tomato"]


def test_measure_word_stalk_removed():
    assert normalize_ingredients(["1 stalk celery"]) == ["celery"]


def test_measure_word_pinch_removed():
    assert normalize_ingredients(["pinch of salt"]) == ["salt"]


def test_measure_word_stalks_plural_removed():
    assert normalize_ingredients(["2 stalks celery"]) == ["celery"]
