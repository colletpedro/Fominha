"""Pipeline de normalização de ingredientes (SPEC.md secao 5.1.2, decisao D-04).

Usado tanto sobre a coluna NER do RecipeNLG quanto sobre o input do usuario
em runtime -- o mesmo pipeline precisa servir os dois para o espaco de
features do TF-IDF coincidir entre index-time e query-time.
"""

import re

FRACTION_CHARS = "¼½¾⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅐⅑"

# Sempre incluir forma singular E plural de cada stopword: a remoção de
# stopwords precede a singularização (seção 5.1.2 do SPEC), então o plural
# vaza se não for listado explicitamente.
INGREDIENT_STOPWORDS = {
    "cup", "cups", "tablespoon", "tablespoons", "tbsp", "teaspoon", "teaspoons", "tsp",
    "ounce", "ounces", "oz", "pound", "pounds", "lb", "gram", "grams", "g", "kg", "ml",
    "package", "packages", "pkg", "can", "cans", "jar", "jars", "box", "boxes",
    "slice", "slices", "sliced", "chopped", "diced", "minced",
    "fresh", "frozen", "softened", "melted", "optional",
    "large", "small", "medium",
    "of", "and",
    "handful", "stalk", "bunch", "sprig", "pinch", "dash", "head", "drop",
    "handfuls", "stalks", "bunches", "sprigs", "pinches", "dashes", "heads", "drops",
}

SINGULAR_WHITELIST = {"molasses", "couscous", "hummus"}

_PAREN_RE = re.compile(r"\([^)]*\)")
_PUNCT_RE = re.compile(rf"[^a-z0-9\s/{re.escape(FRACTION_CHARS)}]")
_NUMBER_RE = re.compile(r"^[\d/]+$")


def _is_number_or_fraction(token: str) -> bool:
    if not token:
        return True
    if all(ch in FRACTION_CHARS for ch in token):
        return True
    return bool(_NUMBER_RE.match(token))


def _singularize(word: str) -> str:
    if word in SINGULAR_WHITELIST:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _normalize_one(raw: str) -> str:
    text = raw.lower().strip()
    text = _PAREN_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)

    words = []
    for word in text.split():
        if _is_number_or_fraction(word):
            continue
        if word in INGREDIENT_STOPWORDS:
            continue
        word = _singularize(word)
        if len(word) < 3:
            continue
        words.append(word)

    return " ".join(words)


def normalize_ingredients(raw_tokens: list[str]) -> list[str]:
    """Normaliza uma lista de strings de ingrediente para tokens canonicos.

    Aplica-se tanto a entradas da coluna NER do dataset quanto a input livre
    do usuario (decisao D-04) -- mesmo pipeline, mesmas etapas.
    """
    seen = set()
    result = []
    for raw in raw_tokens:
        canonical = _normalize_one(raw)
        if not canonical:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        result.append(canonical)
    return result
