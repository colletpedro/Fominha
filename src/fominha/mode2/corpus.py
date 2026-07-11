"""Construcao do texto de embedding por receita (SPEC-MODE2.md secao 5.1)."""

DIRECTIONS_MAX_CHARS = 400


def build_embedding_text(title: str, ingredients_canonical: list[str], directions: str) -> str:
    """Formato exato: "{title}. Ingredients: {tokens unidos por ', '}. {directions[:400]}".

    directions truncado em corte duro de caractere; se vazio, o segmento final
    e omitido sem deixar espaco sobrando no fim.
    """
    ingredients_str = ", ".join(ingredients_canonical)
    base = f"{title}. Ingredients: {ingredients_str}."

    truncated_directions = directions[:DIRECTIONS_MAX_CHARS] if directions else ""
    if not truncated_directions:
        return base

    return f"{base} {truncated_directions}"
