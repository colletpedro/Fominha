

https://github.com/user-attachments/assets/2e353777-f0f4-4365-a165-2d9028a7acc9

# Fominha

Recipe recommender with two retrieval modes: content-based (Mode 1, lexical/TF-IDF) and semantic search (Mode 2, embeddings). Give it ingredients or a natural-language craving, get back ranked recipes.

<video src="docs/demo.mp4" controls width="100%"></video>

*(If the video doesn't render inline, it's at [`docs/demo.mp4`](docs/demo.mp4).)*

## 1. Demo UI

A local single-page app that runs the same query against both modes at once and shows the results side by side — built to make the lexical-vs-semantic contrast visible, not to hide it.

**Run it** (requires the indices already built — see [Running from the terminal](#5-running-from-the-terminal) below):

```bash
pip install -e .
python -m uvicorn fominha.api.app:app
# http://localhost:8000
```

First startup loads both indices (~9s); every query after that responds in ~10ms.

**Design note:** the two result cards are deliberately asymmetric — Mode 2's card has no `matched`/`missing` chips, because there is no notion of ingredient match in embedding space (decision D-28). A permanent banner reminds you that scores aren't comparable across modes. The UI follows the same honesty rules as the rest of this project: it doesn't manufacture a false sense of parity between the two retrieval methods.

## 2. What it does — Mode 1 (lexical)

Input: a free-text list of ingredients (e.g. `"chicken, rice, garlic"`). Output: recipes ranked by cosine similarity between your ingredients and each recipe's ingredient list, both represented as TF-IDF vectors over canonical ingredient tokens.

There is no user history, no ratings, no personalization — this is a single stateless function: `recommend(ingredients, k)`. Each result includes the similarity score plus which of your ingredients matched the recipe and which of the recipe's ingredients you're missing.

## 3. What it does — Mode 2 (semantic)

Input: a natural-language query in English (e.g. `"something light and quick with chicken"`). Output: recipes ranked by cosine similarity between the query's embedding and pre-computed recipe embeddings, via a local FAISS index. Same "no LLM generation anywhere" rule as Mode 1 — this is retrieval, not generation.

Single stateless function: `search(query, k)`. Unlike Mode 1's `Recommendation`, `SemanticResult` carries no `matched_ingredients`/`missing_ingredients` — there is no notion of token match in embedding space, and inventing one would be misleading (decision D-28 in `SPEC-MODE2.md`).

## 4. Why these choices

Summarized from `SPEC.md` section 11 and `SPEC-MODE2.md` section 11 (ADR-lite):

**Mode 1:**
- **Content-based, not collaborative filtering** — RecipeNLG has no user ratings; collaborative filtering needs interaction data the dataset doesn't provide.
- **TF-IDF + cosine similarity** — an interpretable, training-free baseline that also serves as an honest lexical contrast to the semantic mode.
- **RecipeNLG's `NER` column as the ingredient source, plus a custom normalization pipeline** — parsing raw ingredient strings is the biggest quality risk; the dataset's own pre-extracted entities plus deterministic normalization keep that risk bounded without outsourcing it to a heavier NLP stack.
- **Same normalization pipeline for dataset and user query** — index-time and query-time features must live in the same space or cosine similarity breaks silently.
- **Bigrams in the TF-IDF vocabulary (`ngram_range=(1,2)`)** — compound ingredients ("cream cheese", "olive oil") become a single feature instead of two diluted unigrams.
- **Suffix-rule singularization, no spaCy/NLTK** — a full lemmatizer is a heavy dependency for marginal gain over a food-ingredient vocabulary; see [Known limitations](#9-known-limitations) for the trade-off this makes.

**Mode 2:**
- **`sentence-transformers/all-MiniLM-L6-v2`, 384 dims, CPU-capable** — runs locally at no API cost, sufficient retrieval quality.
- **`faiss.IndexFlatIP`, exact search, no ANN** — 100k×384 vectors is small; approximate indexing would add tuning without a real problem to solve.
- **Embedding corpus = title + canonical ingredients + directions[:400]** — title/ingredients carry the bulk of the signal; the explicit cap makes truncation deterministic and visible instead of implicit in the tokenizer.
- **Protocol C reuses Mode 1's exact masked queries** (`build_masked_queries`, decision D-24) — query-by-query comparability between modes; a new sampling procedure would introduce a confound.
- **No quantitative metric for Protocol D; manual analysis instead** (D-26) — there's no ground-truth relevance label for intent queries without human judgment, and reusing Mode 1's Jaccard-overlap label would measure lexical adherence and structurally penalize the semantic mode.

## 5. Running from the terminal

<details>
<summary>Running from the terminal (dataset setup + CLI)</summary>

Requires Python >= 3.11.

**1. Get the dataset.** RecipeNLG is not bundled in this repo (license-gated distribution). Download it manually and place it at `data/raw/full_dataset.csv`:
- Official site: https://recipenlg.cs.put.poznan.pl/ (accept the license, download the full dataset)
- Kaggle: dataset "RecipeNLG" via the Kaggle API (requires `kaggle` auth configured)

**2. Set up the environment.**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Build the indices.**
```bash
python scripts/01_ingest.py --raw data/raw/full_dataset.csv --n-recipes 100000 --seed 42
python scripts/02_build_index.py
python scripts/05_build_embeddings.py
```

`01_ingest.py` reads the raw CSV, samples `--n-recipes` rows with `--seed`, normalizes ingredients, and writes `data/processed/recipes.parquet`. `02_build_index.py` takes no arguments — it builds the TF-IDF index from that parquet and writes `artifacts/`. `05_build_embeddings.py` takes no arguments — it embeds every recipe in the parquet, builds the FAISS index, and writes `artifacts/embeddings.faiss` + `artifacts/embeddings_meta.json`. On this machine it ran in `build_seconds=206.82` using Apple Silicon GPU (MPS) — expect substantially longer on CPU-only hardware (see [Known limitations](#9-known-limitations)). First run of `05_build_embeddings.py` downloads the model weights (~80MB) to `~/.cache` — requires network access once (RNF-21 in `SPEC-MODE2.md`).

**4. Query CLIs.**
```bash
python scripts/04_query.py --ingredients "chicken, rice, garlic" --k 10
python scripts/06_query_semantic.py --query "something light and quick with chicken" --k 10
```

`04_query.py` is the Mode 1 manual query CLI. Example output (`--ingredients "chicken, rice, garlic" --k 5`, run against the 100k-recipe index):

```
1. Chicken And Rice (recipe_id=67995, score=0.7044)
   matched: chicken, rice
   missing:
   link: https://www.cookbooks.com/Recipe-Details.aspx?id=681629

2. On the Go Munchies (recipe_id=3874, score=0.5427)
   matched: rice, garlic
   missing: peanut
   link: https://www.food.com/recipe/on-the-go-munchies-330534

3. Baked Chicken And Rice-A-Roni (recipe_id=33946, score=0.4918)
   matched:
   missing: chicken breast, chicken rice
   link: https://www.cookbooks.com/Recipe-Details.aspx?id=36887

4. Yonggyebaeksuk (Korean Spring Chicken Soup) (recipe_id=77209, score=0.4670)
   matched: rice, garlic
   missing: young roasting chicken, onion, chinese jujube, chestnut, ginger, salt, green onion
   link: https://www.food.com/recipe/yonggyebaeksuk-korean-spring-chicken-soup-488704

5. Chicken Oriental (recipe_id=11391, score=0.4621)
   matched: chicken, rice, garlic
   missing: vegetable, sweet sue chicken
   link: https://www.cookbooks.com/Recipe-Details.aspx?id=581696
```

Result 3 is a useful example of why `matched_ingredients` can be empty while the recipe still ranks: see [Known limitations](#9-known-limitations).

`06_query_semantic.py` is the Mode 2 manual query CLI. Example output (`--query "something light and quick with chicken" --k 5`):

```
1. Pulled Chicken (recipe_id=16514, score=0.6107)
   link: https://www.myrecipes.com/recipe/pulled-chicken

2. Easy Herbed Chicken (recipe_id=23880, score=0.5949)
   link: https://www.cookbooks.com/Recipe-Details.aspx?id=897173

3. Chicken Mole (recipe_id=7299, score=0.5945)
   link: https://www.tasteofhome.com/recipes/chicken-mole/

4. Penthouse Chicken (recipe_id=28342, score=0.5910)
   link: https://www.cookbooks.com/Recipe-Details.aspx?id=1082639

5. Butterflied Chicken With Cracked Spices (recipe_id=87456, score=0.5884)
   link: https://cooking.nytimes.com/recipes/1013156
```

Note there's no `matched`/`missing` here — see [What it does — Mode 2](#3-what-it-does--mode-2-semantic) for why.

**5. Evaluation CLIs.**
```bash
python scripts/03_evaluate.py --k 1 5 10 --n-eval 1000 --mask-frac 0.3 --seed 42
python scripts/07_evaluate_semantic.py --k 1 5 10 --n-eval 1000 --mask-frac 0.3 --seed 42
python scripts/08_compare_modes.py
```

`03_evaluate.py` runs the two Mode 1 evaluation protocols (section 6) and writes `reports/eval_mode1.json`. `07_evaluate_semantic.py` runs Protocol C (section 7) and writes `reports/eval_mode2.json`. `08_compare_modes.py` runs the 15 curated Protocol D queries against both modes and writes `reports/comparison_mode1_vs_mode2.md`.

</details>

## 6. Evaluation — Mode 1

RecipeNLG has no user ratings, so "relevant" is constructed, not given — via two protocols (masked-recipe retrieval + Jaccard overlap), both seeded (`seed=42`). Full protocol definitions in `SPEC.md` section 8.

**Results** (`reports/eval_mode1.json`, 100k-recipe subset, `n_eval_queries=1000`, `seed=42`):

| Metric | Value |
|---|---|
| hit_rate@1 | 0.812 |
| hit_rate@5 | 0.925 |
| hit_rate@10 | 0.953 |
| precision@5 | 0.337 |
| precision@10 | 0.2305 |

Average relevant recipes per query (Protocol B, diagnostic for a degenerate threshold): **1.685** at k=5, **2.305** at k=10 — nontrivial, so the threshold isn't collapsing to "only the origin recipe counts."

**Honest reading:** `hit_rate@10 = 0.953` is high largely because the origin recipe is present in the index and the query is a subset of its own ingredients — that's the point of the protocol (recovery from partial information). `precision@k` is lower and partially circular with the ranker's own signal — read it as a regression check, not a quality proof.

## 7. Evaluation — Mode 2

**Protocol C** feeds Mode 2 the exact same 1000 masked queries used in Mode 1's evaluation (comparable to Protocol A, no new sampling — decision D-24). **Protocol D** runs 15 curated natural-language queries through both modes with no automatic metric (decision D-26) — the manual reading of that comparison is section 9, below. Full protocol definitions in `SPEC-MODE2.md` section 8.

**Results** (`reports/eval_mode2.json`, same 100k-recipe subset, same 1000 masked queries, `seed=42`):

| k | Mode 1 `hit_rate@k` | Mode 2 `hit_rate@k` |
|---|---|---|
| 1 | 0.812 | 0.062 |
| 5 | 0.925 | 0.115 |
| 10 | 0.953 | 0.16 |

**Honest reading:** it is plausible — and confirmed here — for Mode 2 to lose badly on this protocol. The query is a literal, comma-joined ingredient list: the ideal-case scenario for exact lexical match, and structurally unfavorable ground for a semantic model whose corpus is full prose. This is not a failure of Mode 2; it's a quantitative demonstration that the two modes have different regimes. Protocol D (next section) shows the inverse regime.

## 8. Lexical vs. semantic: the contrast

### Lexical vs. semantic retrieval: what the comparison actually shows

*(15 curated natural-language queries, both modes, top-5 each — full tables in
`reports/comparison_mode1_vs_mode2.md`. Four findings, chosen because each
teaches something different.)*

**1. On intent queries, lexical search fails silently — semantic search captures
the intent.** For `"healthy breakfast to meal prep"`, Mode 1 returns "Party
Salad Topping" (0.42) and "Tea Latte" — random matches on ghost tokens,
delivered with the confidence of a real answer. Mode 2 returns "Super Healthy
Grain Breakfast" (0.69) and "Healthy Breakfast" (0.58). The dangerous part is
not that Mode 1 fails, but *how*: it never signals that the query is outside its
vocabulary. Words like "healthy" or "cozy" survive normalization as tokens that
will never match an ingredient, so the ranker returns low-scoring noise instead
of admitting blindness. A recommender that answers plausibly to questions it
doesn't understand is worse than one that errs loudly.

**2. On literal ingredient lists, lexical search wins — cleanly.** For
`"flour sugar butter eggs vanilla"`, Mode 1 finds "Old Fashioned Sugar Cookies"
at a perfect cosine of 1.0; Mode 2 tops out at 0.69. This is the honest half of
the benchmark: when the query *is* a list of ingredients, exact token matching
is structurally better, and no embedding model changes that. Each mode has a
regime.

**3. Negation defeats both modes.** `"easy pasta with garlic and no cream"` asks
for pasta *without* cream. Mode 2's top result: "Creamy One Pot Pasta" (0.71) —
the exact opposite. Mode 1's: "Garlic Potatoes And Ham" — not even pasta. The
embedding sees "cream" and moves *closer*; the lexical ranker sees "cream" as a
token and matches it. Neither representation encodes negation. This is a known
limitation of both bag-of-words and sentence-embedding retrieval, and this
benchmark reproduces it on demand.

**4. Scores from the two modes are not comparable.** For
`"chicken rice garlic onion"`, Mode 2's top hit ("Onion Rice", 0.73) has a
*higher* score than Mode 1's ("Chicken And Rice", 0.64) — yet Mode 1's result is
plainly better (Mode 2 lost the chicken). Cosine in TF-IDF space and cosine in
embedding space are different quantities on different geometries. Never rank
across modes by score.

**Why Mode 2 loses heavily on the quantitative protocol (hit_rate@10: 0.16 vs
0.953).** Protocol C feeds both modes the same masked-ingredient queries — a
bare token list, which is Mode 1's native input. Diagnosis showed the dominant
factor is *query–corpus format asymmetry*: the semantic index stores recipes as
full prose ("Title. Ingredients: … directions"), and a naked ingredient list
embeds far from its own recipe's prose. Evidence: re-querying with each recipe's
full corpus text returns the origin recipe at rank 1 in 20/20 sampled cases —
the index is sound; the query format is hostile. Near-duplicate recipes ("Banana
Bread" vs. "Annie's Banana Bread") are a real but secondary factor. The pair of
results is the point: Mode 2 loses on Mode 1's terrain (Protocol C) and wins on
its own (Protocol D, group A). Neither number alone describes the system.

## 9. Known limitations

This is a portfolio-honesty section, not an apology.

- **Normalization trade-offs are deliberate, not oversights.** `"grnd"` / `"ground"` is preserved as-is rather than stripped, because removing it would be lossy (`ground beef` != `beef`). `"clove"` is intentionally kept in the vocabulary rather than treated as a stopword — it's ambiguous (a unit of garlic *and* the spice) and collapsing it either way would be wrong for the other sense. Suffix-based singularization produces artifacts like `"bay leaves" -> "bay leave"` — a known cost of avoiding a full lemmatizer (decision D-03 in `SPEC.md`).
- **`matched_ingredients` can be empty on a high-scoring result (Mode 1).** The score is computed in the full uni+bigram TF-IDF space, but `matched_ingredients` is only the intersection of canonical unigram tokens. A recipe can rank highly purely on a bigram match (e.g. `"chicken rice"` as a single feature) while showing zero matched unigrams — see result #3 in the Mode 1 query example in [Running from the terminal](#5-running-from-the-terminal) above. The score reflects the real similarity; `matched`/`missing` are explanatory, not the full story.
- **RecipeNLG's `NER` column occasionally includes non-ingredients** (`"wooden skewer"`, `"toothpick"`) that survive normalization as tokens. This is long-tail noise, not filtered in v1.
- **Evaluation runs on a 100k-recipe subset** of the ~2.2M full RecipeNLG dataset. Scaling to the full dataset is explicitly post-gate work (`RNF-02` in `SPEC.md`).
- **Mode 1 and Mode 2 scores are not comparable to each other.** They live in different geometries — TF-IDF cosine over a sparse lexical space vs. cosine over a dense 384-dim sentence embedding — so a 0.65 in one mode says nothing about a 0.65 in the other. Only within-mode ranking is meaningful.
- **`build_seconds=206.82` for the full embeddings build was measured on Apple Silicon GPU (MPS)**, not CPU. On CPU-only hardware the build is substantially slower (`SPEC-MODE2.md` RNF-22 estimates "dezenas de minutos" / tens of minutes for 100k on CPU). Any decision to scale embedding generation to the full 2.2M dataset should use the CPU cost as the reference, not this machine's GPU-accelerated number.
- **The Mode 2 artifact field is `index_built_at`, not `built_at`.** This corrects the ambiguous naming used in Mode 1's `index_meta.json` (`built_at` doesn't say built *what*), a fix that applies to new artifacts going forward only — Mode 1's already-versioned artifact schema is untouched (decision D-29 in `SPEC-MODE2.md`).
- **Negation in queries is understood by neither mode.** `"easy pasta with garlic and no cream"` (Protocol D, q07) returns cream-based results as top hits in both Mode 1 (lexical match on "cream" as a token) and Mode 2 (semantic embedding doesn't encode "no" as an operator over "cream" — it just sees "cream" as a nearby concept). See the contrast analysis above for concrete examples.
- Recipe links come from RecipeNLG (published 2020) and are not validated. Some resolve correctly, some 404, and some domains are gone entirely — the dataset is a 2020 snapshot of the web. Validating ~100k URLs is out of scope; the links are kept as the dataset's original references.

## 10. Scope & non-goals

These are decisions, not omissions (`SPEC.md` section 2.2, `SPEC-MODE2.md` section 2.2):

- **No collaborative filtering** — the dataset has no user ratings to drive it.
- **No generative LLM**, anywhere in the system, in either mode — no LLM reranking, no query expansion, no result summarization. The value is retrieval.
- **No web app, API server, or UI** — this ships as a Python module + CLI scripts, nothing else.
- **No per-user personalization, history, or favorites.**
- **No external infrastructure** — no managed vector DB, no cloud, no Docker, no GPU requirement. Everything runs locally.
- **No approximate/ANN indexing, no quantization, no model fine-tuning** — exact search on 100k is fast enough; optimizing it is a problem this project doesn't have.
- **No datasets other than RecipeNLG, no translation/multilingual support** — a non-English query degrades silently in both modes; this is documented, not handled.
- **No full 2.2M-recipe dataset** — both modes run on the 100k subset; scaling is explicitly post-gate work.

## 11. Roadmap

Mode 2 (semantic search via sentence embeddings + FAISS) is **delivered**: `search()`, Protocol C (quantitative, comparable to Mode 1), and Protocol D (qualitative curated contrast) are all implemented and reported above. What's left, explicitly out of scope for this phase: embedding the full ~2.2M-recipe RecipeNLG dataset (currently both modes run on the 100k subset) — the same "subset first, close the loop before paying for scale" rationale as Mode 1's `RNF-02`.

## 12. Tech stack

Python >= 3.11, `pandas`, `pyarrow`, `scikit-learn`, `scipy`, `pytest` (Mode 1), plus `sentence-transformers` and `faiss-cpu` (Mode 2), plus `fastapi` and `uvicorn` (Demo UI).

---

## Resumo em português

**O que é.** Um recomendador de receitas com dois modos: o Modo 1 (content-based) recebe ingredientes disponíveis e retorna receitas ranqueadas por similaridade de cosseno TF-IDF; o Modo 2 (busca semântica) recebe uma query em linguagem natural e retorna receitas por similaridade de embeddings (`sentence-transformers/all-MiniLM-L6-v2`) via índice FAISS local.

**Demo UI.** O vídeo no topo deste README mostra a demo: uma página local que roda a mesma query nos dois modos e exibe os resultados lado a lado. Para rodar: `pip install -e .` seguido de `python -m uvicorn fominha.api.app:app`.

**Por que essas escolhas.** Sem ratings de usuário no RecipeNLG, então collaborative filtering está fora dos dois modos; TF-IDF + cosseno é um baseline léxico interpretável e sem treino; o Modo 2 usa um modelo de embeddings local (sem custo de API, sem GPU obrigatória) e busca exata via FAISS `IndexFlatIP`, suficiente no subset de 100k.

**Non-goals declarados.** Sem collaborative filtering, sem LLM de geração (em nenhum dos modos), sem web app/API/UI, sem personalização por usuário, sem infraestrutura externa, sem indexação aproximada, sem fine-tuning, sem outros datasets além do RecipeNLG, sem full dataset (2.2M) nesta fase.

**Quickstart condensado (via terminal).**
```bash
# 1. Baixe o RecipeNLG manualmente (licenciado) para data/raw/full_dataset.csv
#    https://recipenlg.cs.put.poznan.pl/  ou via Kaggle API

# 2. Ambiente
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Modo 1
python scripts/01_ingest.py --raw data/raw/full_dataset.csv --n-recipes 100000 --seed 42
python scripts/02_build_index.py
python scripts/03_evaluate.py --k 1 5 10 --n-eval 1000 --mask-frac 0.3 --seed 42
python scripts/04_query.py --ingredients "chicken, rice, garlic" --k 10

# 4. Modo 2 (primeira execução baixa o modelo, ~80MB, requer rede uma vez)
python scripts/05_build_embeddings.py
python scripts/06_query_semantic.py --query "something light and quick with chicken" --k 10
python scripts/07_evaluate_semantic.py --k 1 5 10 --n-eval 1000 --mask-frac 0.3 --seed 42
python scripts/08_compare_modes.py
```

**Avaliação.** Modo 1: hit_rate@10 = 0.953 (Protocolo A, métrica principal); precision@5 = 0.337 (Protocolo B, parcialmente circular, leitura de sanidade). Modo 2: hit_rate@10 = 0.16 no Protocolo C (mesmas 1000 queries mascaradas do Modo 1) — número baixo esperado, pois a query é uma lista literal de ingredientes, terreno ideal do léxico; o Protocolo D (15 queries de linguagem natural, sem métrica automática por decisão D-26) mostra o regime inverso, onde o semântico captura intenção que o léxico não alcança.
