# SPEC-MODE2.md — Fominha v2.0.0 (Modo 2: busca semântica)

Contrato de implementação do Modo 2. Complementa o SPEC.md v1.0.0 (Modo 1, gate fechado em 4/4, commits 8c132d1..b4969ba). Este documento é autocontido para a implementação do Modo 2: todo contrato necessário está aqui, incluindo os herdados do v1. O agente implementa exatamente o que está aqui, na ordem da seção 7. Nada fora deste documento é escopo.

Pré-condição verificada: gate do Modo 1 fechado (dataset tratado, recommend() rodando, avaliação com protocolo válido, README). `data/processed/recipes.parquet` e o pipeline `normalize.py` existem e NÃO devem ser alterados por este spec.

---

## 1. Objetivo

Adicionar ao Fominha o Modo 2: busca semântica por embeddings. Input: query em linguagem natural em inglês (ex.: `"something light and quick with chicken"`). Output: receitas ranqueadas por similaridade de cosseno entre o embedding da query e embeddings pré-computados das receitas, via índice FAISS local. Sem LLM de geração em nenhum ponto. Entregável-diferencial: comparação lado a lado do retrieval léxico (Modo 1) vs. semântico (Modo 2) sobre as mesmas queries, publicada no README.

---

## 2. Escopo

### 2.1 In-scope (fase 2)

- Geração offline de embeddings para as receitas do parquet existente (subset 100k), com `sentence-transformers` local.
- Índice FAISS local (`IndexFlatIP`, busca exata) persistido em `artifacts/`.
- Função pública `search(query, k)` com contrato fechado (seção 6.2).
- Avaliação com dois protocolos (seção 8): Protocolo C (quantitativo, comparável ao Modo 1) e Protocolo D (qualitativo curado, o contraste léxico vs. semântico).
- Atualização do README: seção do Modo 2 + seção de contraste com exemplos reais.
- Adição de `sentence-transformers` e `faiss-cpu` ao `requirements.txt`.

### 2.2 Non-goals (proibido implementar na fase 2)

- **LLM de geração** — em nenhuma forma: sem reranking por LLM, sem expansão de query por LLM, sem sumarização de resultado. O valor é o retrieval.
- **Collaborative filtering** (permanece non-goal, sem ratings no dataset).
- **Web app, API server, UI** (superfície continua módulo + CLI + README).
- **Índice aproximado (IVF/HNSW), GPU, quantização** — busca exata em 100k é rápida; otimização de índice é problema que este projeto não tem.
- **Fine-tuning do modelo de embedding** — modelo usado as-is.
- **Full dataset (2.2M)** — embeddar tudo é trabalho pós-gate da fase 2, mesmo racional do RNF-02 do v1.
- **Queries em português / multilíngua** — dataset e modelo operam em inglês; query PT retorna resultados degradados e isso é documentado, não tratado.
- **Alterar o Modo 1** — `normalize.py`, `mode1/`, protocolo A/B e artefatos do v1 são código congelado nesta fase. Qualquer mudança neles exige aditivo a este spec.

Se algo parecer exigir um item desta lista, PARE e reporte — não implemente.

---

## 3. Requisitos funcionais

- **RF-20** — O sistema gera embeddings para todas as receitas de `data/processed/recipes.parquet`, a partir de um texto de embedding por receita construído deterministicamente (seção 5.1), e persiste índice FAISS + metadados em `artifacts/` (contrato 6.1).
- **RF-21** — Dada uma query em linguagem natural, `search(query, k)` retorna as top-k receitas por similaridade de cosseno (contrato 6.2), embeddando a query em runtime com o MESMO modelo do índice.
- **RF-22** — O sistema calcula hit_rate@k do Modo 2 sob o Protocolo C (seção 8.1) — as MESMAS 1000 queries mascaradas da avaliação do Modo 1, mesma seed — e grava `reports/eval_mode2.json` (contrato 6.4), incluindo a tabela comparativa Modo 1 vs. Modo 2.
- **RF-23** — O sistema executa o Protocolo D (seção 8.2): roda o conjunto curado de queries em linguagem natural (arquivo versionado, contrato 6.3) nos DOIS modos e gera `reports/comparison_mode1_vs_mode2.md` com os top-5 lado a lado por query.
- **RF-24** — Todas as etapas são executáveis por scripts CLI dedicados (contrato 6.5), com seed fixa onde houver aleatoriedade.

---

## 4. Requisitos não-funcionais

- **RNF-20 — Reprodutibilidade:** embeddings são determinísticos dado o modelo fixo; o Protocolo C reusa a MESMA amostragem/mascaramento do Modo 1 (mesmo código de `eval/protocol.py`, mesma seed=42), garantindo comparabilidade query a query. `eval_mode2.json` é byte-idêntico entre dois runs (mesma resolução de `built_at` determinístico adotada no v1: lido dos metadados do índice avaliado).
- **RNF-21 — Modelo:** `sentence-transformers/all-MiniLM-L6-v2`, 384 dims, CPU. Primeira execução baixa os pesos (~80MB) para o cache local (`~/.cache`) — requer rede UMA vez; documentar no README.
- **RNF-22 — Build de embeddings:** batch_size=256, `normalize_embeddings=True`, float32. Reportar tempo total de build no log. Estimativa em CPU para 100k: dezenas de minutos — é custo único e aceitável; NÃO implementar checkpoint/resume (complexidade sem retorno no subset).
- **RNF-23 — Latência de query:** `search()` responde em < 500ms no subset 100k após índice e modelo carregados (embedding da query domina o custo; a busca FlatIP em 100k×384 é <10ms). Carregamento de modelo+índice é custo único de inicialização.
- **RNF-24 — Memória:** índice em RAM ≈ 100k × 384 × 4 bytes ≈ 150MB. Aceitável; nenhuma otimização.
- **RNF-25 — Dependências novas:** `sentence-transformers`, `faiss-cpu`, adicionadas ao `requirements.txt` nesta fase. Nada além.

---

## 5. Arquitetura e componentes

### 5.0 Arquivos novos (nenhum arquivo do Modo 1 é alterado)

```
src/fominha/mode2/
├── __init__.py            # exceções: SemanticIndexNotBuiltError
├── corpus.py              # construção do texto de embedding por receita
├── index.py               # build/load do índice FAISS
└── search.py              # função pública search()
src/fominha/eval/
└── protocol_mode2.py      # Protocolos C e D
eval_queries/
└── nl_queries.json        # queries curadas do Protocolo D (versionado)
scripts/
├── 05_build_embeddings.py
├── 06_query_semantic.py
├── 07_evaluate_semantic.py
└── 08_compare_modes.py
tests/
├── test_corpus.py
└── test_search.py
reports/
├── eval_mode2.json                    # versionado
└── comparison_mode1_vs_mode2.md       # versionado
```

### 5.1 Componente: corpus de embedding (`corpus.py`)

Uma função pública, determinística:

```python
def build_embedding_text(title: str, ingredients_canonical: list[str], directions: str) -> str
```

Formato EXATO do texto por receita:

```
"{title}. Ingredients: {ingredients_canonical unidos por ', '}. {directions[:400]}"
```

Regras:
- `directions` truncado em 400 caracteres (corte duro em caractere, sem tentar cortar em fronteira de palavra — determinismo simples acima de estética). Se `directions` vazio, o segmento final é omitido (sem espaço sobrando).
- Justificativa do 400: o MiniLM trunca em 256 wordpieces de qualquer forma; título+ingredientes carregam o grosso do sinal, e as primeiras frases das instruções agregam termos de contexto ("bake", "grill", "quick") que a query semântica pode referenciar. Cap explícito torna o truncamento visível e determinístico em vez de implícito no tokenizador.
- Este texto usa `ingredients_canonical` (não `ingredients_raw`): o corpus semântico se beneficia da mesma limpeza já validada, e mantém os dois modos operando sobre a mesma representação de ingredientes.

### 5.2 Componente: índice FAISS (`mode2/index.py`)

```python
def build_semantic_index(parquet_path: str) -> None
def load_semantic_index() -> tuple[SentenceTransformer, faiss.Index, pd.DataFrame]
```

- `build_semantic_index`: lê o parquet, constrói textos via `corpus.py`, embedda em batches (RNF-22), normaliza L2, adiciona a um `faiss.IndexFlatIP` NA ORDEM de `recipe_id` (posição i do índice == recipe_id i — invariante crítico), persiste conforme contrato 6.1.
- `load_semantic_index`: carrega modelo, índice e parquet. Artefato ausente → `SemanticIndexNotBuiltError` instruindo rodar `scripts/05_build_embeddings.py`.
- Produto interno sobre vetores L2-normalizados == cosseno (decisão herdada do v1, D-07).

### 5.3 Componente: busca (`mode2/search.py`)

Contrato na seção 6.2. Carrega modelo+índice uma vez (cache de módulo). Embedda a query com `normalize_embeddings=True` e busca `index.search(vec, k)`.

### 5.4 Componente: avaliação (`eval/protocol_mode2.py`)

- **Protocolo C:** importa e REUSA `_build_masked_queries` (ou equivalente público) de `eval/protocol.py` com os mesmos parâmetros/seed do Modo 1, garantindo as mesmas 1000 queries. Converte cada query (lista de tokens) na string `", ".join(tokens)` e a envia para `search()`. hit_rate@k idêntico em definição ao Protocolo A.
- **Protocolo D:** lê `eval_queries/nl_queries.json`, roda cada query nos dois modos (`recommend()` recebe a query como lista de tokens separada por vírgula; `search()` recebe a string crua), gera o markdown comparativo.

---

## 6. Contratos de interface

### 6.1 Artefatos do índice semântico

```
artifacts/embeddings.faiss        # faiss.IndexFlatIP, 384 dims, vetores L2-normalizados,
                                  # posição i alinhada a recipe_id i
artifacts/embeddings_meta.json    # {"model": "sentence-transformers/all-MiniLM-L6-v2",
                                  #  "dims": 384, "n_recipes": int, "aligned_to": "recipe_id",
                                  #  "corpus_format": "title. Ingredients: <canonical>. directions[:400]",
                                  #  "built_at": iso8601, "build_seconds": float}
```

### 6.2 Função pública do Modo 2

```python
def search(
    query: str,        # linguagem natural em inglês; strip aplicado
    k: int = 10,       # 1 <= k <= 100; fora do range -> ValueError
) -> list[SemanticResult]

# Retorno ordenado por score desc. len(retorno) == k sempre que o índice tiver >= k receitas.
@dataclass(frozen=True)
class SemanticResult:
    recipe_id: int
    title: str
    score: float       # cosseno; teoricamente [-1, 1], na prática tipicamente [0, 1]
    link: str
```

Diferenças deliberadas vs. o `Recommendation` do Modo 1: NÃO há `matched_ingredients`/`missing_ingredients` (não existe noção de match de token no espaço de embedding — inventar uma seria enganoso) e NÃO há descarte de score 0 (cosseno de embeddings raramente é exatamente 0 e valores baixos ainda são um ranking válido; o Modo 2 sempre devolve k resultados).

Erros: `EmptyQueryError` (reusar a exceção existente do Modo 1) se a query, após strip, for vazia; `SemanticIndexNotBuiltError` se artefatos ausentes; `ValueError` para k inválido.

### 6.3 Queries curadas do Protocolo D (`eval_queries/nl_queries.json`)

```json
{
  "version": "1.0",
  "queries": [
    {"id": "q01", "text": "something light and quick with chicken",
     "rationale": "semantic: 'light'/'quick' não são ingredientes"},
    {"id": "q02", "text": "cozy winter dessert",
     "rationale": "semantic: nenhum token é ingrediente"}
  ]
}
```

15 queries, cada uma com `rationale` de uma linha explicando por que ela discrimina léxico de semântico. O CONTEÚDO das 15 queries é decisão do Pedro (Decisões a confirmar, item 2) — o arquivo é versionado e congela o benchmark qualitativo.

### 6.4 Relatório de avaliação (`reports/eval_mode2.json`)

```json
{
  "protocol": "masked-retrieval-semantic (C)",
  "n_recipes_index": 99634,
  "n_eval_queries": 1000,
  "mask_frac": 0.3,
  "seed": 42,
  "query_format": "tokens joined by ', '",
  "metrics": {
    "mode2_hit_rate@1": 0.0,
    "mode2_hit_rate@5": 0.0,
    "mode2_hit_rate@10": 0.0
  },
  "mode1_reference": {
    "hit_rate@1": 0.812,
    "hit_rate@5": 0.925,
    "hit_rate@10": 0.953
  },
  "index_built_at": "iso8601"
}
```

Nota: o campo de timestamp chama `index_built_at` (corrige a semântica ambígua do `built_at` do v1; o v1 não é alterado — a correção de nome vale deste artefato em diante).

### 6.5 CLI dos scripts

```
python scripts/05_build_embeddings.py                       # sem args; lê o parquet default
python scripts/06_query_semantic.py  --query "something light with chicken" --k 10
python scripts/07_evaluate_semantic.py  --k 1 5 10 --n-eval 1000 --mask-frac 0.3 --seed 42
python scripts/08_compare_modes.py                          # lê eval_queries/nl_queries.json,
                                                            # gera reports/comparison_mode1_vs_mode2.md
```

### 6.6 Saída do comparativo (`reports/comparison_mode1_vs_mode2.md`)

Por query do Protocolo D: a query + rationale, seguida de duas colunas (tabela markdown): top-5 do Modo 1 (title + score) e top-5 do Modo 2 (title + score). Sem interpretação automática — a leitura dos casos onde cada modo acerta/erra é feita por humano e escrita à mão no README (seção 9, item 4).

---

## 7. Fluxos críticos (ordem de implementação)

### 7.1 Fluxo 1 — Build de embeddings

1. Carregar parquet. Ausente → erro claro apontando `scripts/01_ingest.py`.
2. Construir textos via `corpus.py` na ordem de `recipe_id`.
3. Embeddar em batches de 256, `normalize_embeddings=True`, float32.
4. Criar `IndexFlatIP(384)`, `index.add(embeddings)`.
5. Persistir `embeddings.faiss` + `embeddings_meta.json` (contrato 6.1) com tempo de build.
6. Logar: n_receitas, dims, tempo total.

### 7.2 Fluxo 2 — Query semântica

1. `search(query, k)`: validar k (1..100), strip na query; vazia → `EmptyQueryError`.
2. Embeddar a query (mesmo modelo, normalizada).
3. `index.search(vec, k)` → ids + scores.
4. Mapear ids → metadados do parquet → `SemanticResult`.

### 7.3 Fluxo 3 — Avaliação (Protocolo C)

1. Regenerar as 1000 queries mascaradas com o MESMO código e seed do Modo 1.
2. Para cada: string `", ".join(tokens)` → `search(query, k=10)`.
3. hit_rate@{1,5,10}: origem no top-k.
4. Gravar `eval_mode2.json` com o bloco `mode1_reference` copiado de `reports/eval_mode1.json` (ler do arquivo, não hardcodar).

### 7.4 Fluxo 4 — Comparativo (Protocolo D)

1. Ler `nl_queries.json`.
2. Por query: Modo 1 (`recommend(query.split(", "))`... a query D é linguagem natural, então para o Modo 1 ela passa pelo `normalize_ingredients` normalmente — tokens não-ingrediente viram OOV e é ISSO que o comparativo expõe) e Modo 2 (`search(query)`).
3. Gerar o markdown do contrato 6.6.

---

## 8. Protocolo de avaliação

### 8.1 Protocolo C — Masked retrieval semântico (métrica: hit_rate@k)

- **Definição idêntica ao Protocolo A do v1**, mudando apenas o sistema sob teste: mesmas 1000 receitas amostradas, mesmo mascaramento (mask_frac=0.3, seed=42, ≥2 mantidos, ≥1 mascarado, receitas <3 tokens excluídas), relevante = exclusivamente a receita de origem.
- **Query string:** tokens mantidos unidos por `", "`. O Modo 1 recebeu esses tokens como lista; o Modo 2 recebe a string. É o input mais próximo de idêntico que os dois modos aceitam — qualquer template adicional ("a recipe with...") seria um confounder introduzido só no Modo 2 e quebraria a comparação.
- **Métrica:** hit_rate@{1,5,10}, mesma fórmula do v1.
- **Leitura esperada (registrar no README, não esconder):** é PLAUSÍVEL que o Modo 2 perca do Modo 1 neste protocolo — a query é uma lista literal de ingredientes, o cenário ideal do match léxico. Isso não é falha do Modo 2: é a demonstração quantitativa de que os modos têm regimes diferentes. O Protocolo D mostra o regime inverso.

### 8.2 Protocolo D — Benchmark qualitativo curado (entregável, não métrica)

- 15 queries em linguagem natural, versionadas em `eval_queries/nl_queries.json`, escolhidas para discriminar os regimes: queries com termos não-ingrediente ("light", "cozy", "quick", "comfort food") onde o léxico degrada (tokens viram OOV no Modo 1) e o semântico deve capturar intenção.
- Saída: `comparison_mode1_vs_mode2.md` (contrato 6.6), lado a lado, sem julgamento automático.
- A análise (onde cada modo acerta/erra, com 3-4 exemplos escolhidos) é escrita À MÃO no README — é o entregável-diferencial do projeto e não se automatiza.
- **Por que não há métrica quantitativa aqui:** não existe rótulo de relevância para queries de intenção sem julgamento humano; inventar um (ex.: reusar Jaccard de ingredientes) mediria aderência léxica e penalizaria estruturalmente o modo semântico — exatamente o erro que o protocolo do v1 existe para impedir.
- **Explicitamente descartado:** aplicar o Protocolo B (overlap Jaccard) ao Modo 2. O rótulo Jaccard é construído sobre sobreposição de tokens — é léxico por construção e avaliaria o Modo 2 pela régua do Modo 1.

---

## 9. Critérios de aceite (gate da fase 2)

A fase 2 está aceita quando TODOS forem verdadeiros:

1. **Embeddings gerados e indexados** — `embeddings.faiss` + `embeddings_meta.json` existem, cumprem o contrato 6.1, alinhamento posição==recipe_id verificado por teste.
2. **Busca semântica rodando** — `search()` cumpre o contrato 6.2; `scripts/06_query_semantic.py` retorna resultados coerentes para 3 queries manuais documentadas.
3. **Protocolo C calculado** — `eval_mode2.json` válido (contrato 6.4), reprodutível byte-idêntico entre dois runs, com o bloco comparativo do Modo 1.
4. **Contraste léxico vs. semântico publicado** — `comparison_mode1_vs_mode2.md` gerado a partir das 15 queries versionadas, e README atualizado com: seção do Modo 2 (o que é, como rodar), tabela do Protocolo C com leitura honesta (incluindo o resultado esperado da seção 8.1), e a análise manual do contraste com 3-4 exemplos.
5. **Testes passam** — `test_corpus.py` (formato exato do texto, truncamento em 400, directions vazio) e `test_search.py` (ordenação desc, EmptyQueryError, k inválido, alinhamento id↔posição via fixture pequena) verdes junto com os 39 existentes.
6. **Modo 1 intocado** — `git diff` da fase não toca `normalize.py`, `mode1/`, `eval/protocol.py` (exceto, se necessário, tornar público o helper de masking — mudança de visibilidade apenas, sem alterar lógica, coberta pelos testes existentes).
7. Nenhum non-goal da seção 2.2 implementado.

---

## 10. Edge cases e tratamento de erro

| ID | Caso | Comportamento obrigatório |
|----|------|---------------------------|
| E-20 | Query vazia ou só whitespace | `EmptyQueryError` (resolve o E-09 reservado no v1). |
| E-21 | `k` fora de [1,100] ou não-inteiro | `ValueError`. |
| E-22 | Artefatos semânticos ausentes | `SemanticIndexNotBuiltError` instruindo `scripts/05_build_embeddings.py`. |
| E-23 | Parquet ausente no build | Erro claro apontando `scripts/01_ingest.py`. Não roda ingestão automaticamente. |
| E-24 | Primeira execução sem cache do modelo e sem rede | Deixar a exceção de download do sentence-transformers propagar com nota no README (pré-requisito de rede único). Não fazer retry/fallback. |
| E-25 | Query mais longa que o limite do modelo | Truncamento silencioso pelo tokenizador (256 wordpieces) — comportamento aceito e documentado, sem erro. |
| E-26 | Query em não-inglês | Fora de escopo (non-goal): sem detecção de idioma, resultados degradados aceitos, documentado no README. |
| E-27 | Índice e parquet dessincronizados (n diferente) | `load_semantic_index` compara `n_recipes` do meta com o parquet; divergência → erro instruindo rebuild. |

---

## 11. Decisões técnicas (ADR-lite)

| # | Decisão | Porquê (1 linha) |
|---|---------|------------------|
| D-20 | Spec do Modo 2 em documento separado (SPEC-MODE2.md); SPEC.md v1 congelado | O v1 é contrato cumprido e histórico; reescrevê-lo suja o diff e arrisca reabrir decisões fechadas. |
| D-21 | `all-MiniLM-L6-v2`, 384 dims, CPU | Herdada do v1 (5.3): roda local sem custo de API, qualidade suficiente para retrieval. |
| D-22 | `IndexFlatIP` exato, sem ANN | 100k×384 é pequeno; ANN adiciona tuning e aproximação sem problema real a resolver. |
| D-23 | Corpus = título + ingredientes canônicos + directions[:400] | Título/ingredientes carregam o sinal; cap explícito torna o truncamento determinístico e visível. |
| D-24 | Protocolo C reusa as MESMAS queries mascaradas do Modo 1 | Comparabilidade query a query entre os modos; protocolo novo introduziria confounder de amostragem. |
| D-25 | Query do Protocolo C sem template ("tokens, unidos, por vírgula") | Template adicionado só no Modo 2 seria confounder; input o mais idêntico possível nos dois modos. |
| D-26 | Sem métrica quantitativa no Protocolo D; análise manual no README | Relevância de queries de intenção exige julgamento humano; rótulo automático (Jaccard) é léxico por construção e viesaria contra o semântico. |
| D-27 | Protocolo B (Jaccard) explicitamente NÃO aplicado ao Modo 2 | Avaliaria o modo semântico pela régua léxica — erro de medição, não rigor. |
| D-28 | `SemanticResult` sem matched/missing e sem descarte de score 0 | Não existe match de token em espaço de embedding; inventar seria enganoso (lição do matched/bigrama do v1). |
| D-29 | `index_built_at` no lugar de `built_at` nos artefatos novos | Corrige a semântica ambígua identificada no v1 sem tocar em artefato já versionado. |
| D-30 | Sem checkpoint/resume no build de embeddings | Build de 100k em CPU é custo único de dezenas de minutos; resume é complexidade sem retorno no subset. |
| D-31 | 15 queries curadas versionadas em JSON com rationale | Congela o benchmark qualitativo (diffável, citável no README) em vez de queries ad-hoc irreproduzíveis. |

---

## Decisões a confirmar com o Pedro

Única seção não-determinística. O corpo assume as recomendações; ajustes são pontuais.

1. **Cap de 400 caracteres em `directions` (D-23).** Recomendação: manter. Trade-off: cap maior inclui mais contexto de preparo, mas o MiniLM trunca em 256 wordpieces de qualquer jeito — acima de ~400 chars o excedente tende a ser cortado pelo tokenizador e o cap explícito vira ficção.
2. **Conteúdo das 15 queries do Protocolo D.** Esta é a decisão mais importante da fase e é SUA: as queries são o benchmark do entregável-diferencial. Recomendação: eu proponho uma lista candidata de 15 (mistura de: intenção pura sem ingrediente, ingrediente + qualificador de estilo, prato-conceito tipo "comfort food"), você corta/edita/aprova, e o JSON congela. Trade-off: queries fáceis demais inflam o Modo 2; difíceis demais viram cherry-picking reverso — a lista precisa ter casos onde o Modo 1 VENCE também, senão o contraste é propaganda e não análise.
3. **Expor o helper de masking de `eval/protocol.py` como função pública** (hoje `_build_masked_queries` é privado; o Protocolo C precisa importá-lo). Recomendação: renomear para público (`build_masked_queries`) num commit próprio de refactor mínimo antes da fase 2, com os 9 testes existentes cobrindo. Trade-off: alternativa é duplicar a lógica em `protocol_mode2.py` — pior (duas fontes de verdade para a MESMA amostra quebra a comparabilidade se divergirem).