# SPEC-FRONTEND.md — Fominha v3.0.0 (demo UI local)

Contrato da fase 3: UI de demo local sobre os dois modos existentes. Fases 1 e 2 (gates fechados) são código congelado. Entregável final: app rodando local + GIF/vídeo no README. Referência visual/comportamental: protótipo aprovado do Claude Design (arquivo `design/prototype_reference.html`, commitado como referência — NÃO é código de produção; usa runtime proprietário e deve ser PORTADO para vanilla).

---

## 1. Objetivo

Servir uma página única que envia a mesma query aos dois modos e exibe os resultados lado a lado, com a assimetria dos cards e o aviso de scores não-comparáveis do protótipo. Backend: camada FastAPI fina sobre `recommend()` e `search()` existentes, sem lógica nova de retrieval.

---

## 2. Escopo

### In-scope
- `src/fominha/api/app.py`: FastAPI com 2 endpoints + servir o estático.
- `frontend/index.html`: porte vanilla (HTML/CSS/JS num arquivo) do protótipo — mesmo layout, estados, cores e textos.
- Dependências novas: `fastapi`, `uvicorn` (requirements).
- README: seção "Demo UI" com instruções de execução + GIF.

### Non-goals (proibido nesta fase)
- Hospedagem/deploy, Docker, auth, banco, favoritos, histórico, paginação, filtros, i18n, modo escuro.
- Alterar qualquer código das fases 1/2 (`normalize.py`, `mode1/`, `mode2/`, `eval/`).
- Framework frontend (React/Vue/build step). Vanilla num arquivo.
- Streaming/SSE; respostas são JSON simples.

---

## 3. Contratos de API

### POST /api/recommend  (Modo 1)
Request: `{"query": str, "k": int}` (k default 5, 1<=k<=20 nesta API)
Tokenização da query (DECISÃO): se a query contém vírgula → `split(",")` com strip por item (preserva ingredientes compostos: "cream cheese, sugar"); senão → `split()` por espaço (frases de intenção degradam palavra a palavra, como no Protocolo D).
Response 200: lista (possivelmente vazia) de:
```json
{"recipe_id": 67995, "title": "Chicken And Rice", "score": 0.7044,
 "matched_ingredients": ["chicken","rice"], "missing_ingredients": [],
 "link": "https://..."}
```
EmptyQueryError → 200 com `[]` e header nenhum especial (a UI mostra o estado vazio honesto). Query vazia/whitespace → 422. k inválido → 422.

### POST /api/search  (Modo 2)
Request: `{"query": str, "k": int}` — a string crua vai inteira para `search()`.
Response 200: lista de `{"recipe_id", "title", "score", "link"}`.
Query vazia → 422. k inválido → 422.

### GET /
Serve `frontend/index.html`. Mesma origem para API e estático → sem CORS.

---

## 4. Fluxo e comportamento da UI (porte fiel do protótipo)

- Campo único + botão "Search both" + Enter; 4 exemplos clicáveis com os rótulos do protótipo.
- Duas colunas com loading independente (skeleton), estados idle/loading/empty/done.
- Card Modo 1: título, score 4 casas, chips matched (verde)/missing (cinza), link. Card Modo 2: título, score, link — SEM chips (D-28).
- Aviso permanente entre colunas: "≠ scores are not comparable across modes" com o tooltip explicativo do protótipo.
- Estado vazio do Modo 1 com o texto honesto do protótipo. Erro de query vazia inline.
- Link do rodapé: https://github.com/colletpedro/Fominha (corrige o placeholder do protótipo).

---

## 5. Critérios de aceite (gate da fase 3)

1. `python -m uvicorn fominha.api.app:app` sobe (pacote instalado via `pip install -e .`, src-layout, sem muleta `PYTHONPATH`); `GET /` serve a página; os 2 endpoints respondem conforme seção 3 (validado com 3 queries reais: uma de cada regime).
2. UI portada fiel ao protótipo (side-by-side, assimetria, aviso, estados) em vanilla, sem runtime proprietário, funcionando em browser local.
3. `tests/test_api.py`: contratos dos 2 endpoints (schema da resposta, 422 para query vazia e k inválido, [] para query sem ingredientes no Modo 1). Suíte total verde.
4. README com seção Demo UI: como rodar (2 comandos) + GIF gravado da busca lado a lado.
5. Nenhum non-goal implementado; fases 1/2 intocadas no diff.

---

## 6. Edge cases

| ID | Caso | Comportamento |
|----|------|---------------|
| E-30 | Query vazia/whitespace | 422 nos dois endpoints; UI valida antes e mostra erro inline. |
| E-31 | Query sem ingrediente reconhecível (Modo 1) | 200 + `[]`; UI mostra estado vazio honesto. |
| E-32 | k fora de [1,20] | 422. |
| E-33 | Artefatos não construídos | 503 com mensagem instruindo scripts 02/05; UI mostra o texto do erro. |
| E-34 | Primeira chamada lenta (carga de modelo+índices) | Aceito; carregar índices no startup do app (lifespan) para pagar o custo uma vez. |

---

## 7. Decisões (ADR-lite)

| # | Decisão | Porquê |
|---|---------|--------|
| D-40 | FastAPI servindo estático na mesma origem | Elimina CORS e segundo servidor; demo local com 1 comando. |
| D-41 | Porte vanilla, protótipo como referência (não como código) | O export do Claude Design usa runtime proprietário; portar mantém o design sem herdar dependência invisível. |
| D-42 | Tokenização híbrida no /api/recommend (vírgula OU espaço) | Vírgula preserva ingredientes compostos; espaço reproduz a degradação honesta do Protocolo D em frases. |
| D-43 | Índices carregados no lifespan do app | Latência de query ~10ms depois do startup; custo de carga pago uma vez. |
| D-44 | Erros de domínio como estados de UI, não como 500 | Query sem ingrediente é RESULTADO (o contraste), não falha. |