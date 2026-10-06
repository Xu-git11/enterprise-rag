# Enterprise RAG

A portfolio-ready enterprise knowledge-base assistant built from a Corrective RAG
workflow. It uses DeepSeek for grounded generation, SiliconFlow `BAAI/bge-m3`
for embeddings, Qdrant for vector search, and Streamlit for the UI.

![Architecture](docs/architecture.svg)

## Highlights

- Document ingestion for PDF, TXT, Markdown, and URLs
- Chinese-aware recursive chunking
- DeepSeek-powered relevance grading and query rewriting
- SiliconFlow 1024-dimensional BGE-M3 embeddings
- Local Qdrant vector storage and retrieval
- Page/source labels returned separately from the generated answer
- Optional BGE reranker with automatic fallback
- Repeatable 8-question RAG benchmark
- Secrets stored locally in `.env` and excluded from Git
- Apache-2.0 derivative with upstream attribution

## Evaluation

The benchmark uses `samples/company_policy.txt`, which is split into 4 chunks.
It measures whether the expected fact appears in the answer, whether a citation
is present, and end-to-end latency.

| Configuration | Keyword accuracy | Citation rate | Average latency |
|---|---:|---:|---:|
| Vector retrieval only | 100% | 100% | 1002 ms |
| Vector retrieval + BGE reranker | 100% | 100% | 1041 ms |

For this small corpus, the reranker did not improve answer quality and added
about 39 ms per query. It is implemented but disabled by default.

Detailed results:

- `eval/comparison.md`
- `eval/results.json`
- `eval/results_reranker.json`

## Architecture

```text
User
  -> Streamlit document upload / question
  -> LangGraph Corrective RAG workflow
  -> Qdrant Top-K retrieval
  -> optional SiliconFlow BGE reranker
  -> DeepSeek grounded answer
  -> answer plus document/page source labels
```

Embedding:

```text
Text chunks -> SiliconFlow BAAI/bge-m3 -> 1024-dim vectors -> Qdrant
```

## Quick Start

### 1. Start Qdrant

```powershell
docker pull docker.m.daocloud.io/qdrant/qdrant:latest

docker run -d `
  --name qdrant-local `
  -p 6333:6333 `
  -p 6334:6334 `
  -v qdrant_storage:/qdrant/storage `
  docker.m.daocloud.io/qdrant/qdrant:latest
```

### 2. Configure secrets

Copy `.env.example` to `.env` and set:

```env
DEEPSEEK_API_KEY=...
SILICONFLOW_API_KEY=...
```

### 3. Create the Python environment

```powershell
uv venv --python 3.12.14 .venv
.\.venv\Scripts\Activate.ps1
uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 4. Check external services

```powershell
.\.venv\Scripts\python.exe scripts\check_apis.py
.\.venv\Scripts\python.exe scripts\check_reranker.py
```

Expected:

```text
DeepSeek: OK
SiliconFlow embedding: OK (dimension=1024)
Qdrant: OK
SiliconFlow reranker: OK
```

### 5. Run the UI

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://127.0.0.1:8501`, upload
`samples/company_policy.txt`, and ask:

```text
正式员工入职满三年后，每年有多少天带薪年假？
```

Expected answer: `15 天`, with source labels displayed below the answer.

## Reranker

Enable the optional reranker in `.env`:

```env
ENABLE_RERANKER=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RETRIEVAL_TOP_K=10
RERANK_TOP_N=5
```

When enabled, the application retrieves a wider candidate set and reranks it
before generation. If the reranker API fails, the application falls back to the
original retrieval order.

## Benchmark

```powershell
$env:ENABLE_RERANKER='false'
$env:EVAL_OUTPUT='results.json'
.\.venv\Scripts\python.exe scripts\evaluate_rag.py

$env:ENABLE_RERANKER='true'
$env:EVAL_OUTPUT='results_reranker.json'
.\.venv\Scripts\python.exe scripts\evaluate_rag.py
```

## Project Structure

```text
enterprise-rag/
├── app.py
├── requirements.txt
├── .env.example
├── samples/
│   └── company_policy.txt
├── scripts/
│   ├── check_apis.py
│   ├── check_reranker.py
│   ├── evaluate_rag.py
│   └── smoke_rag.py
├── eval/
│   ├── questions.json
│   ├── comparison.md
│   ├── results.json
│   └── results_reranker.json
├── docs/
│   └── architecture.svg
├── UPSTREAM.json
└── LICENSE
```

## Design Decisions

- DeepSeek handles Chat, relevance grading, and query rewriting through its
  OpenAI-compatible API.
- SiliconFlow provides a single API surface for BGE-M3 embeddings and the
  optional BGE reranker.
- Qdrant runs locally so the first version does not depend on a managed vector
  database.
- Tavily web search is disabled for the local-first version. The workflow can
  still transform queries when retrieval grading fails.
- Reranking is optional because the initial benchmark shows no accuracy gain on
  the small sample corpus.

## Roadmap

- User accounts and tenant-level document isolation
- Hybrid vector + BM25 retrieval
- Larger evaluation set with multi-hop questions
- Langfuse tracing and cost tracking
- FastAPI streaming endpoint
- Docker Compose deployment
- Public hosted demo

## Upstream and License

This project is a derivative of:

```text
Shubhamsaboo/awesome-llm-apps
rag_tutorials/corrective_rag
License: Apache-2.0
```

The exact upstream commit and path are recorded in `UPSTREAM.json`. The original
license is retained in `LICENSE`.
