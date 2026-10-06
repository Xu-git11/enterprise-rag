# RAG Evaluation

`questions.json` contains the initial benchmark for the sample company policy.
The evaluation script compares vector-only retrieval, optional hybrid retrieval,
and optional reranking.

Generated files:

- `results_vector_v3.json`
- `results_vector_reranker_v3.json`
- `results_hybrid_v3.json`
- `results_hybrid_reranker_v3.json`

Metrics:

- Keyword accuracy: expected numeric fact appears in the answer
- Citation rate: answer includes a page/source citation
- Average latency: retrieval plus generation latency

Run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py
```

The benchmark uses a temporary Qdrant collection and removes it after the run.
