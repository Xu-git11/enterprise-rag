# Evaluation Comparison

The same 8-question benchmark was run against a 4-chunk sample document.

| Configuration | Keyword accuracy | Citation rate | Average latency |
|---|---:|---:|---:|
| Vector retrieval only | 100% | 100% | 815 ms |
| Vector retrieval + BGE reranker | 100% | 100% | 940 ms |
| Hybrid vector + BM25 RRF | 100% | 100% | 889 ms |
| Hybrid vector + BM25 RRF + BGE reranker | 100% | 100% | 1081 ms |

## Conclusion

All configurations answer the benchmark correctly, but the small four-chunk corpus does not provide enough retrieval difficulty to show an accuracy gain from Hybrid or Reranker. On this corpus, vector-only retrieval is fastest; Hybrid adds about 73 ms and Reranker adds more. Both features are implemented but disabled by default.

Reproduce:

```powershell
$env:ENABLE_HYBRID_SEARCH='false'
$env:ENABLE_RERANKER='false'
$env:EVAL_OUTPUT='results_vector_v3.json'
.\.venv\Scripts\python.exe scripts\evaluate_rag.py

$env:ENABLE_HYBRID_SEARCH='false'
$env:ENABLE_RERANKER='true'
$env:EVAL_OUTPUT='results_vector_reranker_v3.json'
.\.venv\Scripts\python.exe scripts\evaluate_rag.py

$env:ENABLE_HYBRID_SEARCH='true'
$env:ENABLE_RERANKER='false'
$env:EVAL_OUTPUT='results_hybrid_v3.json'
.\.venv\Scripts\python.exe scripts\evaluate_rag.py

$env:ENABLE_HYBRID_SEARCH='true'
$env:ENABLE_RERANKER='true'
$env:EVAL_OUTPUT='results_hybrid_reranker_v3.json'
.\.venv\Scripts\python.exe scripts\evaluate_rag.py
```
