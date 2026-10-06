# Evaluation Comparison

The same 8-question benchmark was run against a 4-chunk sample document.

| Configuration | Keyword accuracy | Citation rate | Average latency |
|---|---:|---:|---:|
| Vector retrieval only | 100% | 100% | 1002 ms |
| Vector retrieval + BGE reranker | 100% | 100% | 1041 ms |

## Conclusion

For this small document and question set, the reranker did not improve keyword or citation accuracy and added about 39 ms per query. The reranker remains available but is disabled by default.

Reproduce:

```powershell
$env:ENABLE_RERANKER='false'
$env:EVAL_OUTPUT='results.json'
.\.venv\Scripts\python.exe scripts\evaluate_rag.py

$env:ENABLE_RERANKER='true'
$env:EVAL_OUTPUT='results_reranker.json'
.\.venv\Scripts\python.exe scripts\evaluate_rag.py
```
