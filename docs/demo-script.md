# Three-Minute Demo Script

## 0:00-0:30 Problem

"Companies have internal documents, but generic chat models cannot reliably cite
the exact file and page. I built a workspace-isolated RAG service with grounded
answers and source traceability."

## 0:30-1:10 Ingestion and Isolation

1. Log in as `demo / demo123`.
2. Upload `samples/company_policy.txt`.
3. Show the chunk count and workspace name.
4. Mention that all Qdrant points include `workspace_id` and retrieval always
   applies a `metadata.workspace_id` filter.

## 1:10-1:50 Grounded Answer

Ask:

```text
正式员工入职满三年后，每年有多少天带薪年假？
```

Show:

- Correct answer: 15 days
- Source panel below the answer
- LangGraph node outputs

## 1:50-2:20 Engineering Tradeoffs

Explain the benchmark:

```text
Vector only:     815 ms
Hybrid RRF:      889 ms
Vector + rerank:   940 ms
Hybrid + rerank:  1081 ms
```

All configurations reached 100% on the small benchmark, so Hybrid and reranker
remain optional instead of adding latency by default.

## 2:20-2:50 API and Deployment

Show:

```text
http://localhost:8000/docs
```

Mention:

- FastAPI `/v1/chat` and `/v1/chat/stream`
- API Key to workspace mapping
- Docker Compose startup
- Optional Langfuse tracing

## 2:50-3:00 Result

"This project demonstrates RAG engineering, evaluation, multi-tenant isolation,
streaming APIs, observability hooks, and deployment readiness."
