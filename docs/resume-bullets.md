# Resume and Interview Notes

## 中文简历条目

- 基于 Apache-2.0 开源 Corrective RAG 二次开发企业知识库问答系统，使用
  Python 3.12、Streamlit、FastAPI、LangGraph、Qdrant、DeepSeek 和
  SiliconFlow BGE-M3。
- 设计 PBKDF2 登录与 workspace 级数据隔离，在 Qdrant payload 中写入
  `workspace_id/user_id`，所有检索强制带工作区 filter，防止跨租户数据泄露。
- 实现向量检索、BM25 与 Reciprocal Rank Fusion，并支持可选 BGE Reranker；
  构建 8 题评测集，比较四组检索方案，量测准确率、引用率和 815-1081 ms 延迟。
- 提供 FastAPI `/v1/chat` 与 SSE `/v1/chat/stream` 接口、API Key 鉴权、
  Docker Compose 一键启动，以及可选 Langfuse Trace/Span 可观测性。

## English Resume Bullets

- Built a workspace-isolated enterprise RAG service on Python 3.12,
  Streamlit, FastAPI, LangGraph, Qdrant, DeepSeek, and SiliconFlow BGE-M3.
- Implemented PBKDF2 authentication and mandatory Qdrant `workspace_id`
  filtering to prevent cross-tenant retrieval.
- Added optional vector + BM25 retrieval with Reciprocal Rank Fusion and an
  optional BGE reranker; benchmarked four configurations for accuracy,
  citation rate, and latency.
- Delivered `/v1/chat` and SSE streaming APIs, API-key authorization,
  Docker Compose startup, and optional Langfuse tracing.

## Interview Questions

- Why isolate by payload filter instead of one collection per workspace?
- Why did Hybrid and reranking not improve the small benchmark?
- How do you prevent a retrieval bug from returning another tenant's document?
- How would you scale BM25 beyond the current in-memory workspace index?
- How would you add conversation memory without contaminating citations?
- What changes are required before production use?
