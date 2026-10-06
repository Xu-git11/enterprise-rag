"""Core retrieval utilities for workspace isolation and hybrid search."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

import requests
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi

CONTENT_PAYLOAD_KEY = "page_content"
METADATA_PAYLOAD_KEY = "metadata"


def tokenize(text: str) -> list[str]:
    """Small Chinese/Latin tokenizer suitable for the demo corpus."""
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())


def workspace_filter(workspace_id: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key=f"{METADATA_PAYLOAD_KEY}.workspace_id",
                match=MatchValue(value=workspace_id),
            )
        ]
    )


def load_workspace_documents(
    client: QdrantClient,
    collection_name: str,
    workspace_id: str,
    limit: int = 2000,
) -> list[Document]:
    documents: list[Document] = []
    offset = None
    while len(documents) < limit:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=workspace_filter(workspace_id),
            limit=min(256, limit - len(documents)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload: dict[str, Any] = point.payload or {}
            metadata = payload.get(METADATA_PAYLOAD_KEY) or {}
            content = payload.get(CONTENT_PAYLOAD_KEY) or ""
            if content:
                documents.append(Document(page_content=content, metadata=metadata))
        if offset is None:
            break
    return documents


def bm25_search(
    question: str,
    documents: list[Document],
    top_k: int,
) -> list[Document]:
    if not documents:
        return []
    corpus = [tokenize(document.page_content) for document in documents]
    query_tokens = tokenize(question)
    if not query_tokens:
        return []
    scores = BM25Okapi(corpus).get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(documents)),
        key=lambda index: (-float(scores[index]), index),
    )
    return [documents[index] for index in ranked_indices[:top_k]]


def document_key(document: Document) -> str:
    metadata = document.metadata or {}
    raw = "|".join(
        [
            str(metadata.get("workspace_id", "")),
            str(metadata.get("file_name", metadata.get("source", ""))),
            str(metadata.get("page", "")),
            document.page_content,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def rrf_fuse(
    ranked_lists: Iterable[list[Document]],
    top_k: int,
    rrf_k: int = 60,
) -> list[Document]:
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranked_list in ranked_lists:
        for rank, document in enumerate(ranked_list, start=1):
            key = document_key(document)
            documents[key] = document
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    ranked_keys = sorted(scores, key=lambda key: (-scores[key], key))
    return [documents[key] for key in ranked_keys[:top_k]]


def rerank_candidates(
    question: str,
    documents: list[Document],
    *,
    model: str,
    api_key: str,
    base_url: str,
    top_n: int,
    timeout: int = 60,
) -> list[Document]:
    if len(documents) <= 1:
        return documents[:top_n]
    response = requests.post(
        f"{base_url.rstrip('/')}/rerank",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "query": question,
            "documents": [document.page_content for document in documents],
            "return_documents": False,
            "top_n": min(top_n, len(documents)),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    reranked: list[Document] = []
    for item in results:
        index = int(item["index"])
        if 0 <= index < len(documents):
            document = documents[index]
            document.metadata["rerank_score"] = item.get("relevance_score")
            reranked.append(document)
    return reranked or documents[:top_n]


def hybrid_retrieve(
    *,
    client: QdrantClient,
    collection_name: str,
    question: str,
    workspace_id: str,
    vector_candidates: list[Document],
    vector_top_k: int,
    bm25_top_k: int,
    final_top_k: int,
    enable_hybrid: bool,
    enable_reranker: bool,
    reranker_model: str,
    siliconflow_api_key: str,
    siliconflow_base_url: str,
) -> list[Document]:
    candidates = vector_candidates[:vector_top_k]
    if enable_hybrid:
        workspace_documents = load_workspace_documents(
            client, collection_name, workspace_id
        )
        keyword_candidates = bm25_search(question, workspace_documents, bm25_top_k)
        candidates = rrf_fuse([candidates, keyword_candidates], final_top_k)

    if enable_reranker and candidates:
        try:
            return rerank_candidates(
                question,
                candidates,
                model=reranker_model,
                api_key=siliconflow_api_key,
                base_url=siliconflow_base_url,
                top_n=final_top_k,
            )
        except Exception as exc:
            print(f"Reranker fallback: {type(exc).__name__}: {exc}")
    return candidates[:final_top_k]
