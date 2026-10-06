"""Verify workspace-level isolation without external API calls."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from rag_core import bm25_search, hybrid_retrieve, load_workspace_documents, workspace_filter

QDRANT_URL = "http://localhost:6333"


def point(text: str, workspace_id: str, file_name: str) -> PointStruct:
    return PointStruct(
        id=str(uuid.uuid4()),
        vector=[1.0, 0.0],
        payload={
            "page_content": text,
            "metadata": {
                "workspace_id": workspace_id,
                "file_name": file_name,
                "source": file_name,
            },
        },
    )


def main() -> int:
    collection = f"isolation_test_{uuid.uuid4().hex[:8]}"
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    try:
        client.upsert(
            collection_name=collection,
            points=[
                point("demo workspace secret: alpha", "demo_workspace", "demo.txt"),
                point("analyst workspace secret: beta", "analyst_workspace", "analyst.txt"),
            ],
        )
        demo_docs = load_workspace_documents(client, collection, "demo_workspace")
        assert len(demo_docs) == 1, "workspace filter returned the wrong document count"
        assert demo_docs[0].metadata["workspace_id"] == "demo_workspace"
        keyword_hits = bm25_search("demo secret", demo_docs, top_k=1)
        assert keyword_hits and keyword_hits[0].metadata["workspace_id"] == "demo_workspace"
        vector_candidates = [Document(page_content="demo workspace secret: alpha", metadata={"workspace_id": "demo_workspace", "file_name": "demo.txt"})]
        fused = hybrid_retrieve(client=client, collection_name=collection, question="demo secret", workspace_id="demo_workspace", vector_candidates=vector_candidates, vector_top_k=10, bm25_top_k=10, final_top_k=5, enable_hybrid=True, enable_reranker=False, reranker_model="", siliconflow_api_key="", siliconflow_base_url="")
        assert fused, "hybrid retrieval returned no documents"
        assert all(doc.metadata["workspace_id"] == "demo_workspace" for doc in fused)
        filter_result = client.scroll(collection_name=collection, scroll_filter=workspace_filter("analyst_workspace"), limit=10, with_payload=True)[0]
        assert len(filter_result) == 1
        assert filter_result[0].payload["metadata"]["workspace_id"] == "analyst_workspace"
        print("Workspace isolation: OK")
        print("BM25 search: OK")
        print("Hybrid RRF retrieval: OK")
        return 0
    finally:
        try:
            client.delete_collection(collection)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ISOLATION TEST FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
