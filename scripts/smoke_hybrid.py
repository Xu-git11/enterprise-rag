"""End-to-end smoke test for workspace-isolated hybrid retrieval."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env", override=False)

from rag_core import hybrid_retrieve, workspace_filter


def require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing {name}")
    return value


def source_label(document) -> str:
    name = document.metadata.get("file_name", "unknown")
    page = document.metadata.get("page")
    return f"{name} p.{int(page) + 1}" if page is not None else name


def main() -> int:
    workspace_id = "demo_workspace"
    collection = f"hybrid_smoke_{uuid.uuid4().hex[:8]}"
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        api_key=require("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        chunk_size=32,
        timeout=90,
        max_retries=2,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
    docs = TextLoader(str(BASE_DIR / "samples" / "company_policy.txt"), encoding="utf-8").load()
    for document in docs:
        document.metadata.update({"workspace_id": workspace_id, "user_id": "demo", "file_name": "company_policy.txt", "source": "company_policy.txt"})
    splits = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60, separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]).split_documents(docs)
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=15)
    client.create_collection(collection_name=collection, vectors_config=VectorParams(size=int(os.getenv("EMBEDDING_DIMENSION", "1024")), distance=Distance.COSINE))
    try:
        store = Qdrant(client=client, collection_name=collection, embeddings=embeddings)
        store.add_documents(splits)
        question = "正式员工入职满三年后，每年有多少天带薪年假？"
        retriever = store.as_retriever(search_kwargs={"k": 10, "filter": workspace_filter(workspace_id)})
        vector_candidates = retriever.invoke(question)
        retrieved = hybrid_retrieve(client=client, collection_name=collection, question=question, workspace_id=workspace_id, vector_candidates=vector_candidates, vector_top_k=10, bm25_top_k=10, final_top_k=5, enable_hybrid=True, enable_reranker=False, reranker_model="", siliconflow_api_key="", siliconflow_base_url="")
        assert retrieved, "no documents retrieved"
        assert all(doc.metadata.get("workspace_id") == workspace_id for doc in retrieved)
        context = "\n\n".join(doc.page_content for doc in retrieved)
        prompt = PromptTemplate(template="只依据上下文回答问题。\n上下文：\n{context}\n问题：\n{question}\n回答：", input_variables=["context", "question"])
        answer = (prompt | ChatOpenAI(model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), api_key=require("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), temperature=0, max_tokens=120, timeout=90, max_retries=2) | StrOutputParser()).invoke({"context": context, "question": question})
        print(f"Hybrid retrieval: OK ({len(retrieved)} documents)")
        print("Sources:", ", ".join(source_label(doc) for doc in retrieved))
        print("Answer:", answer)
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
        print(f"HYBRID SMOKE FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
