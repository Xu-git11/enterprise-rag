"""Run a small repeatable RAG benchmark against the sample policy document."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
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

from rag_core import hybrid_retrieve, workspace_filter
load_dotenv(BASE_DIR / ".env", override=False)
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() in {
    "1",
    "true",
    "yes",
}
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "false").lower() in {
    "1",
    "true",
    "yes",
}
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
).strip()
OUTPUT_FILE = os.getenv("EVAL_OUTPUT", "results.json").strip()


def require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing {name}")
    return value


def source_label(document) -> str:
    name = document.metadata.get("file_name", "unknown")
    page = document.metadata.get("page")
    if page is None:
        return str(name)
    return f"{name} p.{int(page) + 1}"


def rerank_documents(question: str, documents):
    if not ENABLE_RERANKER or len(documents) <= 1:
        return documents[:RERANK_TOP_N]

    response = requests.post(
        f"{os.getenv('SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1').rstrip('/')}/rerank",
        headers={
            "Authorization": f"Bearer {require('SILICONFLOW_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": RERANKER_MODEL,
            "query": question,
            "documents": [document.page_content for document in documents],
            "return_documents": False,
            "top_n": min(RERANK_TOP_N, len(documents)),
        },
        timeout=60,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    reranked = []
    for item in results:
        index = int(item["index"])
        if 0 <= index < len(documents):
            document = documents[index]
            document.metadata["rerank_score"] = item.get("relevance_score")
            reranked.append(document)
    return reranked or documents[:RERANK_TOP_N]


def main() -> int:
    questions = json.loads(
        (BASE_DIR / "eval" / "questions.json").read_text(encoding="utf-8-sig")
    )
    documents = TextLoader(
        str(BASE_DIR / "samples" / "company_policy.txt"), encoding="utf-8"
    ).load()
    for document in documents:
        document.metadata["file_name"] = "company_policy.txt"
        document.metadata["workspace_id"] = "eval_workspace"
        document.metadata["user_id"] = "eval"

    splits = RecursiveCharacterTextSplitter(
        chunk_size=180,
        chunk_overlap=35,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    ).split_documents(documents)

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
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=require("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
        max_tokens=220,
        timeout=90,
        max_retries=2,
    )

    collection = "enterprise_rag_eval"
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        timeout=15,
    )
    try:
        client.delete_collection(collection)
    except Exception:
        pass
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
            distance=Distance.COSINE,
        ),
    )

    store = Qdrant(client=client, collection_name=collection, embeddings=embeddings)
    store.add_documents(splits)

    prompt = PromptTemplate(
        template="""只依据上下文回答问题。
回答要简短，并在关键结论后使用 [文件名 p.页码] 标注来源。
上下文：
{context}
问题：
{question}
回答：""",
        input_variables=["context", "question"],
    )
    chain = prompt | llm | StrOutputParser()

    retrieval_k = RETRIEVAL_TOP_K if ENABLE_RERANKER else RERANK_TOP_N
    records = []
    try:
        for item in questions:
            started = time.perf_counter()
            retriever = store.as_retriever(
                search_kwargs={
                    "k": retrieval_k,
                    "filter": workspace_filter("eval_workspace"),
                }
            )
            vector_candidates = retriever.invoke(item["question"])
            retrieved = hybrid_retrieve(
                client=client,
                collection_name=collection,
                question=item["question"],
                workspace_id="eval_workspace",
                vector_candidates=vector_candidates,
                vector_top_k=RETRIEVAL_TOP_K,
                bm25_top_k=RETRIEVAL_TOP_K,
                final_top_k=RERANK_TOP_N,
                enable_hybrid=ENABLE_HYBRID_SEARCH,
                enable_reranker=ENABLE_RERANKER,
                reranker_model=RERANKER_MODEL,
                siliconflow_api_key=require("SILICONFLOW_API_KEY"),
                siliconflow_base_url=os.getenv(
                    "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
                ),
            )
            context = "\n\n".join(document.page_content for document in retrieved)
            answer = chain.invoke(
                {"context": context, "question": item["question"]}
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            compact_answer = re.sub(r"\s+", "", answer)
            keyword_hit = any(
                re.sub(r"\s+", "", keyword) in compact_answer
                for keyword in item["expected_any"]
            )
            citation_present = bool(
                re.search(r"p\.\s*\d+", answer, flags=re.IGNORECASE)
            )
            records.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": answer,
                    "expected_any": item["expected_any"],
                    "keyword_hit": keyword_hit,
                    "citation_present": citation_present,
                    "latency_ms": latency_ms,
                    "rerank_scores": [
                        document.metadata.get("rerank_score")
                        for document in retrieved
                    ],
                    "sources": list(
                        dict.fromkeys(source_label(document) for document in retrieved)
                    ),
                }
            )
    finally:
        try:
            client.delete_collection(collection)
        except Exception:
            pass

    total = len(records)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "eval/questions.json",
        "sample_document": "samples/company_policy.txt",
        "reranker_enabled": ENABLE_RERANKER,
        "hybrid_enabled": ENABLE_HYBRID_SEARCH,
        "retrieval_top_k": retrieval_k,
        "rerank_top_n": RERANK_TOP_N,
        "chunk_count": len(splits),
        "total_questions": total,
        "keyword_accuracy": round(
            sum(1 for record in records if record["keyword_hit"]) / total, 4
        ),
        "citation_rate": round(
            sum(1 for record in records if record["citation_present"]) / total, 4
        ),
        "average_latency_ms": round(
            sum(record["latency_ms"] for record in records) / total, 2
        ),
        "records": records,
    }
    result_path = BASE_DIR / "eval" / OUTPUT_FILE
    result_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if summary["keyword_accuracy"] >= 0.75 else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"EVALUATION FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
