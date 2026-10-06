"""End-to-end smoke test for the local RAG core."""

from __future__ import annotations

import os
import sys
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
load_dotenv(BASE_DIR / ".env", override=False)


def require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing {name}")
    return value


def main() -> int:
    sample = BASE_DIR / "samples" / "company_policy.txt"
    question = "正式员工入职满三年后，每年有多少天带薪年假？"

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
    documents = TextLoader(str(sample), encoding="utf-8").load()
    splits = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    ).split_documents(documents)

    collection = os.getenv("QDRANT_COLLECTION", "corrective_rag")
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
    retrieved = store.similarity_search(question, k=3)
    print(f"Qdrant ingestion: OK ({len(splits)} chunks)")
    print(f"Retrieval: OK ({len(retrieved)} documents)")

    context = "\n\n".join(document.page_content for document in retrieved)
    prompt = PromptTemplate(
        template="""只根据上下文回答问题，并给出简短依据。
上下文：
{context}
问题：
{question}
回答：""",
        input_variables=["context", "question"],
    )
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=require("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
        max_tokens=200,
        timeout=90,
        max_retries=2,
    )
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    print("DeepSeek generation: OK")
    print("ANSWER:")
    print(answer)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
