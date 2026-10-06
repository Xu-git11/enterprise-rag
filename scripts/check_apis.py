"""Check DeepSeek, SiliconFlow embedding, and Qdrant connectivity."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env", override=False)


def require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing {name} in .env")
    return value


def check_deepseek() -> None:
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=require("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
        max_tokens=8,
        timeout=60,
        max_retries=1,
    )
    response = llm.invoke("只回复 OK")
    content = getattr(response, "content", "")
    if not content:
        raise RuntimeError("DeepSeek returned an empty response")
    print("DeepSeek: OK")


def check_embedding() -> None:
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        api_key=require("SILICONFLOW_API_KEY"),
        base_url=os.getenv(
            "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
        ),
        chunk_size=32,
        timeout=60,
        max_retries=1,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
    vector = embeddings.embed_query("连接测试")
    if not vector:
        raise RuntimeError("SiliconFlow returned an empty embedding")
    print(f"SiliconFlow embedding: OK (dimension={len(vector)})")


def check_qdrant() -> None:
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        timeout=10,
    )
    client.get_collections()
    print("Qdrant: OK")


def main() -> int:
    checks = [check_deepseek, check_embedding, check_qdrant]
    failed = False
    for check in checks:
        try:
            check()
        except Exception as exc:
            failed = True
            print(f"{check.__name__}: FAILED - {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
