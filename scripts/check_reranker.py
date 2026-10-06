"""Check SiliconFlow reranker connectivity."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env", override=False)


def main() -> int:
    key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing SILICONFLOW_API_KEY")
    base_url = os.getenv(
        "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
    ).rstrip("/")
    response = requests.post(
        f"{base_url}/rerank",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": os.getenv(
                "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
            ),
            "query": "年假有多少天？",
            "documents": [
                "正式员工入职满三年后，每年享有15天带薪年假。",
                "员工每周可以申请2天远程办公。",
                "发现数据泄露风险时，应在30分钟内通知信息安全团队。",
            ],
            "return_documents": False,
            "top_n": 2,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("results", [])
    if not results:
        raise RuntimeError("Reranker returned no results")
    print("SiliconFlow reranker: OK")
    for item in results:
        print(f"index={item['index']} score={item['relevance_score']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"RERANK CHECK FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
