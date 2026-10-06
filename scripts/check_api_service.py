"""Integration test for the FastAPI RAG service."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env", override=False)
os.environ["QDRANT_COLLECTION"] = f"api_test_{uuid.uuid4().hex[:8]}"
os.environ["DEMO_API_KEY"] = os.getenv("DEMO_API_KEY", "demo-api-key")
os.environ["ANALYST_API_KEY"] = os.getenv("ANALYST_API_KEY", "analyst-api-key")
sys.path.insert(0, str(BASE_DIR))

from api import app, get_qdrant_client, QDRANT_COLLECTION


def main() -> int:
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200, health.text
    unauthorized = client.post("/v1/chat", json={"question": "test"})
    assert unauthorized.status_code == 401, unauthorized.text
    headers = {"X-API-Key": "demo-api-key"}
    with (BASE_DIR / "samples" / "company_policy.txt").open("rb") as handle:
        upload = client.post("/v1/documents", headers=headers, files={"file": ("company_policy.txt", handle, "text/plain")})
    assert upload.status_code == 200, upload.text
    assert upload.json()["chunks"] >= 2
    question = "正式员工入职满三年后，每年有多少天带薪年假？"
    response = client.post("/v1/chat", headers=headers, json={"question": question})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "15" in body["answer"]
    assert body["sources"]
    stream = client.post("/v1/chat/stream", headers=headers, json={"question": question})
    assert stream.status_code == 200, stream.text
    assert "event: sources" in stream.text
    assert "event: done" in stream.text
    print("API health: OK")
    print("API authentication: OK")
    print("API document upload: OK")
    print("API chat: OK")
    print("API stream: OK")
    try:
        get_qdrant_client().delete_collection(QDRANT_COLLECTION)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"API TEST FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
