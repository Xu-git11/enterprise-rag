"""Check optional Langfuse configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env", override=False)


def enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


def main() -> int:
    if not enabled(os.getenv("LANGFUSE_ENABLED")):
        print("Langfuse: DISABLED (optional)")
        return 0
    try:
        from langfuse import Langfuse
    except ImportError as exc:
        raise RuntimeError("langfuse package is not installed") from exc
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()
    if not public_key or not secret_key:
        raise RuntimeError("LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is missing")
    client = Langfuse(public_key=public_key, secret_key=secret_key, host=host, enabled=True)
    auth_check = getattr(client, "auth_check", None)
    if callable(auth_check):
        auth_check()
    client.flush()
    print("Langfuse: OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"LANGFUSE CHECK FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
