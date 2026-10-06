"""Check demo authentication hashes and workspace binding."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from auth import authenticate, load_users


def main() -> int:
    users = load_users(BASE_DIR / "users.example.json")
    demo = authenticate("demo", "demo123", users)
    analyst = authenticate("analyst", "analyst123", users)
    assert demo and demo["workspace_id"] == "demo_workspace"
    assert analyst and analyst["workspace_id"] == "analyst_workspace"
    assert authenticate("demo", "wrong-password", users) is None
    print("Authentication: OK")
    print("Workspace binding: OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"AUTH CHECK FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
