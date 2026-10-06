"""Local demo authentication with PBKDF2 password hashes."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any


def verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        200_000,
    ).hex()
    return hmac.compare_digest(digest, expected_hash)


def load_users(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return list(data.get("users", []))


def authenticate(
    username: str,
    password: str,
    users: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for user in users:
        if user.get("username") != username:
            continue
        if verify_password(
            password,
            str(user.get("salt", "")),
            str(user.get("password_hash", "")),
        ):
            return user
        return None
    return None
