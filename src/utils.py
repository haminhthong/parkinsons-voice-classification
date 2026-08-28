"""Tiện ích dùng chung cho siêu dữ liệu và khả năng tái lập."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Tạo mã kiểm tra SHA-256 để định danh chính xác phiên bản tệp."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
