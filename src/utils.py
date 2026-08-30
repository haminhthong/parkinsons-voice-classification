"""Module tiện ích hệ thống và kiểm tra khả năng tái lập.

Cung cấp hàm tính mã băm SHA-256 của tệp để định danh chính xác phiên bản dữ liệu đầu vào.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Tính toán mã kiểm tra băm SHA-256 của một tệp tin.

    Args:
        path: Đường dẫn tệp tin cần tính hash.

    Returns:
        str: Chuỗi 64 ký tự hex đại diện cho mã SHA-256 của tệp.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

