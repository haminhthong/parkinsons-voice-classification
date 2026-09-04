"""Chạy tuần tự các code cell để kiểm tra notebook mà không cần Jupyter."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NOTEBOOK = Path(__file__).with_name("02_colab_reproducible.ipynb")


def display(value) -> None:
    """Đại diện tối giản cho IPython.display trong lần kiểm tra tự động."""
    print(value)


def validate() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"display": display, "__name__": "__notebook__"}
    original_directory = Path.cwd()
    try:
        for index, cell in enumerate(notebook["cells"], start=1):
            if cell["cell_type"] != "code":
                continue
            print(f"\n--- Chạy code cell {index} ---")
            exec(compile(cell["source"], f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(original_directory)


if __name__ == "__main__":
    validate()
