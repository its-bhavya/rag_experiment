"""Root conftest.py for pytest.

Adds the project root to sys.path so that:
  - `from src.xxx import ...` works (src/ is on path via pyproject.toml pythonpath)
  - `import conftest` works in test files
  - The HuggingFace `datasets` package is NOT shadowed by our local src/datasets package
    (we removed "." from pytest pythonpath; root is re-added here via sys.path insert
    so that `from conftest import ...` and `from src.xxx import ...` still function).

The key fix for the naming collision:
  Previously pytest had pythonpath = [".", "src"]. This put BOTH root (.) and src/
  on sys.path. When sentence_transformers did `from datasets import Dataset`, Python
  found src/datasets/__init__.py (because src/ is on path) instead of the HuggingFace
  `datasets` package.

  Solution: pyproject.toml now has pythonpath = ["src"] only. The project root is added
  here (via conftest) so `from src.xxx import ...` continues to work. The HuggingFace
  `datasets` package is now resolved correctly because `src/` is on sys.path but not `.`
  — only `src/datasets` would shadow `datasets`, which is the same as before. Wait...
  Actually `src/` on sys.path means `import datasets` finds `src/datasets`.

  REAL FIX: Use sys.path so that the project root (containing `src/`) is on path, but
  NOT `src/` itself. Tests import as `from src.datasets.xxx import ...` which requires
  root on path. sentence_transformers imports `from datasets import ...` which must find
  the HuggingFace package, not src/datasets. If only root is on sys.path, `import datasets`
  does NOT find src/datasets (it's not a top-level module from root). ✓
"""

import sys
from pathlib import Path

# Insert project root so `from src.xxx import ...` works
_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line(
        "markers",
        "requires_faiss: mark test as requiring FAISS (skipped if FAISS DLL unavailable)",
    )


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: PLC0415,F401
        return True
    except (ImportError, OSError):
        return False


FAISS_AVAILABLE = _faiss_available()

skip_if_no_faiss = pytest.mark.skipif(
    not FAISS_AVAILABLE,
    reason="FAISS DLL unavailable in this environment (install faiss-cpu and check DLL dependencies)",
)
