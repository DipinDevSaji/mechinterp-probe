"""Repo-local shim so `python -m pytest` ignores global plugin autoload."""

import os
import sys
from pathlib import Path


os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

project_root = Path(__file__).resolve().parent
sys.path = [
    path
    for path in sys.path
    if Path(path or os.getcwd()).resolve() != project_root
]

from _pytest.config import main

raise SystemExit(main())
